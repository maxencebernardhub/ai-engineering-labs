#!/usr/bin/env bash
#
# deploy.sh — Build and deploy the lab-08 commercial-agent API to AWS Lambda.
#
# Pipeline (each step is an idempotent, independently runnable stage):
#
#   preflight    Check tooling + credentials, derive account/region/image URI.
#   ecr          Create the ECR repo, build the arm64 image, push it.
#   iam          Create the Lambda execution role (least-privilege S3 + logs).
#   s3           Create the private leads bucket.
#   lambda       Create/update the Lambda function from the container image.
#   url          Create the public Function URL (HTTPS, response streaming).
#   concurrency  Cap the function with reserved concurrency (cost guardrail).
#
# Usage:
#   ./deploy.sh                 # run the whole pipeline (all stages, in order)
#   ./deploy.sh <stage>         # run a single stage (see list above)
#   ./deploy.sh --help
#
# Design notes:
#   * BYOK is enforced in the cloud: NO LLM keys are ever set on the function.
#   * SSE streaming needs the Function URL in RESPONSE_STREAM mode AND the
#     Lambda Web Adapter told to stream (AWS_LWA_INVOKE_MODE=response_stream).
#   * CORS is handled by the app (CORS_ORIGINS env), NOT by the Function URL,
#     to avoid duplicated CORS headers.
#   * Re-running any stage is safe (create-or-update / check-then-create).

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration — override any of these via the environment if needed.
# --------------------------------------------------------------------------- #
AWS_REGION="${AWS_REGION:-ca-central-1}"

ECR_REPO="${ECR_REPO:-lab08-commercial-agent}"
LAMBDA_NAME="${LAMBDA_NAME:-lab08-commercial-agent}"
LAMBDA_ROLE="${LAMBDA_ROLE:-lab08-lambda-role}"
LEADS_BUCKET="${LEADS_BUCKET:-lab08-leads-maxencebernardhub}"
FRONTEND_BUCKET="${FRONTEND_BUCKET:-lab08-frontend-maxencebernardhub}"

LAMBDA_ARCH="${LAMBDA_ARCH:-arm64}"          # Graviton — native to Apple Silicon
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/arm64}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

LAMBDA_MEMORY="${LAMBDA_MEMORY:-2048}"       # MB — more memory ⇒ faster cold start
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-120}"      # seconds
RESERVED_CONCURRENCY="${RESERVED_CONCURRENCY:-5}"

# The S3 static-website origin the browser will call the API from. ca-central-1
# uses the dot-form website endpoint. Handled by the app's CORSMiddleware.
FRONTEND_ORIGIN="${FRONTEND_ORIGIN:-http://${FRONTEND_BUCKET}.s3-website.${AWS_REGION}.amazonaws.com}"

# Paths — this script lives in <lab>/deploy/, the build context is <lab>/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Populated by preflight.
ACCOUNT_ID=""
IMAGE_URI=""
ROLE_ARN=""

# --------------------------------------------------------------------------- #
# Small logging helpers.
# --------------------------------------------------------------------------- #
log()  { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# preflight — verify tooling + credentials, compute derived values.
# --------------------------------------------------------------------------- #
preflight() {
  log "Preflight checks"
  command -v aws >/dev/null    || die "aws CLI not found (brew install awscli)"
  command -v docker >/dev/null || die "docker not found"

  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)" \
    || die "Not authenticated — run 'aws configure' first"
  IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
  ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE}"

  ok "Account ${ACCOUNT_ID} · region ${AWS_REGION} · arch ${LAMBDA_ARCH}"
  echo "    image : ${IMAGE_URI}"
  echo "    role  : ${ROLE_ARN}"
  echo "    leads bucket    : ${LEADS_BUCKET} (private)"
  echo "    frontend origin : ${FRONTEND_ORIGIN}"
}

# --------------------------------------------------------------------------- #
# ecr — repo + build (arm64) + push.
# --------------------------------------------------------------------------- #
stage_ecr() {
  log "ECR: repository, build, push"
  docker info >/dev/null 2>&1 || die "Docker daemon is not running — start Docker Desktop"

  if aws ecr describe-repositories --repository-names "$ECR_REPO" \
        --region "$AWS_REGION" >/dev/null 2>&1; then
    ok "ECR repo '${ECR_REPO}' already exists"
  else
    aws ecr create-repository --repository-name "$ECR_REPO" \
      --region "$AWS_REGION" \
      --image-scanning-configuration scanOnPush=true >/dev/null
    ok "Created ECR repo '${ECR_REPO}'"
  fi

  log "Authenticating Docker to ECR"
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" >/dev/null
  ok "Docker logged in to ECR"

  log "Building image for ${DOCKER_PLATFORM} (this pulls deps on first run)"
  # --provenance=false: skip the buildx attestation/manifest-list wrapper. Lambda
  # rejects the OCI image index that buildx emits by default ("media type ... is
  # not supported"); it needs a plain single-arch image manifest.
  docker build --platform "$DOCKER_PLATFORM" \
    --provenance=false \
    -t "$IMAGE_URI" \
    -f "${PROJECT_DIR}/Dockerfile" \
    "$PROJECT_DIR"

  log "Pushing ${IMAGE_URI}"
  docker push "$IMAGE_URI"
  ok "Image pushed"
}

# --------------------------------------------------------------------------- #
# iam — Lambda execution role: assume-by-Lambda + logs + least-privilege S3.
# --------------------------------------------------------------------------- #
stage_iam() {
  log "IAM: execution role '${LAMBDA_ROLE}'"
  local tmp trust policy
  tmp="$(mktemp -d)"
  trust="${tmp}/trust.json"
  policy="${tmp}/s3-leads.json"
  # shellcheck disable=SC2064
  trap "rm -rf '${tmp}'" RETURN

  cat >"$trust" <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "lambda.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON

  # Least privilege: read/write the leads object, plus ListBucket on the bucket
  # itself. ListBucket is required so that GetObject on a *missing* key returns
  # 404 NoSuchKey (which the S3 store treats as "empty") instead of 403
  # AccessDenied — S3 hides object existence from callers that cannot list the
  # bucket, which otherwise crashes the seed-on-boot path.
  cat >"$policy" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::${LEADS_BUCKET}/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::${LEADS_BUCKET}"
    }
  ]
}
JSON

  if aws iam get-role --role-name "$LAMBDA_ROLE" >/dev/null 2>&1; then
    ok "Role '${LAMBDA_ROLE}' already exists"
  else
    aws iam create-role --role-name "$LAMBDA_ROLE" \
      --assume-role-policy-document "file://${trust}" \
      --description "Execution role for the lab-08 commercial-agent Lambda" >/dev/null
    ok "Created role '${LAMBDA_ROLE}'"
  fi

  # Managed policy for CloudWatch Logs (create log group/stream, put events).
  aws iam attach-role-policy --role-name "$LAMBDA_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  ok "Attached AWSLambdaBasicExecutionRole (CloudWatch Logs)"

  # Inline least-privilege S3 policy.
  aws iam put-role-policy --role-name "$LAMBDA_ROLE" \
    --policy-name lab08-s3-leads-rw \
    --policy-document "file://${policy}"
  ok "Put inline policy 'lab08-s3-leads-rw' (GetObject/PutObject on ${LEADS_BUCKET})"
}

# --------------------------------------------------------------------------- #
# s3 — private leads bucket (public access fully blocked).
# --------------------------------------------------------------------------- #
stage_s3() {
  log "S3: private leads bucket '${LEADS_BUCKET}'"
  if aws s3api head-bucket --bucket "$LEADS_BUCKET" 2>/dev/null; then
    ok "Bucket '${LEADS_BUCKET}' already exists"
  else
    aws s3api create-bucket --bucket "$LEADS_BUCKET" \
      --region "$AWS_REGION" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}" >/dev/null
    ok "Created bucket '${LEADS_BUCKET}'"
  fi

  aws s3api put-public-access-block --bucket "$LEADS_BUCKET" \
    --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  ok "Public access fully blocked on '${LEADS_BUCKET}'"
}

# --------------------------------------------------------------------------- #
# lambda — create or update the container-image function.
# --------------------------------------------------------------------------- #
stage_lambda() {
  log "Lambda: function '${LAMBDA_NAME}'"
  local env_vars
  env_vars="Variables={LEAD_STORE=s3,LEADS_BUCKET=${LEADS_BUCKET},AWS_LWA_INVOKE_MODE=response_stream,CORS_ORIGINS=${FRONTEND_ORIGIN}}"

  if aws lambda get-function --function-name "$LAMBDA_NAME" \
        --region "$AWS_REGION" >/dev/null 2>&1; then
    log "Updating existing function code + configuration"
    aws lambda update-function-code --function-name "$LAMBDA_NAME" \
      --image-uri "$IMAGE_URI" --region "$AWS_REGION" >/dev/null
    aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$AWS_REGION"
    aws lambda update-function-configuration --function-name "$LAMBDA_NAME" \
      --memory-size "$LAMBDA_MEMORY" --timeout "$LAMBDA_TIMEOUT" \
      --environment "$env_vars" --region "$AWS_REGION" >/dev/null
  else
    log "Creating function from image (role must have propagated — retrying)"
    # New roles take a few seconds to become assumable by Lambda; retry briefly.
    local attempt
    for attempt in 1 2 3 4 5 6; do
      if aws lambda create-function --function-name "$LAMBDA_NAME" \
            --package-type Image \
            --code "ImageUri=${IMAGE_URI}" \
            --role "$ROLE_ARN" \
            --architectures "$LAMBDA_ARCH" \
            --memory-size "$LAMBDA_MEMORY" \
            --timeout "$LAMBDA_TIMEOUT" \
            --environment "$env_vars" \
            --region "$AWS_REGION" >/dev/null 2>&1; then
        break
      fi
      warn "create-function attempt ${attempt} failed (role propagation?) — retrying in 10s"
      sleep 10
      [ "$attempt" -eq 6 ] && die "create-function still failing after retries"
    done
  fi

  aws lambda wait function-active-v2 --function-name "$LAMBDA_NAME" --region "$AWS_REGION"
  ok "Function '${LAMBDA_NAME}' is active"
}

# --------------------------------------------------------------------------- #
# url — public Function URL with HTTPS + response streaming.
# --------------------------------------------------------------------------- #
stage_url() {
  log "Function URL: public, RESPONSE_STREAM"
  if aws lambda get-function-url-config --function-name "$LAMBDA_NAME" \
        --region "$AWS_REGION" >/dev/null 2>&1; then
    aws lambda update-function-url-config --function-name "$LAMBDA_NAME" \
      --auth-type NONE --invoke-mode RESPONSE_STREAM --region "$AWS_REGION" >/dev/null
    ok "Updated Function URL config"
  else
    aws lambda create-function-url-config --function-name "$LAMBDA_NAME" \
      --auth-type NONE --invoke-mode RESPONSE_STREAM --region "$AWS_REGION" >/dev/null
    ok "Created Function URL config"
  fi

  # Public invoke permissions for the Function URL (idempotent). Since October
  # 2025, a public (auth-type NONE) Function URL requires BOTH actions in the
  # resource policy: lambda:InvokeFunctionUrl AND lambda:InvokeFunction. Missing
  # the latter yields a 403 "Forbidden" even though the URL config is correct.
  # The FunctionUrlAuthType condition is only valid for InvokeFunctionUrl, so the
  # InvokeFunction grant is added as its own unconditioned statement.
  aws lambda add-permission --function-name "$LAMBDA_NAME" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal '*' \
    --function-url-auth-type NONE \
    --region "$AWS_REGION" >/dev/null 2>&1 \
    && ok "Granted public lambda:InvokeFunctionUrl" \
    || ok "lambda:InvokeFunctionUrl permission already present"

  aws lambda add-permission --function-name "$LAMBDA_NAME" \
    --statement-id FunctionURLAllowPublicInvoke \
    --action lambda:InvokeFunction \
    --principal '*' \
    --region "$AWS_REGION" >/dev/null 2>&1 \
    && ok "Granted public lambda:InvokeFunction" \
    || ok "lambda:InvokeFunction permission already present"

  local url
  url="$(aws lambda get-function-url-config --function-name "$LAMBDA_NAME" \
          --region "$AWS_REGION" --query FunctionUrl --output text)"
  ok "Function URL: ${url}"
}

# --------------------------------------------------------------------------- #
# concurrency — reserved concurrency cap (non-fatal: new accounts may refuse).
# --------------------------------------------------------------------------- #
stage_concurrency() {
  log "Reserved concurrency: ${RESERVED_CONCURRENCY}"
  local limit
  limit="$(aws lambda get-account-settings \
            --query 'AccountLimit.ConcurrentExecutions' --output text \
            --region "$AWS_REGION" 2>/dev/null || echo '?')"
  echo "    account concurrency limit: ${limit}"

  if aws lambda put-function-concurrency --function-name "$LAMBDA_NAME" \
        --reserved-concurrent-executions "$RESERVED_CONCURRENCY" \
        --region "$AWS_REGION" >/dev/null 2>&1; then
    ok "Reserved concurrency set to ${RESERVED_CONCURRENCY}"
  else
    warn "Could not reserve concurrency (account limit '${limit}' likely too low)."
    warn "Not fatal: the account-level limit already caps total concurrency (cost)."
  fi
}

# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
summary() {
  local url
  url="$(aws lambda get-function-url-config --function-name "$LAMBDA_NAME" \
          --region "$AWS_REGION" --query FunctionUrl --output text 2>/dev/null || echo '(none)')"
  echo
  ok "Deployment complete"
  echo "    API base URL : ${url}"
  echo "    Try it       : curl ${url}health"
}

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
main() {
  case "${1:-all}" in
    -h|--help)   usage ;;
    preflight)   preflight ;;
    ecr)         preflight; stage_ecr ;;
    iam)         preflight; stage_iam ;;
    s3)          preflight; stage_s3 ;;
    lambda)      preflight; stage_lambda ;;
    url)         preflight; stage_url ;;
    concurrency) preflight; stage_concurrency ;;
    all)
      preflight
      stage_ecr
      stage_iam
      stage_s3
      stage_lambda
      stage_url
      stage_concurrency
      summary
      ;;
    *) die "Unknown stage '$1' — run './deploy.sh --help'" ;;
  esac
}

main "$@"
