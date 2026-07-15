#!/usr/bin/env bash
#
# frontend_deploy.sh — Publish the static frontend to an S3 static-website bucket
# and point it at the live Lambda Function URL.
#
# What it does:
#   1. Create the frontend bucket (if missing).
#   2. Make it publicly readable (a static website must be public) — disable the
#      bucket's public-access block and attach a public s3:GetObject policy.
#   3. Enable static website hosting (index document = index.html).
#   4. Inject the live Function URL into index.html (rewrites window.API_BASE_URL)
#      in a temp copy, then sync that copy to the bucket.
#   5. Print the website URL.
#
# Usage:
#   ./frontend_deploy.sh
#   ./frontend_deploy.sh --help
#
# CORS: the API's CORS is handled by the app via its CORS_ORIGINS env var, which
# deploy.sh already set to this bucket's website origin — so they match by
# construction. This script does NOT modify the Lambda (updating a single env var
# would replace the whole env map). Verify CORS live in the browser.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration — override via the environment if needed.
# --------------------------------------------------------------------------- #
AWS_REGION="${AWS_REGION:-ca-central-1}"
FRONTEND_BUCKET="${FRONTEND_BUCKET:-lab08-frontend-maxencebernardhub}"
LAMBDA_NAME="${LAMBDA_NAME:-lab08-commercial-agent}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="${PROJECT_DIR}/frontend"

# Website endpoint (ca-central-1 uses the dot-form). Must equal the API's
# CORS_ORIGINS value set by deploy.sh.
WEBSITE_ORIGIN="http://${FRONTEND_BUCKET}.s3-website.${AWS_REGION}.amazonaws.com"

log()  { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
[ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] && { usage; exit 0; }

command -v aws >/dev/null || die "aws CLI not found"
[ -d "$FRONTEND_DIR" ]     || die "frontend/ not found at ${FRONTEND_DIR}"

# --------------------------------------------------------------------------- #
# Resolve the live Function URL from the deployed Lambda.
# --------------------------------------------------------------------------- #
log "Resolving the Function URL from Lambda '${LAMBDA_NAME}'"
FUNCTION_URL="$(aws lambda get-function-url-config --function-name "$LAMBDA_NAME" \
  --region "$AWS_REGION" --query FunctionUrl --output text 2>/dev/null || true)"
[ -n "$FUNCTION_URL" ] && [ "$FUNCTION_URL" != "None" ] \
  || die "No Function URL found — run ./deploy.sh url first"
# Strip a single trailing slash for a clean injected value (app.js also strips).
API_BASE="${FUNCTION_URL%/}"
ok "API base: ${API_BASE}"

# --------------------------------------------------------------------------- #
# 1. Create the bucket (if missing).
# --------------------------------------------------------------------------- #
log "S3: frontend bucket '${FRONTEND_BUCKET}'"
if aws s3api head-bucket --bucket "$FRONTEND_BUCKET" 2>/dev/null; then
  ok "Bucket already exists"
else
  aws s3api create-bucket --bucket "$FRONTEND_BUCKET" \
    --region "$AWS_REGION" \
    --create-bucket-configuration "LocationConstraint=${AWS_REGION}" >/dev/null
  ok "Created bucket"
fi

# --------------------------------------------------------------------------- #
# 2. Make it publicly readable (static website).
# --------------------------------------------------------------------------- #
log "Allowing public read (static website)"
# Disable the block-public-access flags that would otherwise reject a public
# bucket policy.
aws s3api put-public-access-block --bucket "$FRONTEND_BUCKET" \
  --public-access-block-configuration \
    BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false
ok "Public-access block disabled on this bucket"

# Public read-only policy for the website objects.
POLICY_JSON="$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadForWebsite",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${FRONTEND_BUCKET}/*"
    }
  ]
}
JSON
)"
aws s3api put-bucket-policy --bucket "$FRONTEND_BUCKET" --policy "$POLICY_JSON"
ok "Public read policy attached"

# --------------------------------------------------------------------------- #
# 3. Enable static website hosting.
# --------------------------------------------------------------------------- #
log "Enabling static website hosting"
# index.html serves both the index and (404 →) the app, since it's a one-page app.
aws s3 website "s3://${FRONTEND_BUCKET}/" \
  --index-document index.html --error-document index.html
ok "Website hosting enabled"

# --------------------------------------------------------------------------- #
# 4. Inject the Function URL and sync.
# --------------------------------------------------------------------------- #
log "Injecting API base URL and syncing files"
TMP="$(mktemp -d)"
# shellcheck disable=SC2064
trap "rm -rf '${TMP}'" EXIT
cp -R "${FRONTEND_DIR}/." "${TMP}/"

# Rewrite window.API_BASE_URL = "..."; to the live Function URL.
python3 - "$TMP/index.html" "$API_BASE" <<'PY'
import re, sys
path, api_base = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    html = f.read()
new, n = re.subn(
    r'window\.API_BASE_URL\s*=\s*"[^"]*";',
    f'window.API_BASE_URL = "{api_base}";',
    html,
)
if n != 1:
    sys.exit(f"expected exactly 1 API_BASE_URL assignment, replaced {n}")
with open(path, "w", encoding="utf-8") as f:
    f.write(new)
PY
ok "index.html now targets ${API_BASE}"

aws s3 sync "$TMP" "s3://${FRONTEND_BUCKET}/" --delete \
  --exclude ".*" --exclude "*/.*"
ok "Files synced"

# --------------------------------------------------------------------------- #
# 5. Done.
# --------------------------------------------------------------------------- #
echo
ok "Frontend deployed"
echo "    Website URL : ${WEBSITE_ORIGIN}"
echo "    API base    : ${API_BASE}"
echo
echo "    Reminder: the API's CORS_ORIGINS must equal ${WEBSITE_ORIGIN}"
echo "    (deploy.sh set it to this by construction). Verify in the browser."
