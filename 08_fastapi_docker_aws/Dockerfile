# syntax=docker/dockerfile:1

# Multi-stage build: a uv-based builder resolves the locked dependencies into a
# self-contained virtualenv, and a slim non-root runtime carries only that venv
# plus the app. The AWS Lambda Web Adapter binary is bundled so the *same* image
# runs locally (Docker Compose) and on AWS Lambda (Function URL) with no code
# path differences — the adapter is inert unless the Lambda runtime is present.

# --------------------------------------------------------------------------- #
# Stage 1 — builder: install dependencies with uv into /app/.venv
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS builder

# Pinned uv (matches the version that produced uv.lock) for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install the locked runtime dependencies only (no dev group, no project — this
# service is packaged as `package = false`). Kept as its own layer so it is
# cached until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --------------------------------------------------------------------------- #
# Stage 2 — runtime: slim, non-root, with the Lambda Web Adapter
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS runtime

# AWS Lambda Web Adapter (pinned). On Lambda it translates Function URL events
# into local HTTP against uvicorn (native response streaming for our SSE route);
# outside Lambda it does nothing, so local runs are unaffected.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 \
    /lambda-adapter /opt/extensions/lambda-adapter

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The adapter proxies to the app on this port and gates readiness on /health.
    AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /app

# The resolved virtualenv from the builder, then the application code. Only what
# the runtime needs (app + static frontend + seed data); tests/docs/deploy are
# excluded via .dockerignore.
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY frontend ./frontend
COPY data ./data

# Drop privileges: run as a non-root system user that owns the app directory.
RUN useradd --no-create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Same entry point locally and on Lambda: uvicorn serving the ASGI app.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
