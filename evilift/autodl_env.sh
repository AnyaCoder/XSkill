#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/../.env.local}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "AutoDL environment file not found: $ENV_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${AUTODL_BASE_URL:?AUTODL_BASE_URL is required}"
: "${AUTODL_API_KEY:?AUTODL_API_KEY is required}"

export REASONING_MODEL_NAME="${REASONING_MODEL_NAME:-Qwen3.5-397B-A17B}"
export REASONING_API_KEY="$AUTODL_API_KEY"
export REASONING_END_POINT="${AUTODL_BASE_URL%/}/chat/completions"

export EXPERIENCE_MODEL_NAME="${EXPERIENCE_MODEL_NAME:-$REASONING_MODEL_NAME}"
export EXPERIENCE_API_KEY="$AUTODL_API_KEY"
export EXPERIENCE_END_POINT="$REASONING_END_POINT"

export VERIFIER_MODEL_NAME="${VERIFIER_MODEL_NAME:-$REASONING_MODEL_NAME}"
export VERIFIER_API_KEY="$AUTODL_API_KEY"
export VERIFIER_END_POINT="$AUTODL_BASE_URL"

export ENABLE_FUNCTION_CALLING="true"
export ENABLED_TOOLS="zoom"

