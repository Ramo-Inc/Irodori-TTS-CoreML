#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IRODORI_TTS_HOST="${IRODORI_TTS_HOST:-0.0.0.0}"
IRODORI_TTS_PORT="${IRODORI_TTS_PORT:-19841}"

cd "$SCRIPT_DIR"
exec uv run python openai_api_server.py \
  --host "$IRODORI_TTS_HOST" \
  --port "$IRODORI_TTS_PORT" \
  "$@"
