#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IRODORI_TTS_HOST="${IRODORI_TTS_HOST:-0.0.0.0}"
IRODORI_TTS_PORT="${IRODORI_TTS_PORT:-19841}"

cd "$SCRIPT_DIR"
exec uv run python openai_api_server.py \
  --host "$IRODORI_TTS_HOST" \
  --port "$IRODORI_TTS_PORT" \
  --preload \
  --strict-coreml \
  --max-seconds 70 \
  --chars-per-second 5.5 \
  --max-resident-speaker-kv-buckets 5 \
  --default-condition-cache-prepare-text "起動時の短い音声合成ウォームアップです。" \
  --warmup-bucket S=256,T=32,R=160 \
  --warmup-bucket S=512,T=64,R=160 \
  --warmup-bucket S=1024,T=128,R=160 \
  --warmup-bucket S=1536,T=192,R=160 \
  --warmup-bucket S=2048,T=256,R=160 \
  "$@"
