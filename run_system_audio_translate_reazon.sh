#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_ARGS=(
  --device "${REAZON_DEVICE:-cuda}"
  --translate-model "${TRANSLATE_MODEL:-translategemma:4b-it-q4_K_M}"
)

exec "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/system_audio_translate_reazon.py" "${DEFAULT_ARGS[@]}" "$@"
