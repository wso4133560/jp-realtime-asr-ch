#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$ROOT_DIR/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$ROOT_DIR/venv/lib/python3.10/site-packages/nvidia/cublas/lib:$ROOT_DIR/venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

exec "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/benchmark_reazonspeech_k2_cuda.py" "$@"
