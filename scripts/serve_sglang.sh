#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/sglang.qwen3-0.6b.yaml}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

exec uv run --extra cuda inference-lab serve \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  "$@"
