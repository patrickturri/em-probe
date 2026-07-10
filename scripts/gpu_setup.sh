#!/usr/bin/env bash
# Bootstrap a fresh rented GPU box (RunPod/Lambda, Ubuntu + CUDA image).
# Run from the repo root after cloning/rsyncing the repo onto the box.
set -euo pipefail

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync
bash scripts/fetch_data.sh

nvidia-smi --query-gpu=name,memory.total --format=csv
[ -n "${ANTHROPIC_API_KEY:-}" ] || [ -f .env ] || echo "WARNING: no ANTHROPIC_API_KEY and no .env — judging will fail"
echo "ready. e.g.: make finetune CONFIG=configs/qwen7b_medical.yaml"
