#!/usr/bin/env bash
# Qwen3-8B full pipeline on Vast GPU (SFT → iterate_rl GRPO on rl_env).
set -euo pipefail

cd /workspace/regllm 2>/dev/null || cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"

# DNS fix on some Vast hosts
grep -q '8.8.8.8' /etc/resolv.conf 2>/dev/null || echo -e 'nameserver 8.8.8.8\nnameserver 1.1.1.1' >> /etc/resolv.conf || true

PY="${PY:-python3}"
if [[ -x .venv/bin/python ]]; then PY=".venv/bin/python"; fi

echo "=== Installing deps ==="
$PY -m pip install -q --upgrade pip
$PY -m pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
  trl peft accelerate bitsandbytes datasets httpx pyyaml scipy pandas openpyxl 2>/dev/null || \
$PY -m pip install -q unsloth trl peft accelerate bitsandbytes datasets httpx pyyaml scipy pandas openpyxl

mkdir -p training/output training/logs

LOG="training/logs/qwen3_pipeline_$(date +%Y%m%d_%H%M).log"
echo "=== Qwen3-8B pipeline → $LOG ==="

nohup $PY training/run_qwen3_rl_pipeline.py \
  --epochs 3 \
  --max-iters 3 \
  --session-minutes 90 \
  --reset-curriculum \
  > "$LOG" 2>&1 &

echo "PID=$!  tail -f $LOG"
