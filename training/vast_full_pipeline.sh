#!/usr/bin/env bash
# Qwen3-8B on Vast: CPT → SFT → GRPO via Unsloth 4-bit (HF streaming, no git clone).
set -euo pipefail

cd /workspace/regllm
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PIP_NO_CACHE_DIR=1

echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" > /etc/resolv.conf

source /venv/main/bin/activate
uv pip install -q unsloth trl peft datasets accelerate bitsandbytes pyyaml httpx scipy 2>/dev/null \
  || pip install -q unsloth trl peft datasets accelerate bitsandbytes pyyaml httpx scipy

mkdir -p training/output training/logs

LOG="training/logs/qwen3_full_pipeline_$(date +%Y%m%d_%H%M).log"
echo "=== Pipeline log: $LOG ==="

nohup python training/run_qwen3_full_pipeline.py \
  --config training/config_qwen3_8b_vast.yaml \
  --max-iters 3 \
  --session-minutes 90 \
  --reset-curriculum \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "tail -f $LOG"
