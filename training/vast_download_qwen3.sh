#!/usr/bin/env bash
# Pre-download Qwen3-8B (Unsloth 4-bit) on Vast for SFT/GRPO training.
set -euo pipefail

echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 1.1.1.1" >> /etc/resolv.conf

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" /workspace/models

MODEL="unsloth/Qwen3-8B-Instruct-bnb-4bit"
LOCAL="/workspace/models/qwen3-8b-instruct-bnb-4bit"
LOG="/workspace/regllm/training/logs/download_qwen3_8b.log"
mkdir -p "$(dirname "$LOG")"

source /venv/main/bin/activate
uv pip install -q huggingface_hub hf_transfer 2>/dev/null || pip install -q huggingface_hub hf_transfer

echo "=== Download $MODEL ===" | tee "$LOG"
echo "HF_HOME=$HF_HOME" | tee -a "$LOG"
df -h / | tee -a "$LOG"

huggingface-cli download "$MODEL" \
  --local-dir "$LOCAL" \
  --local-dir-use-symlinks False \
  2>&1 | tee -a "$LOG"

echo "=== Done ===" | tee -a "$LOG"
du -sh "$LOCAL" "$HF_HOME" 2>/dev/null | tee -a "$LOG"
ls -la "$LOCAL" | head -20 | tee -a "$LOG"
