#!/usr/bin/env bash
# Run ON the Vast instance — no subir pesos base, solo código + adapter LoRA pequeño.
#
# Base Olmo/Qwen: HuggingFace (Unsloth 4-bit), NO Ollama.
# Ollama sirve inferencia; train_rl.py usa Unsloth+PEFT.
#
# Usage (on Vast, after tiny code upload):
#   cd /workspace/regllm && bash training/vast_rl_setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo ">>> DNS"
echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" | tee /etc/resolv.conf >/dev/null

echo ">>> Python env"
source /venv/main/bin/activate

echo ">>> deps (HF pulls base model on first train — not via Ollama)"
uv pip install -q unsloth trl peft datasets accelerate bitsandbytes pyyaml httpx transformers

echo ">>> disk"
df -h / /workspace 2>/dev/null || df -h /

ADAPTER="${ROOT}/training/output/sft-olmo3-7b"
ADAPTER_HF="${ADAPTER_HF:-}"   # e.g. diego/regllm-olmo3-7b-sft

if [[ -n "${ADAPTER_HF}" ]]; then
  echo ">>> LoRA adapter from HuggingFace: ${ADAPTER_HF}"
  mkdir -p "${ADAPTER}"
  huggingface-cli download "${ADAPTER_HF}" --local-dir "${ADAPTER}" --local-dir-use-symlinks False
fi

if [[ ! -f "${ADAPTER}/adapter_model.safetensors" ]]; then
  echo ""
  echo "FALTA el adapter LoRA (~305 MB). Opciones:"
  echo "  A) HuggingFace (recomendado, sin subir pesos base):"
  echo "     # en tu PC: huggingface-cli upload TU_USER/regllm-olmo3-7b-sft training/output/sft-olmo3-7b"
  echo "     ADAPTER_HF=TU_USER/regllm-olmo3-7b-sft bash training/vast_rl_setup.sh"
  echo ""
  echo "  B) Solo el adapter por scp (~305 MB, NO el Olmo base):"
  echo "     scp -P <PORT> training/output/sft-olmo3-7b/adapter_model.safetensors \\"
  echo "         training/output/sft-olmo3-7b/adapter_config.json \\"
  echo "         training/output/sft-olmo3-7b/regllm_meta.json \\"
  echo "         root@<IP>:${ADAPTER}/"
  echo ""
  echo "  Base Olmo/Qwen NO se sube: Unsloth los baja de HuggingFace al entrenar."
  echo "  Ollama NO sirve para GRPO (solo inferencia)."
  exit 1
fi

echo ">>> adapter OK: $(du -sh "${ADAPTER}" | cut -f1)"
echo ">>> launch GRPO (base Olmo se descarga de HF ~4-5 GB en cache)"
mkdir -p training/output/grpo-vast training/output/rl_sessions

nohup python training/train_rl.py \
  --base-adapter "${ADAPTER}" \
  --output training/output/grpo-vast \
  --batch-size 12 \
  --num-generations 8 \
  --max-completion-length 512 \
  --max-minutes 180 \
  --adaptive-bugs \
  --session-name vast_3090_olmo \
  > training/output/grpo-vast/train.log 2>&1 &

echo "PID=$!  log=training/output/grpo-vast/train.log"
echo "tail -f training/output/grpo-vast/train.log"
