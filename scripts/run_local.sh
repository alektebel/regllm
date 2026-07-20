#!/usr/bin/env bash
# Run the DQC app locally: FastAPI backend + Angular dev server, with the
# LLM served by AWS Bedrock (Nova by default) through your local AWS
# credentials (aws configure / SSO / env vars — the standard boto3 chain).
#
#   ./scripts/run_local.sh                          # Bedrock Nova Micro
#   BEDROCK_MODEL_ID=eu.anthropic.claude-3-5-haiku-20241022-v1:0 \
#       ./scripts/run_local.sh                      # another Bedrock model
#   REGLLM_LLM=ollama ./scripts/run_local.sh        # local Ollama instead
#   REGLLM_LLM=stub   ./scripts/run_local.sh        # no model (UI smoke)
#
# Backend on :8000, frontend on :4200 (proxying /api → :8000). Ctrl-C
# stops both. First run: pip install -r requirements-dqc.txt uvicorn
# and (cd DQC/app && npm ci).
set -euo pipefail

export REGLLM_LLM="${REGLLM_LLM:-bedrock}"
export BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-eu.amazon.nova-micro-v1:0}"
export BEDROCK_REGION="${BEDROCK_REGION:-eu-west-1}"
export REGLLM_ROUTERS="${REGLLM_ROUTERS:-dqc}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$REGLLM_LLM" == "bedrock" ]]; then
    if command -v aws >/dev/null 2>&1 && aws sts get-caller-identity >/dev/null 2>&1; then
        echo "• AWS credentials OK — Bedrock model: $BEDROCK_MODEL_ID ($BEDROCK_REGION)"
        echo "  (make sure the model is enabled: console → Bedrock → Model access)"
    else
        echo "⚠ AWS CLI not authenticated — Bedrock calls will fail. Run 'aws configure'"
        echo "  or use REGLLM_LLM=ollama / REGLLM_LLM=stub."
    fi
fi

echo "• backend  → http://localhost:8000  (REGLLM_LLM=$REGLLM_LLM)"
(cd "$ROOT" && python -m uvicorn api.main:app --port 8000 --reload) &
BACK_PID=$!
trap 'kill "$BACK_PID" 2>/dev/null || true' EXIT INT TERM

echo "• frontend → http://localhost:4200"
cd "$ROOT/DQC/app" && npx ng serve --proxy-config proxy.conf.json --port 4200
