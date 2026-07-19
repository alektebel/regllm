#!/usr/bin/env bash
# Deploy the RegLLM DQC stack to AWS from a clean account.
#
# This script is self-contained: it resolves the account's default VPC
# (creating one if absent), bootstraps CDK, deploys the stack (ECR repos,
# ALB, ECS cluster + service inside that VPC), builds and pushes the API
# and frontend Docker images (the frontend bundles the demo dictionary and
# cases Excels under /assets/demo, auto-loaded by the UI on startup), and
# finally triggers an ECS redeploy so the tasks pull the fresh images.
#
# Prerequisites:
#   - aws CLI configured (credentials + region) for the target account
#   - docker
#   - node + npm (to install the CDK CLI globally)
#
# Override defaults via environment variables:
#   AWS_REGION=eu-west-1       AWS region
#   PROJECT=regllm-dqc         resource name prefix
#   VPC_ID=vpc-...             deploy into this VPC (default: default VPC)
#   SUBNET_IDS=subnet-a,...    subnets for ALB+tasks (default: all in VPC)
#   GEMINI_API_KEY=...         Google Gemini key (switches backend to gemini)
#   GEMINI_MODEL=gemini-2.5-pro
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-west-1}"
PROJECT="${PROJECT:-regllm-dqc}"
STACK_NAME="RegllmDqcStack"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Optional: load GEMINI_API_KEY from the repo .env if not already in the env
if [[ -z "${GEMINI_API_KEY:-}" && -f "$REPO_ROOT/.env" ]]; then
    # shellcheck disable=SC1090
    set -a; . "$REPO_ROOT/.env"; set +a
fi

# ── Prerequisites ─────────────────────────────────────────────────────────
command -v aws >/dev/null 2>&1 || { echo "✗ aws CLI required (configure credentials first)"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "✗ docker required"; exit 1; }
if ! command -v cdk >/dev/null 2>&1; then
    echo "• installing CDK CLI…"
    npm install -g aws-cdk
fi

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "$ACCOUNT" ]] || { echo "✗ AWS authentication failed"; exit 1; }
echo "• account: $ACCOUNT  region: $AWS_REGION"

# ── Resolve VPC + subnets (the stack looks up an existing VPC) ─────────────
# Override with VPC_ID / SUBNET_IDS env vars; defaults to the default VPC.
if [[ -z "${VPC_ID:-}" ]]; then
    VPC_ID="$(aws ec2 describe-vpcs --region "$AWS_REGION" \
        --filters Name=isDefault,Values=true \
        --query 'Vpcs[0].VpcId' --output text)"
    if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
        echo "• no default VPC found — creating one…"
        aws ec2 create-default-vpc --region "$AWS_REGION" >/dev/null
        VPC_ID="$(aws ec2 describe-vpcs --region "$AWS_REGION" \
            --filters Name=isDefault,Values=true \
            --query 'Vpcs[0].VpcId' --output text)"
    fi
fi
if [[ -z "${SUBNET_IDS:-}" ]]; then
    SUBNET_IDS="$(aws ec2 describe-subnets --region "$AWS_REGION" \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[].SubnetId' --output text | tr '[:space:]' ',' | sed 's/,\+$//')"
fi
[[ -n "$VPC_ID" && "$VPC_ID" != "None" && -n "$SUBNET_IDS" ]] \
    || { echo "✗ could not resolve VPC/subnets (set VPC_ID and SUBNET_IDS)"; exit 1; }
echo "• vpc: $VPC_ID  subnets: $SUBNET_IDS"

cd "$REPO_ROOT/DQC/cdk"
pip install -q -r requirements.txt

# ── 1. Bootstrap + deploy infra (VPC, ECR, ALB, ECS) ──────────────────────
echo "• bootstrapping CDK (idempotent)…"
cdk bootstrap "aws://${ACCOUNT}/${AWS_REGION}" || true

GEMINI_ARGS=()
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    GEMINI_ARGS+=(-c "gemini_api_key=${GEMINI_API_KEY}" -c "gemini_model=${GEMINI_MODEL:-gemini-2.5-pro}")
    echo "• Gemini backend will be configured"
else
    echo "• no GEMINI_API_KEY set — defaulting to Bedrock backend"
fi

echo "• deploying CDK stack…"
cdk deploy --app "python3 app.py" \
    -c "aws_region=${AWS_REGION}" \
    -c "aws_account=${ACCOUNT}" \
    -c "project=${PROJECT}" \
    -c "vpc_id=${VPC_ID}" \
    -c "subnet_ids=${SUBNET_IDS}" \
    "${GEMINI_ARGS[@]}" \
    --require-approval never

# ── 2. Read ECR repo URIs + ALB DNS from stack outputs ────────────────────
stack_output() {
    aws cloudformation describe-stacks \
        --region "$AWS_REGION" --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
ECR_API="$(stack_output EcrApiUrl)"
ECR_DQC="$(stack_output EcrDqcUrl)"
ALB_DNS="$(stack_output AlbDns)"
[[ -n "$ECR_API" && -n "$ECR_DQC" ]] || { echo "✗ could not read ECR outputs"; exit 1; }

# ── 3. Build + push images to the CDK-created ECR repos ──────────────────
REGISTRY_DOMAIN="$(echo "$ECR_API" | cut -d/ -f1)"
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY_DOMAIN"

echo "• building + pushing API image…"
docker build -t "$ECR_API:latest" "$REPO_ROOT"
docker push "$ECR_API:latest"

echo "• building + pushing DQC frontend image…"
docker build -t "$ECR_DQC:latest" "$REPO_ROOT/DQC/app"
docker push "$ECR_DQC:latest"

# ── 4. Force the ECS service to pull the new images ──────────────────────
echo "• triggering ECS redeploy…"
aws ecs update-service --region "$AWS_REGION" \
    --cluster "$PROJECT" --service "$PROJECT" \
    --force-new-deployment >/dev/null

echo ""
echo "✓ Deployed."
echo "  App URL:  http://${ALB_DNS}"
echo "  Health:   http://${ALB_DNS}/api/health"
echo "  Stack:    $STACK_NAME  (region $AWS_REGION)"
