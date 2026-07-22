#!/usr/bin/env bash
# Deploy the RegLLM DQC stack to AWS from a clean account.
#
# This script is self-contained: it resolves a VPC to deploy into (explicit
# VPC_ID, else the account default, else the region's only VPC; it will try
# to create a default VPC but tolerates accounts that can't), bootstraps CDK,
# deploys the stack (ECR repos,
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

# ── Resolve VPC + subnets (the stack deploys into an EXISTING VPC) ─────────
# Resolution order: explicit VPC_ID env → account default VPC → the sole VPC
# in the region. Creating a VPC needs ec2:CreateDefaultVpc / ec2:CreateVpc,
# which locked-down accounts (e.g. read-only or SCP-restricted) do not grant,
# so a create attempt is best-effort: on failure we surface the real reason
# and fall back to an existing VPC instead of dying on a raw AWS error.
if [[ -z "${VPC_ID:-}" ]]; then
    VPC_ID="$(aws ec2 describe-vpcs --region "$AWS_REGION" \
        --filters Name=isDefault,Values=true \
        --query 'Vpcs[0].VpcId' --output text 2>/dev/null || true)"

    if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
        echo "• no default VPC — attempting to create one…"
        vpc_err="$(mktemp)"
        if aws ec2 create-default-vpc --region "$AWS_REGION" >/dev/null 2>"$vpc_err"; then
            VPC_ID="$(aws ec2 describe-vpcs --region "$AWS_REGION" \
                --filters Name=isDefault,Values=true \
                --query 'Vpcs[0].VpcId' --output text)"
        else
            echo "  ↳ cannot create a default VPC (this account lacks the permission):"
            sed 's/^/    /' "$vpc_err"
            echo "• falling back to an existing VPC in $AWS_REGION…"
            # Collect existing VPCs (space/newline separated) into an array.
            read -r -a _vpcs <<<"$(aws ec2 describe-vpcs --region "$AWS_REGION" \
                --query 'Vpcs[].VpcId' --output text 2>/dev/null | tr '\t' ' ')"
            if [[ "${#_vpcs[@]}" -eq 1 ]]; then
                VPC_ID="${_vpcs[0]}"
                echo "  ↳ using the only VPC found: $VPC_ID"
            elif [[ "${#_vpcs[@]}" -gt 1 ]]; then
                echo "✗ several VPCs exist and none is default — pick one explicitly:"
                printf '    VPC_ID=%s ./DQC/cdk/deploy.sh\n' "${_vpcs[@]}"
                exit 1
            else
                echo "✗ no VPC is available and none can be created."
                echo "  Ask your AWS admin for a VPC + subnet IDs, then re-run with:"
                echo "    VPC_ID=vpc-xxxx SUBNET_IDS=subnet-a,subnet-b ./DQC/cdk/deploy.sh"
                exit 1
            fi
        fi
        rm -f "$vpc_err"
    fi
fi
if [[ -z "${SUBNET_IDS:-}" ]]; then
    SUBNET_IDS="$(aws ec2 describe-subnets --region "$AWS_REGION" \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[].SubnetId' --output text 2>/dev/null | tr '[:space:]' ',' | sed 's/,\+$//')"
fi
if [[ -z "$VPC_ID" || "$VPC_ID" == "None" || -z "$SUBNET_IDS" ]]; then
    echo "✗ could not resolve a VPC and its subnets."
    echo "  Set them explicitly (get the IDs from your admin or the AWS console):"
    echo "    VPC_ID=vpc-xxxx SUBNET_IDS=subnet-a,subnet-b ./DQC/cdk/deploy.sh"
    exit 1
fi
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
