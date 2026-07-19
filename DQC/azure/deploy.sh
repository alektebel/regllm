#!/usr/bin/env bash
# Deploy the RegLLM DQC stack to Azure from a clean subscription.
#
# Azure mirror of DQC/cdk/deploy.sh (AWS): one Azure Container App running
# the same two containers that share localhost on Fargate —
#   api  (FastAPI, port 8000, LLM backend Azure OpenAI)
#   dqc  (Angular + nginx, port 80, proxies /api → localhost:8000; bundles
#         the demo dictionary/cases Excels auto-loaded by the UI)
# behind the Container App's managed HTTPS ingress. Images are built
# server-side with `az acr build` (no local Docker needed).
#
# Prerequisites:
#   - az CLI logged in (`az login`) with a subscription selected
#   - LLM: an Azure OpenAI resource + chat deployment. Pass it via:
#       AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com
#       AZURE_OPENAI_API_KEY=...
#       AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini        (default)
#     Without these the app deploys with the stub backend (UI works,
#     generation degrades) — you can update env vars later with
#     `az containerapp update`.
#
# Overridable env vars:
#   LOCATION=westeurope        Azure region
#   PROJECT=regllm-dqc         resource name prefix
#   RG=$PROJECT-rg             resource group
set -euo pipefail

LOCATION="${LOCATION:-westeurope}"
PROJECT="${PROJECT:-regllm-dqc}"
RG="${RG:-${PROJECT}-rg}"
ENV_NAME="${PROJECT}-env"
APP_NAME="${PROJECT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

command -v az >/dev/null 2>&1 || { echo "✗ az CLI required (az login first)"; exit 1; }
SUB_ID="$(az account show --query id -o tsv)"
[[ -n "$SUB_ID" ]] || { echo "✗ Azure authentication failed — run az login"; exit 1; }
# ACR names: 5-50 alphanumerics, globally unique — prefix + subscription hash
ACR_NAME="$(echo "${PROJECT}${SUB_ID}" | tr -cd 'a-z0-9' | cut -c1-24)"
echo "• subscription: $SUB_ID  region: $LOCATION  rg: $RG  acr: $ACR_NAME"

# ── 1. Resource group + registry ──────────────────────────────────────────
az group create --name "$RG" --location "$LOCATION" --output none
az acr create --resource-group "$RG" --name "$ACR_NAME" \
    --sku Basic --admin-enabled true --output none
ACR_SERVER="$(az acr show -g "$RG" -n "$ACR_NAME" --query loginServer -o tsv)"

# ── 2. Build both images in the cloud (no local Docker) ──────────────────
echo "• building API image in ACR…"
az acr build --registry "$ACR_NAME" --image "${PROJECT}-api:latest" \
    --file "$REPO_ROOT/Dockerfile" "$REPO_ROOT"
echo "• building DQC frontend image in ACR…"
az acr build --registry "$ACR_NAME" --image "${PROJECT}-dqc:latest" \
    "$REPO_ROOT/DQC/app"

# ── 3. Container Apps environment ─────────────────────────────────────────
az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az containerapp env create --name "$ENV_NAME" --resource-group "$RG" \
    --location "$LOCATION" --output none

ACR_USER="$(az acr credential show -g "$RG" -n "$ACR_NAME" --query username -o tsv)"
ACR_PWD="$(az acr credential show -g "$RG" -n "$ACR_NAME" --query 'passwords[0].value' -o tsv)"
ENV_ID="$(az containerapp env show -g "$RG" -n "$ENV_NAME" --query id -o tsv)"

# ── 4. App spec: two containers sharing localhost, HTTPS ingress on 80 ───
if [[ -n "${AZURE_OPENAI_ENDPOINT:-}" && -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
    LLM_BACKEND="azure"
    echo "• Azure OpenAI backend configured (deployment: ${AZURE_OPENAI_DEPLOYMENT:-gpt-4o-mini})"
else
    LLM_BACKEND="stub"
    echo "⚠ no AZURE_OPENAI_ENDPOINT/API_KEY — deploying with stub LLM backend"
fi

APP_YAML="$(mktemp --suffix=.yaml)"
trap 'rm -f "$APP_YAML"' EXIT
cat > "$APP_YAML" <<YAML
properties:
  environmentId: ${ENV_ID}
  configuration:
    ingress:
      external: true
      targetPort: 80
      transport: auto
    registries:
      - server: ${ACR_SERVER}
        username: ${ACR_USER}
        passwordSecretRef: acr-pwd
    secrets:
      - name: acr-pwd
        value: ${ACR_PWD}
      - name: azure-openai-key
        value: ${AZURE_OPENAI_API_KEY:-unused}
  template:
    containers:
      - name: api
        image: ${ACR_SERVER}/${PROJECT}-api:latest
        env:
          - name: REGLLM_LLM
            value: "${LLM_BACKEND}"
          - name: REGLLM_ROUTERS
            value: "dqc"
          - name: CORS_ORIGINS
            value: "*"
          - name: AZURE_OPENAI_ENDPOINT
            value: "${AZURE_OPENAI_ENDPOINT:-}"
          - name: AZURE_OPENAI_DEPLOYMENT
            value: "${AZURE_OPENAI_DEPLOYMENT:-gpt-4o-mini}"
          - name: AZURE_OPENAI_API_VERSION
            value: "${AZURE_OPENAI_API_VERSION:-2024-06-01}"
          - name: AZURE_OPENAI_API_KEY
            secretRef: azure-openai-key
        resources:
          cpu: 1.0
          memory: 2Gi
      - name: dqc
        image: ${ACR_SERVER}/${PROJECT}-dqc:latest
        env:
          - name: API_UPSTREAM
            value: "localhost:8000"
        resources:
          cpu: 0.5
          memory: 1Gi
    scale:
      minReplicas: 1
      maxReplicas: 1
YAML

echo "• creating/updating container app…"
if az containerapp show -g "$RG" -n "$APP_NAME" --output none 2>/dev/null; then
    az containerapp update -g "$RG" -n "$APP_NAME" --yaml "$APP_YAML" --output none
else
    az containerapp create -g "$RG" -n "$APP_NAME" --yaml "$APP_YAML" --output none
fi

FQDN="$(az containerapp show -g "$RG" -n "$APP_NAME" \
    --query properties.configuration.ingress.fqdn -o tsv)"
echo ""
echo "✓ Deployed."
echo "  App URL:  https://${FQDN}"
echo "  Health:   https://${FQDN}/api/health"
echo "  Logs:     az containerapp logs show -g $RG -n $APP_NAME --follow"
echo "  Destroy:  az group delete --name $RG --yes --no-wait"
