#!/usr/bin/env bash
# Deploy the RegLLM DQC demo to AWS **without any VPC** and get one public
# HTTPS URL to share.
#
#   Backend : Lambda (FastAPI via Mangum) + API Gateway HTTP API
#             → ephemeral /tmp SQLite (no EFS ⇒ no VPC), LLM on Bedrock
#   Frontend: Angular built to static files on a PRIVATE S3 bucket, served
#             through CloudFront (Origin Access Control)
#   Glue    : a CloudFront Function strips the `/api` prefix at the edge and
#             routes `/api/*` to API Gateway — same origin, so no CORS and no
#             frontend code changes (identical to the local dev proxy).
#
# The one URL it prints (https://<id>.cloudfront.net) serves the demo UI and
# proxies the API. Send it to anyone.
#
# This creates NO VPC, subnets, NAT, ALB, or ECS — nothing that needs the
# ec2:Create* permissions a locked-down account tends to lack.
#
# Prerequisites on the machine running this (all present in AWS CloudShell):
#   - aws CLI configured (credentials + region)
#   - python3 + pip, node + npm, zip
#
# Required IAM permissions (NOT ec2/vpc): lambda:*, apigateway (apigatewayv2),
#   iam:CreateRole/AttachRolePolicy/PutRolePolicy/PassRole, s3:*,
#   cloudfront:* (incl. functions + origin-access-control), and Bedrock model
#   access enabled in the region.
#
# Override defaults via environment variables:
#   AWS_REGION=eu-west-1              AWS region
#   PROJECT=regllm-dqc               resource name prefix
#   BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0
#   INSPECT_BEDROCK_MODEL_ID=...     cheap per-upload inspect model (default: =model)
#   GEMINI_API_KEY=...               use Gemini instead of Bedrock
#   GEMINI_MODEL=gemini-2.5-pro
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-west-1}"
PROJECT="${PROJECT:-regllm-dqc}"
BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-eu.amazon.nova-micro-v1:0}"
INSPECT_BEDROCK_MODEL_ID="${INSPECT_BEDROCK_MODEL_ID:-$BEDROCK_MODEL_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Optional: load GEMINI_API_KEY from the repo .env if not already set
if [[ -z "${GEMINI_API_KEY:-}" && -f "$REPO_ROOT/.env" ]]; then
    # shellcheck disable=SC1090
    set -a; . "$REPO_ROOT/.env"; set +a
fi

# ── Prerequisites ─────────────────────────────────────────────────────────
for bin in aws python3 npm zip; do
    command -v "$bin" >/dev/null 2>&1 || { echo "✗ '$bin' is required"; exit 1; }
done
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "$ACCOUNT" && "$ACCOUNT" != "None" ]] || { echo "✗ AWS authentication failed"; exit 1; }
echo "• account: $ACCOUNT  region: $AWS_REGION  (no VPC will be created)"

# Resource names
ROLE="${PROJECT}-lambda-exec"
FUNC="${PROJECT}-api"
API_NAME="${PROJECT}-http"
SITE_BUCKET="${PROJECT}-site-${ACCOUNT}-${AWS_REGION}"
OAC_NAME="${PROJECT}-oac"
CF_FUNC_NAME="${PROJECT}-strip-api"
CF_COMMENT="${PROJECT} serverless demo"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# LLM backend env for the Lambda
declare -a LLM_ENV
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    echo "• LLM backend: Gemini (${GEMINI_MODEL:-gemini-2.5-pro})"
    LLM_ENV=(REGLLM_LLM=gemini "GEMINI_API_KEY=${GEMINI_API_KEY}" "GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-pro}")
else
    echo "• LLM backend: Bedrock ($BEDROCK_MODEL_ID)"
    LLM_ENV=(REGLLM_LLM=bedrock "BEDROCK_REGION=${AWS_REGION}" "BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID}" "INSPECT_BEDROCK_MODEL_ID=${INSPECT_BEDROCK_MODEL_ID}")
fi

# ── 1. Build the Angular frontend ─────────────────────────────────────────
echo "• building Angular frontend…"
( cd "$REPO_ROOT/DQC/app" && { [[ -d node_modules ]] || npm ci; } && npx ng build --configuration production )
DIST="$REPO_ROOT/DQC/app/dist/dqc-app/browser"
[[ -f "$DIST/index.html" ]] || { echo "✗ frontend build missing $DIST/index.html"; exit 1; }

# ── 2. Package the Lambda (Amazon Linux x86_64 wheels) ────────────────────
echo "• packaging Lambda deployment zip…"
PKG="$WORK/pkg"
mkdir -p "$PKG"
# Cross-platform-safe install: fetch manylinux/x86_64 wheels for py3.11 so a
# build from macOS/ARM still produces Linux-compatible binaries (pydantic-core).
pip install -q \
    --platform manylinux2014_x86_64 --implementation cp --python-version 3.11 \
    --only-binary=:all: --upgrade \
    -r "$REPO_ROOT/requirements-dqc.txt" -t "$PKG"
cp "$REPO_ROOT/DQC/lambda/handler.py" "$PKG/"
cp "$REPO_ROOT/config.yaml" "$PKG/"
for d in api src training data; do
    cp -R "$REPO_ROOT/$d" "$PKG/$d"
done
( cd "$PKG" && zip -qr "$WORK/api.zip" . )
echo "  ↳ zip: $(du -h "$WORK/api.zip" | cut -f1)"

# ── 3. S3 bucket (private) for the static site + lambda artifact ──────────
echo "• ensuring S3 bucket $SITE_BUCKET…"
if ! aws s3api head-bucket --bucket "$SITE_BUCKET" >/dev/null 2>&1; then
    if [[ "$AWS_REGION" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "$SITE_BUCKET" --region "$AWS_REGION" >/dev/null
    else
        aws s3api create-bucket --bucket "$SITE_BUCKET" --region "$AWS_REGION" \
            --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
    fi
fi
# Stage the lambda zip in S3 (larger-than-direct-upload safe)
aws s3 cp "$WORK/api.zip" "s3://$SITE_BUCKET/_deploy/api.zip" --only-show-errors

# ── 4. IAM execution role (Lambda + Bedrock) ──────────────────────────────
echo "• ensuring IAM role $ROLE…"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    cat >"$WORK/trust.json" <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
    aws iam create-role --role-name "$ROLE" \
        --assume-role-policy-document "file://$WORK/trust.json" >/dev/null
    aws iam attach-role-policy --role-name "$ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null
    echo "  ↳ waiting for role to propagate…"
    sleep 12
fi
cat >"$WORK/bedrock.json" <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],
  "Resource":["arn:aws:bedrock:${AWS_REGION}:${ACCOUNT}:inference-profile/${BEDROCK_MODEL_ID}",
              "arn:aws:bedrock:*::foundation-model/*"]}]}
JSON
aws iam put-role-policy --role-name "$ROLE" --policy-name bedrock-invoke \
    --policy-document "file://$WORK/bedrock.json" >/dev/null
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query 'Role.Arn' --output text)"

# ── 5. Lambda function ────────────────────────────────────────────────────
# Assemble env vars into the Variables={..} form the CLI expects
ENV_KV="REGLLM_ROUTERS=dqc,REGLLM_CHECKS_DB=/tmp/checks.db,CORS_ORIGINS=*"
for kv in "${LLM_ENV[@]}"; do ENV_KV+=",$kv"; done

if aws lambda get-function --function-name "$FUNC" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "• updating Lambda $FUNC…"
    aws lambda update-function-code --region "$AWS_REGION" --function-name "$FUNC" \
        --s3-bucket "$SITE_BUCKET" --s3-key _deploy/api.zip >/dev/null
    aws lambda wait function-updated --region "$AWS_REGION" --function-name "$FUNC"
    aws lambda update-function-configuration --region "$AWS_REGION" --function-name "$FUNC" \
        --handler handler.handler --runtime python3.11 --timeout 29 --memory-size 2048 \
        --environment "Variables={${ENV_KV}}" >/dev/null
else
    echo "• creating Lambda $FUNC…"
    aws lambda create-function --region "$AWS_REGION" --function-name "$FUNC" \
        --runtime python3.11 --role "$ROLE_ARN" --handler handler.handler \
        --code "S3Bucket=$SITE_BUCKET,S3Key=_deploy/api.zip" \
        --timeout 29 --memory-size 2048 \
        --environment "Variables={${ENV_KV}}" >/dev/null
fi
aws lambda wait function-updated --region "$AWS_REGION" --function-name "$FUNC"
FUNC_ARN="$(aws lambda get-function --region "$AWS_REGION" --function-name "$FUNC" \
    --query 'Configuration.FunctionArn' --output text)"

# ── 6. API Gateway HTTP API (quick-create: integration + $default route) ──
API_ID="$(aws apigatewayv2 get-apis --region "$AWS_REGION" \
    --query "Items[?Name=='$API_NAME'].ApiId | [0]" --output text)"
if [[ -z "$API_ID" || "$API_ID" == "None" ]]; then
    echo "• creating API Gateway HTTP API…"
    API_ID="$(aws apigatewayv2 create-api --region "$AWS_REGION" \
        --name "$API_NAME" --protocol-type HTTP --target "$FUNC_ARN" \
        --query 'ApiId' --output text)"
fi
# Allow API Gateway to invoke the Lambda (idempotent — ignore if it exists)
aws lambda add-permission --region "$AWS_REGION" --function-name "$FUNC" \
    --statement-id apigw-invoke --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${AWS_REGION}:${ACCOUNT}:${API_ID}/*/*" \
    >/dev/null 2>&1 || true
API_DOMAIN="${API_ID}.execute-api.${AWS_REGION}.amazonaws.com"
echo "  ↳ api: https://$API_DOMAIN"

# ── 7. CloudFront: Origin Access Control + edge function ──────────────────
OAC_ID="$(aws cloudfront list-origin-access-controls \
    --query "OriginAccessControlList.Items[?Name=='$OAC_NAME'].Id | [0]" --output text 2>/dev/null || true)"
if [[ -z "$OAC_ID" || "$OAC_ID" == "None" ]]; then
    echo "• creating CloudFront Origin Access Control…"
    OAC_ID="$(aws cloudfront create-origin-access-control --origin-access-control-config \
        "Name=$OAC_NAME,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
        --query 'OriginAccessControl.Id' --output text)"
fi

CF_FUNC_ARN="$(aws cloudfront list-functions \
    --query "FunctionList.Items[?Name=='$CF_FUNC_NAME'].FunctionMetadata.FunctionARN | [0]" \
    --output text 2>/dev/null || true)"
if [[ -z "$CF_FUNC_ARN" || "$CF_FUNC_ARN" == "None" ]]; then
    echo "• creating CloudFront Function (strip /api)…"
    CF_FUNC_ARN="$(aws cloudfront create-function --name "$CF_FUNC_NAME" \
        --function-config "Comment=strip /api prefix,Runtime=cloudfront-js-2.0" \
        --function-code "fileb://$SCRIPT_DIR/cf-strip-api.js" \
        --query 'FunctionSummary.FunctionMetadata.FunctionARN' --output text)"
    ETAG="$(aws cloudfront describe-function --name "$CF_FUNC_NAME" --query 'ETag' --output text)"
    aws cloudfront publish-function --name "$CF_FUNC_NAME" --if-match "$ETAG" >/dev/null
fi

# ── 8. CloudFront distribution (create once; reused on re-runs) ────────────
DIST_ID="$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Comment=='$CF_COMMENT'].Id | [0]" --output text 2>/dev/null || true)"
if [[ -z "$DIST_ID" || "$DIST_ID" == "None" ]]; then
    echo "• creating CloudFront distribution (first run: ~10-15 min to deploy)…"
    S3_DOMAIN="${SITE_BUCKET}.s3.${AWS_REGION}.amazonaws.com"
    CALLER_REF="${PROJECT}-$(date +%s)"
    # Managed policy IDs: CachingOptimized / CachingDisabled / AllViewerExceptHostHeader
    python3 - "$WORK/dist.json" \
        "$S3_DOMAIN" "$OAC_ID" "$API_DOMAIN" "$CF_FUNC_ARN" "$CF_COMMENT" "$CALLER_REF" <<'PY'
import json, sys
out, s3, oac, api, fn, comment, ref = sys.argv[1:8]
cfg = {
  "CallerReference": ref,
  "Comment": comment,
  "Enabled": True,
  "DefaultRootObject": "index.html",
  "Origins": {"Quantity": 2, "Items": [
    {"Id": "s3-site", "DomainName": s3, "OriginAccessControlId": oac,
     "S3OriginConfig": {"OriginAccessIdentity": ""}},
    {"Id": "api-gw", "DomainName": api,
     "CustomOriginConfig": {"HTTPPort": 80, "HTTPSPort": 443,
        "OriginProtocolPolicy": "https-only",
        "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]}}},
  ]},
  "DefaultCacheBehavior": {
    "TargetOriginId": "s3-site", "ViewerProtocolPolicy": "redirect-to-https",
    "Compress": True,
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
    "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"],
        "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
  },
  "CacheBehaviors": {"Quantity": 1, "Items": [
    {"PathPattern": "/api/*", "TargetOriginId": "api-gw",
     "ViewerProtocolPolicy": "redirect-to-https", "Compress": True,
     "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
     "OriginRequestPolicyId": "b689b0a8-53d0-40ab-baf2-68738e2966ac",
     "AllowedMethods": {"Quantity": 7,
        "Items": ["GET","HEAD","OPTIONS","PUT","POST","PATCH","DELETE"],
        "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
     "FunctionAssociations": {"Quantity": 1, "Items": [
        {"EventType": "viewer-request", "FunctionARN": fn}]}},
  ]},
  "CustomErrorResponses": {"Quantity": 2, "Items": [
    {"ErrorCode": 403, "ResponseCode": "200", "ResponsePagePath": "/index.html",
     "ErrorCachingMinTTL": 10},
    {"ErrorCode": 404, "ResponseCode": "200", "ResponsePagePath": "/index.html",
     "ErrorCachingMinTTL": 10}]},
}
json.dump({"DistributionConfig": cfg}, open(out, "w"))
PY
    DIST_ID="$(aws cloudfront create-distribution \
        --distribution-config "file://$WORK/dist.json" \
        --query 'Distribution.Id' --output text)"
else
    echo "• reusing CloudFront distribution $DIST_ID"
fi
DIST_DOMAIN="$(aws cloudfront get-distribution --id "$DIST_ID" \
    --query 'Distribution.DomainName' --output text)"

# ── 9. Bucket policy: allow this distribution (OAC) to read the site ──────
cat >"$WORK/bucket.json" <<JSON
{"Version":"2012-10-17","Statement":[{"Sid":"AllowCloudFrontOAC",
  "Effect":"Allow","Principal":{"Service":"cloudfront.amazonaws.com"},
  "Action":"s3:GetObject","Resource":"arn:aws:s3:::${SITE_BUCKET}/*",
  "Condition":{"StringEquals":{"AWS:SourceArn":"arn:aws:cloudfront::${ACCOUNT}:distribution/${DIST_ID}"}}}]}
JSON
aws s3api put-bucket-policy --bucket "$SITE_BUCKET" --policy "file://$WORK/bucket.json"

# ── 10. Upload the site + invalidate cache ────────────────────────────────
echo "• uploading frontend to S3…"
aws s3 sync "$DIST" "s3://$SITE_BUCKET/" --delete --only-show-errors
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null

echo ""
echo "✓ Deployed — no VPC involved."
echo "  Share this URL:  https://${DIST_DOMAIN}"
echo "  API (direct):    https://${API_DOMAIN}/health"
echo ""
echo "  First deploy? CloudFront takes ~10-15 min to finish rolling out —"
echo "  the URL 404s or shows a stale page until then. Re-runs are fast."
echo "  Tear down:  ./DQC/serverless/destroy.sh"
