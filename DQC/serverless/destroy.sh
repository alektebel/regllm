#!/usr/bin/env bash
# Tear down everything ./deploy.sh created (Lambda, API Gateway, S3 site,
# CloudFront distribution + function + OAC, IAM role). Nothing here touches
# a VPC. CloudFront disable→delete is slow (~15 min); the script disables the
# distribution, waits for it to deploy, then deletes it.
#
#   AWS_REGION=eu-west-1 PROJECT=regllm-dqc ./DQC/serverless/destroy.sh
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-west-1}"
PROJECT="${PROJECT:-regllm-dqc}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"

ROLE="${PROJECT}-lambda-exec"
FUNC="${PROJECT}-api"
API_NAME="${PROJECT}-http"
SITE_BUCKET="${PROJECT}-site-${ACCOUNT}-${AWS_REGION}"
OAC_NAME="${PROJECT}-oac"
CF_FUNC_NAME="${PROJECT}-strip-api"
CF_COMMENT="${PROJECT} serverless demo"

echo "• tearing down '$PROJECT' in $AWS_REGION (account $ACCOUNT)"

# ── CloudFront distribution: disable, wait, delete ────────────────────────
DIST_ID="$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Comment=='$CF_COMMENT'].Id | [0]" --output text 2>/dev/null || true)"
if [[ -n "$DIST_ID" && "$DIST_ID" != "None" ]]; then
    ETAG="$(aws cloudfront get-distribution-config --id "$DIST_ID" --query 'ETag' --output text)"
    if [[ "$(aws cloudfront get-distribution --id "$DIST_ID" --query 'Distribution.DistributionConfig.Enabled' --output text)" == "True" ]]; then
        echo "• disabling CloudFront $DIST_ID…"
        aws cloudfront get-distribution-config --id "$DIST_ID" --query 'DistributionConfig' > /tmp/_cf.json
        python3 -c "import json;d=json.load(open('/tmp/_cf.json'));d['Enabled']=False;json.dump(d,open('/tmp/_cf.json','w'))"
        aws cloudfront update-distribution --id "$DIST_ID" --if-match "$ETAG" \
            --distribution-config file:///tmp/_cf.json >/dev/null
        echo "  ↳ waiting for the disable to deploy (~15 min)…"
        aws cloudfront wait distribution-deployed --id "$DIST_ID"
        ETAG="$(aws cloudfront get-distribution-config --id "$DIST_ID" --query 'ETag' --output text)"
    fi
    echo "• deleting CloudFront $DIST_ID…"
    aws cloudfront delete-distribution --id "$DIST_ID" --if-match "$ETAG"
fi

# ── CloudFront function ───────────────────────────────────────────────────
if aws cloudfront describe-function --name "$CF_FUNC_NAME" >/dev/null 2>&1; then
    ETAG="$(aws cloudfront describe-function --name "$CF_FUNC_NAME" --query 'ETag' --output text)"
    echo "• deleting CloudFront function $CF_FUNC_NAME…"
    aws cloudfront delete-function --name "$CF_FUNC_NAME" --if-match "$ETAG" || true
fi

# ── Origin Access Control ─────────────────────────────────────────────────
OAC_ID="$(aws cloudfront list-origin-access-controls \
    --query "OriginAccessControlList.Items[?Name=='$OAC_NAME'].Id | [0]" --output text 2>/dev/null || true)"
if [[ -n "$OAC_ID" && "$OAC_ID" != "None" ]]; then
    ETAG="$(aws cloudfront get-origin-access-control --id "$OAC_ID" --query 'ETag' --output text)"
    echo "• deleting OAC $OAC_ID…"
    aws cloudfront delete-origin-access-control --id "$OAC_ID" --if-match "$ETAG" || true
fi

# ── API Gateway ───────────────────────────────────────────────────────────
API_ID="$(aws apigatewayv2 get-apis --region "$AWS_REGION" \
    --query "Items[?Name=='$API_NAME'].ApiId | [0]" --output text 2>/dev/null || true)"
if [[ -n "$API_ID" && "$API_ID" != "None" ]]; then
    echo "• deleting API Gateway $API_ID…"
    aws apigatewayv2 delete-api --region "$AWS_REGION" --api-id "$API_ID" || true
fi

# ── Lambda ────────────────────────────────────────────────────────────────
if aws lambda get-function --region "$AWS_REGION" --function-name "$FUNC" >/dev/null 2>&1; then
    echo "• deleting Lambda $FUNC…"
    aws lambda delete-function --region "$AWS_REGION" --function-name "$FUNC" || true
fi

# ── S3 site bucket ────────────────────────────────────────────────────────
if aws s3api head-bucket --bucket "$SITE_BUCKET" >/dev/null 2>&1; then
    echo "• emptying + deleting S3 bucket $SITE_BUCKET…"
    aws s3 rm "s3://$SITE_BUCKET" --recursive --only-show-errors || true
    aws s3api delete-bucket --bucket "$SITE_BUCKET" --region "$AWS_REGION" || true
fi

# ── IAM role ──────────────────────────────────────────────────────────────
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    echo "• deleting IAM role $ROLE…"
    aws iam delete-role-policy --role-name "$ROLE" --policy-name bedrock-invoke 2>/dev/null || true
    aws iam detach-role-policy --role-name "$ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
    aws iam delete-role --role-name "$ROLE" || true
fi

echo "✓ Teardown complete."
