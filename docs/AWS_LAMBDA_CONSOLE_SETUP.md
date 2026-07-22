# DQC backend on AWS Lambda, without AWS CLI

This setup keeps the Angular frontend on the developer machine and exposes the
DQC API through API Gateway and Lambda. No AWS CLI command is used. The local
Angular development server proxies `/api` to the HTTPS endpoint.

## Architecture

```text
Angular at http://localhost:4200 -> API Gateway HTTPS API -> Lambda (FastAPI via Mangum) -> Amazon Bedrock
```

Lambda is stateless. The DQC SQLite validation store must use an **Amazon EFS
access point** mounted at `/mnt/regllm-data` for durable shared review data.
Set `REGLLM_CHECKS_DB=/mnt/regllm-data/checks.db`; do not use `/tmp`.

## 1. Enable Bedrock

1. In the AWS Console, open **Amazon Bedrock** in the deployment region.
2. Open **Model access**, request access to the `BEDROCK_MODEL_ID` model.
3. Use a Playground prompt to confirm access.

## 2. Create EFS storage

1. In **Amazon EFS**, create a file system in the Lambda VPC.
2. Create an access point with POSIX user/group `1000` and root directory
   `/regllm-data`.
3. Allow inbound NFS/TCP `2049` to its security group only from Lambda.

## 3. Create the Lambda execution role

1. In **IAM**, create a role for **Lambda**.
2. Attach `AWSLambdaBasicExecutionRole`.
3. Add `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` for
   the approved model ARN.
4. Add `elasticfilesystem:ClientMount` and `elasticfilesystem:ClientWrite` for
   the EFS access point ARN.

## 4. Package and create Lambda

Package locally with Python, not the AWS CLI:

```bash
mkdir -p .lambda-package
python -m pip install -r requirements-dqc.txt -t .lambda-package
cp -R api src training data config.yaml .lambda-package/
cp DQC/lambda/handler.py .lambda-package/
```

ZIP the contents of `.lambda-package`, ensuring `handler.py` is at the ZIP
root. In the **Lambda** console create a Python 3.11 function, select the role,
upload the ZIP, and set handler `handler.handler`.

Set memory to at least 2048 MB and timeout to 29 seconds. Add environment
variables: `REGLLM_ROUTERS=dqc`, `REGLLM_LLM=bedrock`,
`BEDROCK_REGION=<region>`, `BEDROCK_MODEL_ID=<model>`,
`REGLLM_CHECKS_DB=/mnt/regllm-data/checks.db`, and
`CORS_ORIGINS=http://localhost:4200`. Under **Configuration > VPC**, select
private subnets, attach Lambda's security group, and add the EFS access point at
`/mnt/regllm-data`.

## 5. Create API Gateway

1. In **API Gateway**, create an **HTTP API** with the Lambda integration.
2. Add route `ANY /{proxy+}`.
3. Enable CORS for `http://localhost:4200`, methods `GET`, `POST`, `DELETE`,
   and headers `Content-Type`, `Authorization`.
4. Deploy the default stage and copy the Invoke URL.
5. Check `<invoke-url>/health` in a browser.

## 6. Run Angular locally

Replace `YOUR_API_GATEWAY_URL` in `DQC/app/proxy.aws.conf.json` with the Invoke
URL, then run:

```bash
cd DQC/app
npm install
npx ng serve --proxy-config proxy.aws.conf.json --port 4200
```

The frontend remains local and `/api/dqc/*` is sent to API Gateway.

## Constraints

- API Gateway has a 29-second integration limit and 10 MB request/response
  limit. Use presigned S3 uploads for larger workbooks.
- API Gateway buffers SSE. The DQC decision tree is still returned when the
  generation completes, but it is not live. Use ECS or Lambda response
  streaming when live SSE is required.
