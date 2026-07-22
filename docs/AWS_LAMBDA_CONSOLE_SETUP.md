# DQC backend on AWS Lambda, without AWS CLI

This setup keeps the Angular frontend on the developer machine and exposes the
DQC API as an HTTPS API Gateway endpoint backed by a Lambda function. No AWS
CLI command is used. The local Angular development server proxies `/api` to
that endpoint, so browser requests remain same-origin during development.

## Architecture

```text
Angular at http://localhost:4200
        |
        | /api/dqc/* via Angular dev proxy
        v
API Gateway HTTPS API
        |
        v
Lambda (FastAPI via Mangum) -> Amazon Bedrock
```

Lambda is stateless. The current DQC validation store is SQLite and therefore
must use an **Amazon EFS access point** mounted at `/mnt/regllm-data` before it
is used for shared review data. Set `REGLLM_CHECKS_DB=/mnt/regllm-data/checks.db`.
Do not rely on
Lambda's `/tmp` directory for DQC review state: it is not shared or durable.

## 1. Enable the Bedrock model

1. In the AWS Console, open **Amazon Bedrock** in the deployment region.
2. Open **Model access**, request access to the model selected by
   `BEDROCK_MODEL_ID`, and wait until it is available.
3. In **Playgrounds**, send a small test prompt to confirm account access.

## 2. Create durable storage

1. In **Amazon EFS**, create a file system in the VPC that Lambda will use.
2. Create an access point with POSIX user/group `1000` and root directory
   `/regllm-data`, granting owner read/write/execute permissions.
3. Create a security group for EFS and allow inbound NFS/TCP `2049` only from
   the Lambda security group created in step 4.

## 3. Create the Lambda execution role

1. Open **IAM**, create role, select **Lambda** as trusted entity.
2. Attach AWS managed policy `AWSLambdaBasicExecutionRole`.
3. Add an inline policy permitting `bedrock:InvokeModel` and
   `bedrock:InvokeModelWithResponseStream` for the selected model ARN.
4. Add an inline policy permitting EFS client mount/write for the EFS access
   point ARN: `elasticfilesystem:ClientMount` and
   `elasticfilesystem:ClientWrite`.

## 4. Package and create Lambda

Package dependencies locally with Python, not AWS CLI:

```bash
mkdir -p .lambda-package
python -m pip install -r requirements-dqc.txt -t .lambda-package
cp -R api src training data config.yaml DQC/lambda/handler.py .lambda-package/
```

Create a ZIP from the contents of `.lambda-package` using the operating system
file manager or any ZIP utility. Ensure `handler.py` is at the ZIP root.

1. In **Lambda**, choose **Create function**, then **Author from scratch**.
2. Use Python 3.11, choose the role from step 3, and set handler to
   `handler.handler`.
3. Upload the ZIP on the **Code** tab.
4. Set memory to at least 2048 MB and timeout to 29 seconds. API Gateway has a
   29-second integration limit.
5. Under **Configuration > Environment variables**, set:
   `REGLLM_ROUTERS=dqc`, `REGLLM_LLM=bedrock`, `BEDROCK_REGION=<region>`,
   `BEDROCK_MODEL_ID=<approved-model-id>`,
   `REGLLM_CHECKS_DB=/mnt/regllm-data/checks.db`, and
   `CORS_ORIGINS=http://localhost:4200`.
6. Under **Configuration > VPC**, select the EFS VPC and private subnets,
   attach the Lambda security group, then add the EFS access point under
   **File systems** with local mount path `/mnt/regllm-data`.

## 5. Create API Gateway

1. In **API Gateway**, create an **HTTP API**.
2. Add a Lambda integration and select the function above.
3. Add route `ANY /{proxy+}` and attach the integration.
4. Enable CORS for origin `http://localhost:4200`, methods `GET`, `POST`,
   `DELETE`, and headers `Content-Type`, `Authorization`.
5. Deploy the default stage and copy the Invoke URL, for example
   `https://abc123.execute-api.eu-west-1.amazonaws.com`.
6. Test `<invoke-url>/health` in the browser. It must return JSON with
   `status: "ok"`.

## 6. Run the Angular app locally against the API

Edit `DQC/app/proxy.aws.conf.json`, replacing `YOUR_API_GATEWAY_URL` with the
Invoke URL from step 5. Then run:

```bash
cd DQC/app
npm install
npx ng serve --proxy-config proxy.aws.conf.json --port 4200
```

Open `http://localhost:4200`. The frontend is local; all `/api/dqc/*` calls go
to API Gateway and Lambda.

## Lambda constraints

- API Gateway limits request and response payloads to 10 MB. Keep Excel files
  below that limit; use presigned S3 uploads for larger files.
- API Gateway does not preserve the app's SSE streaming behavior. The frontend
  falls back to a completed response only if a non-streaming endpoint is used.
  For live decision-tree streaming, use Lambda Function URLs with response
  streaming or retain the existing ECS deployment.
- A Lambda invocation cannot run longer than API Gateway's 29 seconds. Keep
  Bedrock generation short or move long-running generation to Step Functions.
