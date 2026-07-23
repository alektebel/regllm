# Serverless demo deploy — one public URL, no VPC

Deploys the DQC demo to AWS and gives you **one HTTPS URL to share** — the
whole thing runs on serverless building blocks, so it needs **no VPC, no
subnets, no NAT, no ALB, no ECS**. Use this when your account can't create a
VPC (locked-down / read-only-ish EC2), or you just want a quick shareable
link without running anything on your laptop.

```bash
AWS_REGION=eu-west-1 ./DQC/serverless/deploy.sh
# … ~10-15 min on first run …
# ✓ Share this URL:  https://d1234abcd.cloudfront.net
```

## Architecture

```text
                 https://<id>.cloudfront.net      (the URL you share)
                         │
              ┌──────────┴───────────┐
   default ("/*")                 "/api/*"
        │                              │  ← CloudFront Function strips "/api"
   S3 (private, OAC)         Lambda Function URL   (no API Gateway)
   Angular static site                 │
                                 Lambda (FastAPI via Mangum)
                                        │
                                  Amazon Bedrock
```

- **Frontend** — Angular built to static files, stored in a **private** S3
  bucket and served through CloudFront using Origin Access Control (the
  bucket is never public, so it works even with S3 Block Public Access on).
- **Backend** — the same FastAPI app (`REGLLM_ROUTERS=dqc`) packaged for
  Lambda via Mangum (`DQC/lambda/handler.py`), exposed through a **Lambda
  Function URL** (the function's own built-in HTTPS endpoint — **no API
  Gateway**, so no `apigateway:*` permission needed). The review store is
  SQLite on `/tmp` — **ephemeral** (no EFS, so no VPC).
- **The trick** — a tiny CloudFront Function (`cf-strip-api.js`) rewrites
  `/api/dqc/x` → `/dqc/x` at the edge, exactly like the local dev proxy
  (`proxy.conf.json`). Because the UI and the API share one CloudFront
  domain, there are **no CORS headers to configure and no frontend code
  changes** — the production build is identical to local dev.

## Prerequisites

On the machine running the script (all present in **AWS CloudShell**, which
is the easiest place to run this):

- `aws` CLI configured with credentials + a default region
- `python3` + `pip`, `node` + `npm`, `zip`
- **Bedrock model access** enabled in the region (Console → Bedrock → Model
  access → enable *Amazon Nova Micro*, or pass `BEDROCK_MODEL_ID=…`)

### IAM permissions it needs

These are all serverless/edge permissions — **none are `ec2:*` / VPC**:

- `lambda:*` (incl. `lambda:CreateFunctionUrlConfig` for the endpoint —
  **no API Gateway is used**)
- `iam:CreateRole`, `AttachRolePolicy`, `PutRolePolicy`, `PassRole`,
  `GetRole` (for the Lambda execution role)
- `s3:*` on the site bucket
- `cloudfront:*` (distributions, functions, origin-access-controls)
- `bedrock:InvokeModel*`

If your account is locked down enough that even these are denied (can't
create a VPC / API Gateway / Function URL), you can't host this on AWS at
all — use one of these instead, neither of which creates AWS infrastructure:

- **Share without deploying:** `./scripts/share_demo.sh` runs the app on
  your machine and exposes **one public https URL via a tunnel**
  (cloudflared/ngrok). Only needs Bedrock InvokeModel — no resource
  creation. The URL lives while your machine is running. Best for a live
  demo to other people.
- **Just run it yourself:** `./scripts/run_local.sh` (no public URL, no AWS
  resource creation; Bedrock, or `REGLLM_LLM=ollama`/`stub`).

## Configuration

Override via environment variables:

| Var | Default | Meaning |
|---|---|---|
| `AWS_REGION` | `eu-west-1` | Region for every resource |
| `PROJECT` | `regllm-dqc` | Name prefix for all resources |
| `BEDROCK_MODEL_ID` | `eu.amazon.nova-micro-v1:0` | Generation model |
| `INSPECT_BEDROCK_MODEL_ID` | = model | Cheap per-upload inspect model |
| `GEMINI_API_KEY` | — | If set, use Google Gemini instead of Bedrock |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Gemini model when the key is set |

## Re-deploying

Re-run `deploy.sh`. It's idempotent: it reuses the existing bucket, role,
Lambda, API, and CloudFront distribution, pushes the new code + static
files, and invalidates the CloudFront cache. Re-runs finish in a minute or
two (no 15-min wait).

## Tearing down

```bash
AWS_REGION=eu-west-1 ./DQC/serverless/destroy.sh
```

Removes everything (Lambda, API Gateway, S3 site, CloudFront distribution +
function + OAC, IAM role). Disabling a CloudFront distribution before delete
takes ~15 min — the script waits for you.

## Caveats (it's a demo, be honest about them)

- **Public** — the URL and the API are open to anyone with the link. Fine
  for a demo; don't leave it running indefinitely (it can burn Bedrock
  tokens). Tear it down when you're done.
- **No live streaming** — CloudFront buffers the response, so the
  decision-tree SSE animation isn't live; the tree appears when generation
  finishes. (The result is identical, just not streamed.)
- **~60-second limit** — CloudFront's origin read timeout is set to 60 s, so
  a single request must finish within that. Nova Micro on a small demo
  dictionary is well under it; very large inputs may time out.
- **Payload size** — a Lambda Function URL buffers up to ~6 MB per
  request/response. The demo Excels are a few KB; huge workbooks would need
  presigned-S3 uploads (not wired here).
- **Ephemeral data** — validated/rejected review state lives on Lambda
  `/tmp` and resets on cold starts. Durable state needs EFS, which *would*
  require a VPC — deliberately avoided here.

Need durability or live streaming later? That's the ECS/Fargate path in
`DQC/cdk/` — it needs an existing VPC (see `DQC/cdk/deploy.sh`).
