# AWS deployment — PoC setup guide

Step-by-step instructions to stand up the DQC Generator PoC on a clean
AWS account, and to run it locally against Bedrock during development.

**What gets deployed:** one ECS Fargate task with two containers behind a
public Application Load Balancer —

| Container | Image source | Role |
|---|---|---|
| `api` | root `Dockerfile` (slim, `requirements-dqc.txt`) | FastAPI backend, port 8000, LLM on **Amazon Bedrock** (`REGLLM_LLM=bedrock`) |
| `dqc` | `DQC/app/Dockerfile` | Angular UI + nginx on port 80, proxies `/api` → localhost:8000; bundles the demo dictionary/cases Excels the UI auto-loads |

Everything is created by the CDK stack in `DQC/cdk/` (ECR repos, ALB, ECS
cluster/service, IAM roles with Bedrock invoke permissions, CloudWatch
logs). The URL of the PoC is the ALB's DNS name — no domain needed.

---

## 0. Prerequisites (your machine)

- **AWS CLI v2**, authenticated: `aws configure` (or SSO), then verify
  with `aws sts get-caller-identity`.
- **Docker** (images are built locally and pushed).
- **Node.js + npm** (for the CDK CLI) and **Python 3.11+**.
- IAM permissions to create CloudFormation/ECS/ECR/EC2/IAM/Logs resources
  (admin on a sandbox account is the simple answer for a PoC).

## 1. Enable the Bedrock model (one-time, console)

AWS console → **Bedrock** → *Model access* (region `eu-west-1`) → enable
**Amazon Nova Micro** (`eu.amazon.nova-micro-v1:0`, the stack default).
Optionally enable a Claude model too and pass it at deploy time with
`-c bedrock_model_id=...` for better generation quality.

Without this step the app deploys fine but every LLM call fails.

## 2. Deploy — one command

```bash
AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh
```

The script is idempotent and does, in order:

1. Resolves your **default VPC and subnets** (creates a default VPC if the
   account has none). Override with `VPC_ID=vpc-... SUBNET_IDS=subnet-a,subnet-b`
   for a corporate VPC.
2. `cdk bootstrap` (first run only) and `cdk deploy` → all infrastructure.
3. Builds and pushes both Docker images to the CDK-created ECR repos.
   On an Apple Silicon Mac Docker builds the right arch automatically via
   the script's plain `docker build`; if you hit an exec format error add
   `--platform linux/amd64` to the two builds.
4. Forces an ECS redeploy so the service pulls the fresh images.

It ends by printing:

```
✓ Deployed.
  App URL:  http://<alb-dns>.eu-west-1.elb.amazonaws.com
  Health:   http://<alb-dns>.eu-west-1.elb.amazonaws.com/api/health
```

Give the task ~2 minutes to pass health checks, then open the App URL:
the chat loads with the demo dictionary and cases Excel already in place —
type rules (or paste `demo/reglas_demo.txt`) and press **Generar DQCs**.

## 3. Updating the PoC (re-upload)

Re-run the same command after any code change:

```bash
AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh
```

It rebuilds the images, pushes `:latest`, and restarts the service. The
infrastructure is only changed if the CDK stack itself changed.

## 4. Troubleshooting

- **Logs:** CloudWatch → log group `/ecs/regllm-dqc` (streams `api/…` and
  `dqc/…`), or
  `aws logs tail /ecs/regllm-dqc --follow --region eu-west-1`.
- **LLM errors ("AccessDeniedException")** → the Bedrock model isn't
  enabled in this region (step 1), or the model id doesn't match an
  enabled one.
- **Service never healthy** → almost always empty ECR repos (deploy ran
  before images were pushed): re-run the script.
- **Task role** already covers cross-region inference profiles
  (`eu.amazon.nova-micro-v1:0` routes across EU regions).

## 5. Tear-down

```bash
cd DQC/cdk
cdk destroy --app "python3 app.py" \
  -c aws_region=eu-west-1 \
  -c aws_account=$(aws sts get-caller-identity --query Account --output text) \
  -c vpc_id=<VPC_ID> -c subnet_ids=<SUBNETS> --force
```

ECR repos auto-empty on delete, so destroy is clean. Idle cost while it
runs: ~€35–45/month (Fargate 1 vCPU/4 GB ≈ €18, ALB ≈ €18) plus per-token
Bedrock usage (Nova Micro: fractions of a cent per request).

## 6. PoC caveats (accepted for a PoC, fix before anything real)

- The ALB is **plain HTTP, open to the internet** — add an ACM cert +
  HTTPS listener and restrict the security group before sharing widely.
- Generated DQCs (and their decision traces / detected cases) persist in
  **SQLite inside the container** — lost on every redeploy. EFS or RDS is
  the durable fix.
- One task, no autoscaling (`desired_count=1`).

## 7. Local development against Bedrock (no deployment)

```bash
./scripts/run_local.sh
```

Backend on :8000 (`REGLLM_LLM=bedrock`, Nova Micro, your local AWS
credentials), frontend on :4200 proxying `/api`. Overrides:
`BEDROCK_MODEL_ID`, `BEDROCK_REGION`, or `REGLLM_LLM=ollama|stub` to work
offline. For a fully scripted, model-free walkthrough of every pipeline
branch, see [`demo/README.md`](../demo/README.md).

An Azure equivalent (Container Apps + Azure OpenAI) exists at
`DQC/azure/deploy.sh` — see [`docs/DEPLOYMENT.md`](DEPLOYMENT.md).
