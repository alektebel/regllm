# Deploy to AWS — quickstart

The fastest path to a working DQC Generator on AWS. Three commands. For the
full explanation (architecture, troubleshooting, cost, tear-down) see
[`docs/AWS_POC_SETUP.md`](docs/AWS_POC_SETUP.md).

---

## Before you start (2 checks)

```bash
# 1. Authenticated to the right AWS account?
aws sts get-caller-identity          # should print YOUR account id

# 2. Docker running?
docker info >/dev/null && echo "docker OK"
```

You also need `node`/`npm` (for the CDK CLI) and Python 3.11+ installed.

## Step 1 — enable the model (one click, once per account)

AWS console → **Bedrock** → **Model access** → region **eu-west-1** →
enable **Amazon Nova Micro**.

> Skip this and the app deploys fine but every generation fails with
> `AccessDeniedException`.

## Step 2 — deploy (one command)

From the repo root:

```bash
AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh
```

This resolves your default VPC, creates all the infrastructure (ECR, ALB,
ECS Fargate, IAM, logs), builds and pushes both Docker images, and starts
the service. First run takes ~10 minutes. It ends by printing:

```
✓ Deployed.
  App URL:  http://<something>.eu-west-1.elb.amazonaws.com
```

## Step 3 — verify

```bash
# wait ~2 min after deploy, then (paste the App URL from step 2):
curl http://<something>.eu-west-1.elb.amazonaws.com/api/health
# → {"status":"ok","backend":"bedrock",...}
```

Open the **App URL** in a browser: the chat loads with the demo dictionary
and cases Excel already in place. Type a rule (or paste a few lines from
[`demo/reglas_demo.txt`](demo/reglas_demo.txt)) and press **Generar DQCs**.

---

## If something breaks

| Symptom | Fix |
|---|---|
| Deploy fails: `vpc/subnets` | Your account has no default VPC. Run `aws ec2 create-default-vpc --region eu-west-1` and re-run step 2. Corporate VPC? `VPC_ID=vpc-xxx SUBNET_IDS=subnet-a,subnet-b AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh`. |
| App URL loads but generation errors | The Bedrock model isn't enabled in `eu-west-1` (step 1), or you want a different one: add `-c bedrock_model_id=...` — but the simplest is to run the script with the model enabled. |
| Health check never turns OK | Usually the images hadn't been pushed on the first attempt — just re-run step 2 (it's idempotent). |
| Docker "exec format error" (Apple Silicon) | Edit `DQC/cdk/deploy.sh`, add `--platform linux/amd64` to the two `docker build` lines. |
| See live logs | `aws logs tail /ecs/regllm-dqc --follow --region eu-west-1` |

## Update after code changes

Re-run the same command — it rebuilds, pushes, and restarts:

```bash
AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh
```

## Shut it down (stops the ~€35–45/month cost)

```bash
cd DQC/cdk && cdk destroy --app "python3 app.py" \
  -c aws_region=eu-west-1 \
  -c aws_account=$(aws sts get-caller-identity --query Account --output text) \
  -c vpc_id=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region eu-west-1) \
  -c subnet_ids=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region eu-west-1) --query 'Subnets[].SubnetId' --output text --region eu-west-1 | tr '\t' ',') \
  --force
```

---

**Want to try it without deploying?** `./scripts/run_local.sh` runs the whole
app on your machine against Bedrock (your local AWS credentials). Prefer
Azure? `./DQC/azure/deploy.sh` — see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).
