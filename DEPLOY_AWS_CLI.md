# AWS CLI deploy — Angular frontend + AWS backend

You have access keys now, so here's the full command pipeline, start to
finish. It deploys the **Angular frontend** and the **FastAPI backend**
(LLM on Amazon Bedrock) as one ECS Fargate service behind a public URL.

## 1. Configure the CLI (once)

```bash
aws configure
#   AWS Access Key ID     : <your key>
#   AWS Secret Access Key : <your secret>
#   Default region name   : eu-west-1
#   Default output format : json

aws sts get-caller-identity          # verify → prints your account id
```

## 2. Enable the model (once per account)

Console → **Bedrock → Model access** (region `eu-west-1`) → enable
**Amazon Nova Micro**. Or check from the CLI it's already enabled:

```bash
aws bedrock list-foundation-models --region eu-west-1 \
  --query "modelSummaries[?contains(modelId,'nova-micro')].modelId" --output text
```

## 3. Deploy — one command

Needs `docker`, `node`/`npm`, `python3` on your machine (all present in AWS
CloudShell too).

```bash
AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh
```

This does everything: resolves your default VPC, bootstraps CDK, creates
ECR + ALB + ECS, **builds and pushes both images** (backend from the root
`Dockerfile`, Angular frontend from `DQC/app/Dockerfile`), and starts the
service. First run ~10 min; it prints:

```
✓ Deployed.
  App URL:  http://<something>.eu-west-1.elb.amazonaws.com
```

## 4. Verify

```bash
APP=<paste the App URL from step 3>
curl "$APP/api/health"               # → {"status":"ok","backend":"bedrock",...}
open "$APP"                          # or paste in a browser
```

The chat loads with the demo dictionary + cases Excel already in place —
type rules (or paste `demo/reglas_demo.txt`) and press **Generar DQCs**.

## 5. Redeploy after changes / tear down

```bash
# push new code
AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh

# stop everything (halts the ~€35–45/mo cost)
cd DQC/cdk && cdk destroy --app "python3 app.py" \
  -c aws_region=eu-west-1 \
  -c aws_account=$(aws sts get-caller-identity --query Account --output text) \
  -c vpc_id=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region eu-west-1) \
  -c subnet_ids=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region eu-west-1) --query 'Subnets[].SubnetId' --output text --region eu-west-1 | tr '\t' ',') \
  --force
```

## If it breaks

| Symptom | Fix |
|---|---|
| `deploy` fails on vpc/subnets | `aws ec2 create-default-vpc --region eu-west-1`, re-run. Corporate VPC? prefix with `VPC_ID=vpc-xxx SUBNET_IDS=subnet-a,subnet-b`. |
| Generation errors (`AccessDeniedException`) | model not enabled in `eu-west-1` (step 2). |
| Health never OK | images weren't pushed on the first try — re-run step 3 (idempotent). |
| Docker "exec format error" (Apple Silicon) | add `--platform linux/amd64` to the two `docker build` lines in `DQC/cdk/deploy.sh`. |
| Watch live logs | `aws logs tail /ecs/regllm-dqc --follow --region eu-west-1` |

---

Full reference: [`docs/AWS_POC_SETUP.md`](docs/AWS_POC_SETUP.md) ·
permissions: [`docs/AWS_POC_SETUP.md#0b`](docs/AWS_POC_SETUP.md) ·
run locally instead: `./scripts/run_local.sh`.
