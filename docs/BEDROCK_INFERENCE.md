# Use AWS Bedrock for all LLM inference

Step-by-step to route **every** LLM call in the app — generation, the
per-upload Excel inspection, and (optionally) embeddings — through Amazon
Bedrock, both locally and when deployed.

## How inference is wired

| Call | Function | Backend selector |
|---|---|---|
| DQC generation, sufficiency, judge | `get_client()` | `REGLLM_LLM` |
| Excel inspection / sheet mapping | `get_inspect_client()` | inherits `REGLLM_LLM`; model via `INSPECT_BEDROCK_MODEL_ID` |
| Embeddings (Tier-1 semantic field filter, opt-in) | `get_embedding_service()` | `REGLLM_EMBED_BACKEND` |

Setting **`REGLLM_LLM=bedrock`** forces every chat client onto Bedrock —
that single switch already covers generation *and* inspection. The only
extra knob is which Bedrock model each uses (below).

## Step 1 — enable the model(s) in Bedrock

AWS console → **Bedrock → Model access** (in your region, e.g.
`eu-west-1`) → enable the model you want:

- **Amazon Nova Micro** (`eu.amazon.nova-micro-v1:0`) — cheap/fast, the
  default; fine for the whole pipeline in a PoC.
- Optionally a stronger model (e.g. a **Claude** id) for better generation.
- For embeddings (only if you turn on the semantic field filter):
  **Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).

## Step 2 — give the app AWS credentials

Bedrock auth uses the standard AWS credential chain — pick one:

- **Deployed on ECS/Fargate:** nothing to do. The task role already grants
  `bedrock:InvokeModel*` (see `DQC/cdk/stacks/dqc_stack.py`).
- **Local:** `aws configure` / SSO / env vars — any identity the boto3
  chain can find, with Bedrock invoke permission.

## Step 3 — select the backend and models

### Local

```bash
export REGLLM_LLM=bedrock
export BEDROCK_REGION=eu-west-1
export BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0          # generation
# optional: keep inspection on a cheap model if generation uses a strong one
export INSPECT_BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0
./scripts/run_local.sh
```

`scripts/run_local.sh` already defaults `REGLLM_LLM=bedrock` +
`BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0`, so the bare
`./scripts/run_local.sh` is enough for the single-model case.

Or set it once in `config.yaml` instead of env vars:

```yaml
llm:
  backend: "bedrock"
  bedrock_model_id: "eu.amazon.nova-micro-v1:0"
  bedrock_region: "eu-west-1"
  # inspect_bedrock_model_id: "eu.amazon.nova-micro-v1:0"   # optional
```

### Deployed (ECS/Fargate)

The CDK stack already sets `REGLLM_LLM=bedrock` and passes the models as
container env. Choose them at deploy time:

```bash
# single model everywhere (simplest)
AWS_REGION=eu-west-1 ./DQC/cdk/deploy.sh
#   → uses eu.amazon.nova-micro-v1:0 for generation AND inspection

# stronger generation model, cheap inspection (recommended for quality+cost)
cd DQC/cdk && cdk deploy --app "python3 app.py" \
  -c aws_region=eu-west-1 -c aws_account=$(aws sts get-caller-identity --query Account --output text) \
  -c vpc_id=<VPC> -c subnet_ids=<SUBNETS> \
  -c bedrock_model_id=eu.anthropic.claude-3-5-sonnet-...-v1:0 \
  -c inspect_bedrock_model_id=eu.amazon.nova-micro-v1:0 \
  --require-approval never
```

`inspect_bedrock_model_id` defaults to `bedrock_model_id`, so you only pass
it when you want inspection on a cheaper model than generation.

## Step 4 — (optional) embeddings on Bedrock

Only needed if you enable the Tier-1 semantic field filter
(`REGLLM_DQC_SEMANTIC_FIELDS=1`). Point embeddings at Bedrock Titan:

```bash
export REGLLM_EMBED_BACKEND=bedrock
export REGLLM_EMBED_MODEL=amazon.titan-embed-text-v2:0
export REGLLM_DQC_SEMANTIC_FIELDS=1
```

Without this the filter degrades to lexical-only (zero vectors) — safe, no
error. Leave it off unless you've enabled Titan in Model access.

## Step 5 — verify it's really on Bedrock

```bash
curl http://localhost:8000/health          # local  (or /api/health via the UI)
# → {"status":"ok","backend":"bedrock", ...}
```

The health endpoint reports the resolved backend. Then upload the demo
dictionary and generate — both the inspection message and the DQCs come
from Bedrock. Deployed, the same check is `http://<alb-dns>/api/health`.

## Common issues

| Symptom | Cause / fix |
|---|---|
| `health` shows `"backend":"stub"` locally | `boto3` not installed, or `REGLLM_LLM` not `bedrock`. `pip install boto3`, re-export the var. |
| `AccessDeniedException` on generate | model not enabled in this region (step 1), or the model id doesn't match an enabled one. |
| Inspection slow / costs too much | generation model is strong and inspection reuses it — set `INSPECT_BEDROCK_MODEL_ID` / `-c inspect_bedrock_model_id` to Nova Micro. |
| Region mismatch | `BEDROCK_REGION` (local) or the stack region must be one where the model/inference-profile exists. |

See also: [`AWS_POC_SETUP.md`](AWS_POC_SETUP.md) (deploy),
[`DEPLOYMENT.md`](DEPLOYMENT.md) (image/router details).
