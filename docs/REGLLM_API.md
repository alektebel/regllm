# Use a custom HTTP inference API (REGLLM_API)

Point all of the app's LLM inference at **your own HTTP endpoint** — an AWS
API Gateway that fronts Bedrock, an internal "LLM gateway", or an
OpenAI-compatible proxy such as
[AWS Bedrock Access Gateway](https://github.com/aws-samples/bedrock-access-gateway)
— instead of calling `bedrock-runtime` directly.

## What it expects

The `api` backend speaks **OpenAI chat-completions** by default (the most
common gateway contract, and what Bedrock Access Gateway exposes):

- **Request** — `POST` with
  `{"model", "messages": [{"role","content"}], "temperature", "max_tokens", "response_format"?}`
- **Response** — `{"choices": [{"message": {"content": "…"}}]}`
  (it also tolerates `output_text` / `content` / `completion` top-level
  fields for lightly-custom gateways).

If your gateway uses a different JSON shape, tell us and we'll add a
format adapter — the parsing lives in `LocalLLMClient._chat_api`.

## Configure it

Env vars (or the `llm.api_*` keys in `config.yaml`):

| Variable | Meaning | Example |
|---|---|---|
| `REGLLM_LLM` | select this backend | `api` |
| `REGLLM_API_URL` (or `REGLLM_API`) | endpoint. A bare base URL gets `/v1/chat/completions` appended; a URL with a path is used as-is | `https://abc123.execute-api.eu-west-1.amazonaws.com/prod` |
| `REGLLM_API_KEY` | auth secret (optional) | `…` |
| `REGLLM_API_KEY_HEADER` | header for the key. Default `Authorization` → sent as `Bearer <key>`; set to `x-api-key` for API-Gateway keys | `x-api-key` |
| `REGLLM_API_MODEL` | model name the gateway expects | `amazon.nova-micro-v1:0` |

### Local

```bash
export REGLLM_LLM=api
export REGLLM_API_URL="https://abc123.execute-api.eu-west-1.amazonaws.com/prod"
export REGLLM_API_KEY="…"                 # if your API needs one
export REGLLM_API_KEY_HEADER="x-api-key"  # API Gateway usage-plan keys
export REGLLM_API_MODEL="amazon.nova-micro-v1:0"
./scripts/run_local.sh                    # (it won't override REGLLM_LLM if you set it)
```

### config.yaml

```yaml
llm:
  backend: "api"
  api_url: "https://abc123.execute-api.eu-west-1.amazonaws.com/prod"
  api_key_header: "x-api-key"
  api_model: "amazon.nova-micro-v1:0"
  # api_key: "…"    # prefer the REGLLM_API_KEY env var / a secret
```

### Deployed (ECS)

Add the same vars to the `api` container environment in
`DQC/cdk/stacks/dqc_stack.py` (replacing the `REGLLM_LLM=bedrock` /
`BEDROCK_*` block), and put `REGLLM_API_KEY` in a Secrets Manager secret
rather than plaintext env.

## What this covers

Setting `REGLLM_LLM=api` routes **every** chat call through the gateway —
generation, sufficiency, the semantic judge, and the per-upload Excel
inspection (all `LocalLLMClient`s inherit the backend). Embeddings for the
optional Tier-1 semantic filter are separate; if your gateway also serves
embeddings, that path would need its own adapter (not wired yet).

## Auth notes

- **API key / Bearer** — covered by `REGLLM_API_KEY` + `REGLLM_API_KEY_HEADER`.
- **IAM / SigV4-signed API Gateway** — not built in yet (needs request
  signing with botocore). If that's your setup, say so and it's a small
  addition to `_api_headers`.

## Verify

```bash
curl http://localhost:8000/health     # → {"status":"ok","backend":"api", ...}
```

Then generate DQCs — both the inspection message and the checks come from
your gateway.
