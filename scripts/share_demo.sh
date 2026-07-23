#!/usr/bin/env bash
# Run the DQC demo locally and expose ONE public https URL via a tunnel — so
# you can show it to other people WITHOUT deploying anything to AWS. Nothing
# here creates AWS infrastructure; the only AWS call is Bedrock InvokeModel
# (your existing credentials), so it works on a locked-down account that can't
# make VPCs / API Gateways / Function URLs.
#
# How it works: the Angular app is built to static files and served — together
# with an /api reverse-proxy to the local FastAPI backend — by one small local
# server (scripts/_demo_proxy.py). A tunnel points at that server, so the
# single public URL serves the UI and proxies the API on the same origin (no
# CORS), mirroring the CloudFront design but on your machine.
#
#   ./scripts/share_demo.sh
#
# Prerequisites:
#   - python3 (+ the DQC deps — install into the SAME interpreter:
#       python3 -m pip install -r requirements-dqc.txt uvicorn)
#   - node + npm (to build the frontend)
#   - a tunnel tool: cloudflared (preferred, no account) or ngrok
#       macOS:  brew install cloudflared
#       linux:  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
#   - AWS credentials with Bedrock InvokeModel access (aws configure)
#
# Env overrides: BEDROCK_MODEL_ID, BEDROCK_REGION, FRONTEND_PORT (4200),
#   BACKEND_PORT (8000), REGLLM_LLM (bedrock|ollama|stub).
set -euo pipefail

export REGLLM_LLM="${REGLLM_LLM:-bedrock}"
export BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-eu.amazon.nova-micro-v1:0}"
export BEDROCK_REGION="${BEDROCK_REGION:-eu-west-1}"
export INSPECT_BEDROCK_MODEL_ID="${INSPECT_BEDROCK_MODEL_ID:-$BEDROCK_MODEL_ID}"
export REGLLM_ROUTERS="${REGLLM_ROUTERS:-dqc}"
export CORS_ORIGINS="${CORS_ORIGINS:-*}"
FRONTEND_PORT="${FRONTEND_PORT:-4200}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
# Interpreter to run the backend with. Override if `python3` isn't the one you
# pip-installed into, e.g.  PYTHON=python3.11  or  PYTHON=./venv/bin/python
PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Prerequisites ─────────────────────────────────────────────────────────
command -v "$PYTHON" >/dev/null || { echo "✗ '$PYTHON' not found (set PYTHON=...)"; exit 1; }
command -v npm       >/dev/null || { echo "✗ node/npm required (to build the frontend)"; exit 1; }
if ! "$PYTHON" -c "import uvicorn, fastapi, openpyxl" 2>/dev/null; then
    WHICH="$("$PYTHON" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PYTHON")"
    echo "✗ backend deps aren't importable by the interpreter this script uses:"
    echo "    $WHICH"
    echo "  You likely pip-installed into a DIFFERENT Python. Install into THIS one:"
    echo "    $PYTHON -m pip install -r requirements-dqc.txt uvicorn"
    echo "  …or point the script at the Python that has them:"
    echo "    PYTHON=/path/to/your/python ./scripts/share_demo.sh"
    echo "    (if you used a venv, 'source venv/bin/activate' first)"
    exit 1
fi

TUNNEL=""
if   command -v cloudflared >/dev/null; then TUNNEL=cloudflared
elif command -v ngrok       >/dev/null; then TUNNEL=ngrok
else
    echo "✗ no tunnel tool found. Install one:"
    echo "    cloudflared (no account):  brew install cloudflared"
    echo "        or  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    echo "    ngrok:  https://ngrok.com/download"
    exit 1
fi

if [[ "$REGLLM_LLM" == "bedrock" ]]; then
    if command -v aws >/dev/null && aws sts get-caller-identity >/dev/null 2>&1; then
        echo "• AWS creds OK — Bedrock $BEDROCK_MODEL_ID ($BEDROCK_REGION)"
        echo "  (ensure the model is enabled: console → Bedrock → Model access)"
    else
        echo "⚠ AWS not authenticated — Bedrock calls will fail."
        echo "  Run 'aws configure', or use REGLLM_LLM=ollama / REGLLM_LLM=stub."
    fi
fi

# ── Clean shutdown of every child on Ctrl-C / exit ────────────────────────
PIDS=()
cleanup() { echo; echo "• stopping…"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

# ── 1. Build the frontend (the real production build) ─────────────────────
echo "• building the Angular frontend…"
( cd "$ROOT/DQC/app" && { [[ -d node_modules ]] || npm ci; } && npx ng build --configuration production )
DIST="$ROOT/DQC/app/dist/dqc-app/browser"
[[ -f "$DIST/index.html" ]] || { echo "✗ build missing $DIST/index.html"; exit 1; }

# ── 2. Backend (FastAPI on localhost) ─────────────────────────────────────
echo "• backend → http://localhost:$BACKEND_PORT"
( cd "$ROOT" && exec "$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port "$BACKEND_PORT" ) &
PIDS+=($!)

# ── 3. Static + /api proxy server (what the tunnel points at) ─────────────
echo "• demo server → http://localhost:$FRONTEND_PORT  (serves UI + proxies /api)"
DIST_DIR="$DIST" BACKEND="http://127.0.0.1:$BACKEND_PORT" PORT="$FRONTEND_PORT" \
    "$PYTHON" "$ROOT/scripts/_demo_proxy.py" &
PIDS+=($!)

# Wait for the demo server to answer before opening the tunnel
for _ in $(seq 1 30); do
    curl -sf "http://127.0.0.1:$FRONTEND_PORT" >/dev/null 2>&1 && break
    sleep 1
done

# ── 4. Public tunnel ──────────────────────────────────────────────────────
echo ""
echo "• opening a public tunnel with $TUNNEL — the https URL it prints is the"
echo "  one you share. Press Ctrl-C here to stop everything."
echo ""
if [[ "$TUNNEL" == cloudflared ]]; then
    exec cloudflared tunnel --url "http://localhost:$FRONTEND_PORT"
else
    exec ngrok http "$FRONTEND_PORT"
fi
