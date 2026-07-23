#!/usr/bin/env bash
# Run the DQC demo locally and expose ONE public https URL via a tunnel — so
# you can show it to other people WITHOUT deploying anything to AWS. Nothing
# here creates AWS infrastructure; the only AWS call is Bedrock InvokeModel
# (your existing credentials), so it works on a locked-down account that can't
# make VPCs / API Gateways / Function URLs.
#
# How it works: the FastAPI backend runs on localhost, and the Angular app is
# served by its dev server (`ng serve`) on one port, which proxies /api to the
# backend (same origin, no CORS). A tunnel — or your LAN — points at that one
# port, so a single URL serves the UI and the API. `--allowed-hosts` lets the
# dev server accept tunnel/LAN hostnames.
#
#   ./scripts/share_demo.sh                 # auto: cloudflared → ngrok → LAN
#   TUNNEL=ngrok ./scripts/share_demo.sh    # force ngrok
#   TUNNEL=lan   ./scripts/share_demo.sh    # no public URL; serve on your LAN
#
# Prerequisites:
#   - python3 (+ the DQC deps — install into the SAME interpreter:
#       python3 -m pip install -r requirements-dqc.txt uvicorn)
#   - node + npm (runs the Angular dev server)
#   - for a PUBLIC url, a tunnel tool (not needed for TUNNEL=lan):
#       cloudflared (no account):  brew install cloudflared  · or
#         https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
#       ngrok (free acct + authtoken):  https://ngrok.com/download
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
# The demo needs no durable data. Keep the review store (SQLite) in a
# throwaway temp file so every run starts clean and there's never a write
# issue on a path inside the repo. Generation still works — results stream
# back in the response; the store only holds validate/reject review state.
export REGLLM_CHECKS_DB="${REGLLM_CHECKS_DB:-$(mktemp -u --suffix=.db 2>/dev/null || mktemp -u)}"
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

# How to expose the demo. TUNNEL=auto (default) prefers cloudflared, then
# ngrok, then falls back to LAN. Force one with TUNNEL=cloudflared|ngrok|lan.
#   lan  = no public tunnel; serve on your local network (same Wi-Fi/LAN).
MODE="${TUNNEL:-auto}"
case "$MODE" in
    lan|none|local) MODE=lan ;;
    cloudflared) command -v cloudflared >/dev/null || { echo "✗ cloudflared not found"; exit 1; } ;;
    ngrok)       command -v ngrok       >/dev/null || { echo "✗ ngrok not found (https://ngrok.com/download)"; exit 1; } ;;
    auto)
        if   command -v cloudflared >/dev/null; then MODE=cloudflared
        elif command -v ngrok       >/dev/null; then MODE=ngrok
        else
            echo "• no tunnel tool found — serving on your LOCAL NETWORK instead."
            echo "  (for a public URL: install cloudflared/ngrok, or set TUNNEL=ngrok)"
            MODE=lan
        fi ;;
    *) echo "✗ unknown TUNNEL='$MODE' (use cloudflared | ngrok | lan)"; exit 1 ;;
esac

if [[ "$REGLLM_LLM" == "bedrock" ]]; then
    if command -v aws >/dev/null && aws sts get-caller-identity >/dev/null 2>&1; then
        echo "• AWS creds OK — Bedrock $BEDROCK_MODEL_ID ($BEDROCK_REGION)"
        echo "  (ensure the model is enabled: console → Bedrock → Model access)"
    else
        echo "⚠ AWS not authenticated — Bedrock calls will fail."
        echo "  Run 'aws configure', or use REGLLM_LLM=ollama / REGLLM_LLM=stub."
    fi
fi

# ── Free the ports we need (kill stragglers from a previous run) ──────────
free_port() {
    local port="$1" pids=""
    if   command -v lsof  >/dev/null; then pids="$(lsof -ti "tcp:$port" 2>/dev/null || true)"
    elif command -v fuser >/dev/null; then pids="$(fuser "$port/tcp" 2>/dev/null | tr -s ' ' '\n' || true)"
    fi
    [[ -n "$pids" ]] || return 0
    echo "• port $port busy — stopping stale process(es): $(echo $pids | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    for p in $pids; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null || true; done
}
free_port "$BACKEND_PORT"
free_port "$FRONTEND_PORT"

# ── Clean shutdown of every child on Ctrl-C / exit ────────────────────────
PIDS=()
cleanup() { echo; echo "• stopping…"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; rm -f "${PROXY_CFG:-}" "${REGLLM_CHECKS_DB:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# ── 1. Backend (FastAPI on localhost) ─────────────────────────────────────
echo "• backend → http://localhost:$BACKEND_PORT"
( cd "$ROOT" && exec "$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port "$BACKEND_PORT" ) &
PIDS+=($!)

# Wait for the backend to actually answer before serving the UI.
echo "• waiting for the backend to come up…"
backend_ok=""
for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then backend_ok=1; break; fi
    kill -0 "${PIDS[-1]}" 2>/dev/null || { echo "✗ backend process exited — see its error above (bad deps? port still busy?)."; exit 1; }
    sleep 1
done
[[ -n "$backend_ok" ]] || echo "⚠ backend not healthy on :$BACKEND_PORT yet — /api calls may fail until it is."

# ── 2. Frontend via Angular dev server (proxies /api → backend) ───────────
# A temp proxy config points /api at the chosen backend port and strips the
# /api prefix (the backend serves /dqc, /health, …). --allowed-hosts lets the
# dev server accept tunnel/LAN hostnames.
echo "• installing frontend deps if needed…"
( cd "$ROOT/DQC/app" && [[ -d node_modules ]] || ( cd "$ROOT/DQC/app" && npm ci ) )
PROXY_CFG="$(mktemp --suffix=.json 2>/dev/null || mktemp)"
cat >"$PROXY_CFG" <<JSON
{ "/api": { "target": "http://127.0.0.1:$BACKEND_PORT", "secure": false, "pathRewrite": { "^/api": "" } } }
JSON

echo "• starting the Angular dev server on :$FRONTEND_PORT (first run takes a bit)…"
( cd "$ROOT/DQC/app" && exec npx ng serve \
    --host 0.0.0.0 --port "$FRONTEND_PORT" --allowed-hosts \
    --proxy-config "$PROXY_CFG" ) &
PIDS+=($!)

# Wait for the dev server to actually answer, and confirm it (or fail loud)
demo_ok=""
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$FRONTEND_PORT" >/dev/null 2>&1; then demo_ok=1; break; fi
    kill -0 "${PIDS[-1]}" 2>/dev/null || { echo "✗ dev server exited — see its error above."; exit 1; }
    sleep 1
done
if [[ -z "$demo_ok" ]]; then
    echo "✗ dev server isn't responding on :$FRONTEND_PORT after 2 min. See output above."
    exit 1
fi
echo ""
echo "  ✓ READY — the demo is now serving on port $FRONTEND_PORT."

# ── 4. Expose it (public tunnel, or local network) ────────────────────────
echo ""
case "$MODE" in
    cloudflared)
        echo "• opening a public tunnel with cloudflared — share the https URL it"
        echo "  prints below. Ctrl-C to stop everything."
        echo ""
        exec cloudflared tunnel --url "http://localhost:$FRONTEND_PORT" ;;
    ngrok)
        echo "• opening a public tunnel with ngrok — share the https URL it shows."
        echo "  (first time only: ngrok config add-authtoken <token> — free at"
        echo "   https://dashboard.ngrok.com). Ctrl-C to stop everything."
        echo ""
        exec ngrok http "$FRONTEND_PORT" ;;
    lan)
        LAN_IP="$("$PYTHON" -c 'import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8",80)); print(s.getsockname()[0])
except Exception:
    print("YOUR-IP")
finally:
    s.close()' 2>/dev/null || echo "YOUR-IP")"
        echo "✓ Serving on your LOCAL NETWORK — no public tunnel."
        echo "  People on the SAME Wi-Fi/LAN can open:"
        echo ""
        echo "        http://${LAN_IP}:${FRONTEND_PORT}"
        echo ""
        echo "  On this machine:  http://localhost:${FRONTEND_PORT}"
        echo "  Can't reach it from another device? Your OS firewall may be"
        echo "  blocking port ${FRONTEND_PORT} — allow it, or check you're on the"
        echo "  same network (this won't work over the public internet)."
        echo "  Ctrl-C to stop."
        wait ;;
esac
