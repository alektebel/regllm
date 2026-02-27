#!/bin/bash
# Launch RegLLM local fine-tuned Qwen2.5-7B + Cloudflare public tunnel
#
# Usage:
#   ./launch_ui.sh               # port 7860, auto-detect latest adapter
#   ./launch_ui.sh --port 8080
#   ./launch_ui.sh --no-tunnel   # skip Cloudflare, local only
#   ./launch_ui.sh --share       # use Gradio's built-in share link instead

set -e
cd "$(dirname "$0")"

# ─── Defaults ─────────────────────────────────────────────────────────────────
PORT=7860
NO_TUNNEL=0
GRADIO_SHARE=""

for arg in "$@"; do
  case $arg in
    --port=*) PORT="${arg#*=}" ;;
    --port)   shift; PORT="$1" ;;
    --no-tunnel) NO_TUNNEL=1 ;;
    --share)  GRADIO_SHARE="--share"; NO_TUNNEL=1 ;;
  esac
done

# ─── Activate venv ────────────────────────────────────────────────────────────
if [ -f "regllm.env/bin/activate" ]; then
  source regllm.env/bin/activate
else
  echo "WARNING: venv not found at regllm.env/ — using system Python"
fi

# ─── Find model adapter ───────────────────────────────────────────────────────
MODEL_PATH=$(ls -td models/finetuned/run_*/final_model 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH=$(ls -td models/finetuned/run_*/checkpoint-* 2>/dev/null | head -1)
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/adapter_model.safetensors" ]; then
  echo ""
  echo "ERROR: No trained adapter found under models/finetuned/"
  echo ""
  echo "Available runs:"
  ls -1 models/finetuned/ 2>/dev/null || echo "  (none)"
  echo ""
  echo "Train first:  python scripts/train_combined.py"
  exit 1
fi

# ─── Find cloudflared ─────────────────────────────────────────────────────────
CLOUDFLARED=""
for candidate in "$HOME/bin/cloudflared" "/usr/local/bin/cloudflared" "/usr/bin/cloudflared" "$(which cloudflared 2>/dev/null)"; do
  if [ -x "$candidate" ]; then
    CLOUDFLARED="$candidate"
    break
  fi
done

if [ "$NO_TUNNEL" -eq 0 ] && [ -z "$CLOUDFLARED" ]; then
  echo "WARNING: cloudflared not found — skipping tunnel (use --no-tunnel to silence this)"
  NO_TUNNEL=1
fi

# ─── Kill any stale app.py (local) instance ──────────────────────────────────
STALE=$(pgrep -f "python.*app\.py" 2>/dev/null | tr '\n' ' ')
if [ -n "$STALE" ]; then
  echo "Killing stale app.py (PIDs: $STALE) to free GPU memory..."
  kill $STALE 2>/dev/null || true
  sleep 3
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RegLLM — Fine-tuned Qwen2.5-7B"
echo "  Adapter  : $MODEL_PATH"
echo "  Local    : http://localhost:$PORT"
[ "$NO_TUNNEL" -eq 0 ] && echo "  Tunnel   : https://regllm.xyz"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ─── Start Gradio app ─────────────────────────────────────────────────────────
echo "▶  Starting Gradio (loading model + RAG, may take 30-60 s)..."
python app.py \
  --backend local \
  --adapter "$MODEL_PATH" \
  --port "$PORT" \
  $GRADIO_SHARE &
GRADIO_PID=$!

# ─── Wait for Gradio to be ready ──────────────────────────────────────────────
echo "⏳  Waiting for Gradio on port $PORT..."
READY=0
for i in $(seq 1 90); do
  if curl -s --max-time 1 "http://localhost:$PORT" > /dev/null 2>&1; then
    echo "✓  Gradio ready → http://localhost:$PORT"
    READY=1
    break
  fi
  # Show spinner every 5 s
  if (( i % 5 == 0 )); then
    echo "   ... still loading ($i s)"
  fi
  sleep 1
done

if [ "$READY" -eq 0 ]; then
  echo "WARNING: Gradio did not respond after 90 s — tunnel may start before UI is ready"
fi

# ─── Start Cloudflare tunnel ──────────────────────────────────────────────────
TUNNEL_PID=""
if [ "$NO_TUNNEL" -eq 0 ]; then
  echo ""
  echo "  Starting Cloudflare tunnel..."
  echo "    https://regllm.xyz"
  echo ""
  "$CLOUDFLARED" tunnel run regllm 2>&1 | \
    grep --line-buffered -E "regllm\.xyz|ERR|error|INF" &
  TUNNEL_PID=$!
fi

# ─── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$GRADIO_PID"  ] && kill "$GRADIO_PID"  2>/dev/null || true
  [ -n "$TUNNEL_PID"  ] && kill "$TUNNEL_PID"  2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "Press Ctrl+C to stop."
wait "$GRADIO_PID"
