#!/bin/bash
# launch_ollama.sh — Start RegLLM with Ollama backend + Cloudflare public tunnel
#
# First time: build the model with ./scripts/build_ollama_model.sh
#
# Usage:
#   ./launch_ollama.sh               # port 7860, streaming UI
#   ./launch_ollama.sh --port 8080   # custom port

set -e
cd "$(dirname "$0")"

PORT=7860

for arg in "$@"; do
  case $arg in
    --port=*) PORT="${arg#*=}" ;;
    --port)   shift; PORT="$1" ;;
  esac
done

# ─── Verify GGUF exists ───────────────────────────────────────────────────────
GGUF="models/gguf/regllm-q4_k_m.gguf"
if [ ! -f "$GGUF" ]; then
  echo ""
  echo "❌  GGUF model not found: $GGUF"
  echo ""
  echo "   Build it first:"
  echo "   ./scripts/build_ollama_model.sh"
  echo ""
  exit 1
fi

# ─── Verify Ollama model is registered ───────────────────────────────────────
if ! ollama list 2>/dev/null | grep -q "^regllm"; then
  echo ""
  echo "⚠️   regllm not found in Ollama — registering now..."
  # Need serve running to register
  ollama serve >/tmp/ollama_tmp.log 2>&1 &
  TEMP_SERVE=$!
  sleep 3
  ollama create regllm -f Modelfile
  kill "$TEMP_SERVE" 2>/dev/null || true
  sleep 1
  echo "✓  regllm registered"
fi

echo ""
echo "🚀  Starting RegLLM + Ollama..."
echo "    Model  : regllm (Q4_K_M)"
echo "    Local  : http://localhost:$PORT"
echo ""

# ─── Activate venv ────────────────────────────────────────────────────────────
source regllm.env/bin/activate

# ─── Start Ollama server ──────────────────────────────────────────────────────
if curl -s http://localhost:11434 &>/dev/null; then
  echo "✓  Ollama already running"
  OLLAMA_PID=""
else
  echo "▶  Starting ollama serve..."
  ollama serve >/tmp/ollama.log 2>&1 &
  OLLAMA_PID=$!
  for i in $(seq 1 20); do
    if curl -s http://localhost:11434 &>/dev/null; then
      echo "✓  Ollama server up"
      break
    fi
    sleep 1
  done
fi

# ─── Start Gradio ─────────────────────────────────────────────────────────────
echo "▶  Starting Gradio UI..."
python app.py --backend ollama --port "$PORT" &
GRADIO_PID=$!

echo "⏳  Waiting for Gradio (RAG init can take ~20 s)..."
for i in $(seq 1 60); do
  if curl -s "http://localhost:$PORT" &>/dev/null; then
    echo "✓  Gradio is up → http://localhost:$PORT"
    break
  fi
  sleep 1
done

# ─── Start Cloudflare tunnel ──────────────────────────────────────────────────
echo ""
echo "🌍  Starting Cloudflare tunnel..."
echo "    Your public URL will appear below (*.trycloudflare.com)"
echo ""
~/bin/cloudflared tunnel --url "http://localhost:$PORT" &
TUNNEL_PID=$!

# ─── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$GRADIO_PID" "$TUNNEL_PID" 2>/dev/null || true
  [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "Press Ctrl+C to stop."
wait "$GRADIO_PID"
