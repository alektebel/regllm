#!/bin/bash
# launch_groq.sh — Start RegLLM with Groq backend + Cloudflare public tunnel
#
# Usage:
#   ./launch_groq.sh
#
# Requirements:
#   - GROQ_API_KEY in .env (get free key at https://console.groq.com)
#   - cloudflared at ~/bin/cloudflared (already installed)

set -e
cd "$(dirname "$0")"

PORT=7860

# Check API key
source .env 2>/dev/null || true
if [ -z "$GROQ_API_KEY" ]; then
  echo ""
  echo "❌  GROQ_API_KEY not set!"
  echo "    1. Go to https://console.groq.com → API Keys → Create key"
  echo "    2. Add to regllm/.env:  GROQ_API_KEY=gsk_..."
  echo ""
  exit 1
fi

echo ""
echo "🚀  Starting RegLLM + Groq..."
echo "    Model  : llama-3.3-70b-versatile (free)"
echo "    Local  : http://localhost:$PORT"
echo ""

# Activate env
source regllm.env/bin/activate

# Launch Gradio in background
python app.py --backend groq --port $PORT &
GRADIO_PID=$!

# Wait for Gradio to be ready
echo "⏳  Waiting for Gradio to start..."
for i in $(seq 1 30); do
  if curl -s http://localhost:$PORT > /dev/null 2>&1; then
    echo "✅  Gradio is up"
    break
  fi
  sleep 1
done

# Launch cloudflare tunnel
echo "🌍  Starting Cloudflare tunnel..."
echo "    Your public URL will appear below (*.trycloudflare.com)"
echo ""
~/bin/cloudflared tunnel --url http://localhost:$PORT &
TUNNEL_PID=$!

# Cleanup on exit
trap "echo 'Shutting down...'; kill $GRADIO_PID $TUNNEL_PID 2>/dev/null" EXIT INT TERM

echo ""
echo "Press Ctrl+C to stop."
wait $GRADIO_PID
