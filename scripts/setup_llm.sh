#!/usr/bin/env bash
# Set up a local LLM for the GraphRAG justifier.
#
# Default path: pull Qwen 2.5 14B Instruct (~8 GB) into a running Ollama
# server. This is fast, fits on a single 24 GB GPU, and is excellent at
# structured-JSON output — which is exactly what the GraphRAG verdict needs.
#
# Optional alternative: Google Gemma 4 12B served by LiteRT-LM
# (https://developers.google.com/edge/litert-lm/cli/openai_server) on
# http://localhost:9379. Pass `--gemma` to use that instead.
#
# Both backends are auto-detected by the API client at runtime.
set -euo pipefail

MODE="${1:-qwen}"
case "$MODE" in
  --gemma|gemma) MODE="gemma" ;;
  --qwen|qwen|"") MODE="qwen" ;;
  -h|--help)
    cat <<EOF
Usage: $0 [qwen|gemma]

  qwen   (default)  Pull qwen2.5:14b-instruct-q4_K_M into Ollama
  gemma             Install LiteRT-LM and serve Google Gemma 4 12B

Both servers are auto-detected by the API client. The active one shows
up in the top-right "local LLM" pill of the /diff page.
EOF
    exit 0
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac

if [[ "$MODE" == "qwen" ]]; then
  MODEL="${OLLAMA_MODEL:-qwen2.5:14b-instruct-q4_K_M}"
  echo "[setup_llm] Using Ollama with model: $MODEL"
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is not installed. Install it from https://ollama.com first." >&2
    exit 1
  fi
  if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
    echo "[setup_llm] Ollama daemon not running. Start it in another terminal with:"
    echo "    ollama serve"
    exit 1
  fi
  if ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
    echo "[setup_llm] $MODEL is already pulled."
  else
    echo "[setup_llm] Pulling $MODEL (this may take a few minutes)…"
    ollama pull "$MODEL"
  fi
  echo
  echo "Done. Smoke-test:"
  echo "  curl http://localhost:11434/api/chat -H 'Content-Type: application/json' \\"
  echo "    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}'"
  exit 0
fi

# Gemma 4 12B via LiteRT-LM
GEMMA_REPO="litert-community/gemma-4-12B-it-litert-lm"
GEMMA_LOCAL_NAME="gemma4-12b"
GEMMA_FILE="gemma-4-12B-it.litertlm"

echo "[setup_llm] Checking for litert-lm…"
if ! command -v litert-lm >/dev/null 2>&1; then
  echo "[setup_llm] litert-lm not found. Installing…"
  pip install --user --upgrade litert-lm
  echo "[setup_llm] Installed. Make sure ~/.local/bin is on PATH."
fi

echo "[setup_llm] Importing $GEMMA_REPO from Hugging Face…"
litert-lm import \
  --from-huggingface-repo="$GEMMA_REPO" \
  "$GEMMA_FILE" \
  "$GEMMA_LOCAL_NAME"

echo "[setup_llm] Starting OpenAI-compatible server on http://localhost:9379"
exec litert-lm serve
