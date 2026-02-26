#!/bin/bash
# scripts/build_ollama_model.sh — Build regllm Ollama model from LoRA adapter
#
# Pipeline:
#   1. Merge LoRA adapter → full HF weights  (models/merged/, ~14 GB)
#   2. Setup llama.cpp  (clones + builds if not found at ~/llama.cpp)
#   3. Convert merged model → GGUF F16        (models/gguf/regllm-f16.gguf)
#   4. Quantize → Q4_K_M                      (models/gguf/regllm-q4_k_m.gguf, ~4 GB)
#   5. Write Modelfile and run: ollama create regllm
#
# Usage:
#   ./scripts/build_ollama_model.sh
#   ./scripts/build_ollama_model.sh --skip-merge      # models/merged/ already exists
#   ./scripts/build_ollama_model.sh --skip-convert    # GGUF already built
#
# Requirements:
#   - ~20 GB free disk  (14 GB merged + 4 GB GGUF; F16 temp is deleted after quant)
#   - ~14 GB RAM  (for merge step; use --device cuda if you have 16 GB+ VRAM)
#   - cmake + a C++ compiler  (for llama.cpp build)
#   - Ollama already installed  (already present at /snap/bin/ollama)

set -euo pipefail
cd "$(dirname "$0")/.."

# ─── Flags ────────────────────────────────────────────────────────────────────
SKIP_MERGE=0
SKIP_CONVERT=0
SKIP_QUANTIZE=0
MERGE_DEVICE="cpu"

for arg in "$@"; do
  case $arg in
    --skip-merge)    SKIP_MERGE=1 ;;
    --skip-convert)  SKIP_CONVERT=1 ; SKIP_QUANTIZE=1 ;;
    --device=*)      MERGE_DEVICE="${arg#*=}" ;;
    --help|-h)
      echo "Usage: $0 [--skip-merge] [--skip-convert] [--device=cpu|cuda]"
      exit 0
      ;;
  esac
done

LLAMA_CPP_DIR="$HOME/llama.cpp"
MERGED_DIR="$(pwd)/models/merged"
GGUF_DIR="$(pwd)/models/gguf"
GGUF_F16="$GGUF_DIR/regllm-f16.gguf"
GGUF_Q4="$GGUF_DIR/regllm-q4_k_m.gguf"

mkdir -p "$GGUF_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RegLLM → Ollama build pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─── Verify Ollama ────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo ""
  echo "❌  Ollama not found in PATH. Install with:"
  echo "    curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi
echo ""
echo "✓  Ollama: $(ollama --version 2>/dev/null || echo 'found')"

# ─── Step 1: Merge LoRA adapter ───────────────────────────────────────────────
if [ "$SKIP_MERGE" -eq 0 ]; then
  if [ -f "$MERGED_DIR/config.json" ]; then
    echo "✓  Merged model exists ($MERGED_DIR) — skipping"
    echo "   (delete models/merged/ to redo)"
  else
    echo ""
    echo "▶  [1/4] Merging LoRA adapter into base model..."
    echo "   Device: $MERGE_DEVICE  |  Requires ~14 GB RAM  |  ~5-10 min on CPU"
    echo ""
    source regllm.env/bin/activate
    python scripts/merge_adapter.py --device "$MERGE_DEVICE"
    echo ""
    echo "✓  Merge complete"
  fi
else
  echo "⏭  Skipping merge (--skip-merge)"
  if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "⚠️  Warning: models/merged/config.json not found — convert step may fail"
  fi
fi

# ─── Step 2: Setup llama.cpp ──────────────────────────────────────────────────
echo ""
echo "▶  [2/4] Setting up llama.cpp..."

LLAMA_QUANTIZE=""
CONVERT_SCRIPT=""

# Search for existing llama-quantize binary
for candidate in \
  "$(command -v llama-quantize 2>/dev/null || true)" \
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  "$LLAMA_CPP_DIR/build/bin/llama-quantize.exe"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    LLAMA_QUANTIZE="$candidate"
    break
  fi
done

# Search for convert script
for candidate in \
  "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$LLAMA_CPP_DIR/convert-hf-to-gguf.py"; do
  if [ -f "$candidate" ]; then
    CONVERT_SCRIPT="$candidate"
    break
  fi
done

if [ -z "$LLAMA_QUANTIZE" ] || [ -z "$CONVERT_SCRIPT" ]; then
  echo "   llama.cpp not found — cloning and building (~5 min)..."

  if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
  else
    echo "   Existing clone found — pulling latest..."
    git -C "$LLAMA_CPP_DIR" pull --ff-only 2>/dev/null || true
  fi

  # Python deps for convert script
  source regllm.env/bin/activate
  pip install -q gguf sentencepiece protobuf

  # Detect CUDA and build
  if command -v nvcc &>/dev/null || [ -d "/usr/local/cuda" ]; then
    echo "   CUDA detected — enabling GPU support in llama.cpp"
    CMAKE_FLAGS="-DGGML_CUDA=ON"
  else
    echo "   Building CPU-only llama.cpp"
    CMAKE_FLAGS=""
  fi

  cmake -B "$LLAMA_CPP_DIR/build" "$LLAMA_CPP_DIR" \
    -DCMAKE_BUILD_TYPE=Release $CMAKE_FLAGS \
    2>&1 | tail -5

  cmake --build "$LLAMA_CPP_DIR/build" --config Release \
    -j"$(nproc)" 2>&1 | tail -15

  LLAMA_QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"
  CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
fi

echo "✓  llama-quantize : $LLAMA_QUANTIZE"
echo "✓  convert script : $CONVERT_SCRIPT"

# ─── Step 3: Convert to GGUF F16 ─────────────────────────────────────────────
if [ "$SKIP_CONVERT" -eq 0 ]; then
  if [ -f "$GGUF_F16" ]; then
    echo ""
    echo "✓  GGUF F16 exists — skipping conversion"
  else
    echo ""
    echo "▶  [3/4] Converting merged model to GGUF F16..."
    source regllm.env/bin/activate
    python "$CONVERT_SCRIPT" \
      "$MERGED_DIR" \
      --outtype f16 \
      --outfile "$GGUF_F16"
    echo "✓  GGUF F16: $GGUF_F16"
  fi
else
  echo "⏭  Skipping convert (--skip-convert)"
fi

# ─── Step 4: Quantize to Q4_K_M ──────────────────────────────────────────────
if [ "$SKIP_QUANTIZE" -eq 0 ]; then
  if [ -f "$GGUF_Q4" ]; then
    echo ""
    echo "✓  Q4_K_M already exists — skipping quantization"
  else
    echo ""
    echo "▶  [4/4] Quantizing to Q4_K_M (~2 min)..."
    "$LLAMA_QUANTIZE" "$GGUF_F16" "$GGUF_Q4" Q4_K_M
    echo "✓  Q4_K_M: $GGUF_Q4"
    echo "   Removing F16 intermediate (~14 GB freed)..."
    rm -f "$GGUF_F16"
  fi
fi

# ─── Write Modelfile ──────────────────────────────────────────────────────────
echo ""
echo "▶  Writing Modelfile..."

cat > Modelfile <<EOF
FROM $GGUF_Q4

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

SYSTEM """Eres un experto asistente en regulación bancaria y riesgo de crédito.
Responde siempre en español. Sé preciso, cita las fuentes proporcionadas y nunca inventes
artículos o normativas. Si no sabes la respuesta, dilo claramente.
Cuando cites regulación, usa el formato: [FUENTE: nombre_del_documento, fragmento]."""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop <|im_end|>
PARAMETER num_ctx 4096
EOF

echo "✓  Modelfile written"

# ─── Create Ollama model ──────────────────────────────────────────────────────
echo ""
echo "▶  Creating Ollama model 'regllm'..."

# Start ollama serve if not already running
if ! curl -s http://localhost:11434 &>/dev/null; then
  echo "   Starting ollama serve (background)..."
  ollama serve >/tmp/ollama_build.log 2>&1 &
  OLLAMA_SERVE_PID=$!
  # Wait for server to be ready
  for i in $(seq 1 15); do
    if curl -s http://localhost:11434 &>/dev/null; then
      break
    fi
    sleep 1
  done
  # Cleanup this background serve when script exits
  trap "kill $OLLAMA_SERVE_PID 2>/dev/null || true" EXIT
fi

ollama create regllm -f Modelfile

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Build complete!"
echo ""
echo "  Quick test : ollama run regllm"
echo "  Full UI    : ./launch_ollama.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
