#!/bin/bash
# Launch the RegLLM chat interface

echo "🏦 Launching Spanish Banking Regulation Chatbot..."
echo ""

# Find the most recent trained model
MODEL_PATH="models/finetuned/run_20260116_212356/final_model"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    echo "Available models:"
    ls -1 models/finetuned/
    exit 1
fi

echo "✓ Using model: $MODEL_PATH"
echo ""
echo "Starting web interface on http://localhost:7860"
echo "Press Ctrl+C to stop"
echo ""

python src/ui/chat_interface.py \
    --model-path "$MODEL_PATH" \
    --interface web \
    --port 7860
