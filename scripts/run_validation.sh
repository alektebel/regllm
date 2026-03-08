#!/bin/bash
# Quick validation runner for RegLLM

set -e

echo "🚀 RegLLM Model Validation Runner"
echo "=================================="
echo ""

# Check if ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Error: Ollama is not running!"
    echo "   Start it with: ollama serve"
    exit 1
fi

echo "✓ Ollama is running"
echo ""

# Default values
TEST_FILE="${1:-data/test_ground_truth.json}"
MAX_QUESTIONS="${2:-50}"
OUTPUT_DIR="validation/$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Test file: $TEST_FILE"
echo "  Max questions: $MAX_QUESTIONS"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    echo ""
    echo "Usage: $0 [test_file] [max_questions]"
    echo "Example: $0 data/test_ground_truth.json 50"
    exit 1
fi

echo "🔄 Running comprehensive validation..."
echo ""

python scripts/validate_model.py \
    --test-file "$TEST_FILE" \
    --max-questions "$MAX_QUESTIONS" \
    --output "$OUTPUT_DIR/results.jsonl"

echo ""
echo "✅ Validation complete!"
echo ""
echo "📊 Results saved to:"
echo "   - $OUTPUT_DIR/results.jsonl"
echo "   - $OUTPUT_DIR/validation_report.md"
echo ""
echo "📖 View the report:"
echo "   cat $OUTPUT_DIR/validation_report.md"
echo "   or"
echo "   code $OUTPUT_DIR/validation_report.md"
echo ""
