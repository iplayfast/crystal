#!/bin/bash
# Test custom GGUF model with llama.cpp
# Usage: ./test_custom_gguf.sh /path/to/model.gguf "Your prompt"

MODEL="${1:-}"
PROMPT="${2:-Hello, who are you?}"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <path-to-gguf> [prompt]"
    echo "  Example: $0 /tmp/myquantized.gguf 'What is 2+2?'"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found: $MODEL"
    exit 1
fi

# Find llama-cli binary
DEMO_DIR="/home/chris/ai/Bonsai-demo"
BIN=""
for _d in "$DEMO_DIR/llama.cpp/build/bin" "$DEMO_DIR/bin/cpu" "$DEMO_DIR/bin/cuda" "$DEMO_DIR/bin/mac"; do
    [ -f "$_d/llama-cli" ] && BIN="$_d/llama-cli" && break
done

if [ -z "$BIN" ]; then
    # Fall back to system llama-cli
    BIN=$(command -v llama-cli 2>/dev/null || command -v llama 2>/dev/null || echo "")
fi

if [ -z "$BIN" ]; then
    echo "Error: llama-cli not found. Build from llama.cpp or run Bonsai-demo setup."
    exit 1
fi

# Set library path
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Detect GPU
NGL=0
if command -v nvidia-smi >/dev/null 2>&1; then
    NGL=99
elif [ "$(uname -s)" = "Darwin" ]; then
    NGL=99
fi

echo "Model: $MODEL"
echo "Binary: $BIN"
echo "NGL: $NGL"
echo ""

"$BIN" -m "$MODEL" -ngl "$NGL" -c 4096 --log-disable \
    --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0 \
    -p "$PROMPT" -n 256 --no-display
