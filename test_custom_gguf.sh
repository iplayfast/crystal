#!/bin/bash
# Test a crystal-quantized GGUF model using Bonsai-demo's llama.cpp infrastructure.
# Defaults to interactive conversation mode.
#
# Usage:
#   ./test_custom_gguf.sh /path/to/quantized.gguf              # interactive chat
#   ./test_custom_gguf.sh /path/to/quantized.gguf -p "Hello"   # single prompt
#   ./test_custom_gguf.sh /path/to/quantized.gguf --server      # start OpenAI-compatible server
#
# Requires: Bonsai-demo setup at ~/ai/Bonsai-demo (run its setup.sh first)
set -e

BONSAI_DIR="${BONSAI_DIR:-$HOME/ai/Bonsai-demo}"

# ── Source Bonsai helpers ──
if [ ! -f "$BONSAI_DIR/scripts/common.sh" ]; then
    echo "[ERR] Bonsai-demo not found at $BONSAI_DIR"
    echo "  Clone it:  git clone https://github.com/PrismML-Eng/Bonsai-demo.git ~/ai/Bonsai-demo"
    echo "  Then run:  cd ~/ai/Bonsai-demo && ./setup.sh"
    echo "  Or set:    BONSAI_DIR=/path/to/Bonsai-demo"
    exit 1
fi
. "$BONSAI_DIR/scripts/common.sh"

# ── Parse arguments ──
MODEL="${1:-}"
SERVER_MODE=false

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <path-to-gguf> [llama-cli options...]"
    echo ""
    echo "Modes:"
    echo "  $0 model.gguf                    Interactive conversation (default)"
    echo "  $0 model.gguf -p 'Hello'         Single prompt"
    echo "  $0 model.gguf --server            Start OpenAI-compatible server on :8080"
    echo ""
    echo "Examples:"
    echo "  $0 /tmp/crystal-quantized.gguf"
    echo "  $0 /tmp/crystal-quantized.gguf -p 'What is 2+2?' -n 256"
    echo ""
    echo "Environment:"
    echo "  BONSAI_DIR   Path to Bonsai-demo (default: ~/ai/Bonsai-demo)"
    echo "  BONSAI_NGL   GPU layer count override (default: auto-detect)"
    exit 1
fi
shift

if [ ! -f "$MODEL" ]; then
    err "Model file not found: $MODEL"
    exit 1
fi

# Check for --server flag
for arg in "$@"; do
    if [ "$arg" = "--server" ]; then
        SERVER_MODE=true
        break
    fi
done

# ── Find binary ──
BIN_NAME="llama-cli"
$SERVER_MODE && BIN_NAME="llama-server"

BIN=""
for _d in bin/mac bin/cuda bin/rocm bin/hip bin/vulkan bin/cpu llama.cpp/build/bin llama.cpp/build-mac/bin llama.cpp/build-cuda/bin; do
    [ -f "$BONSAI_DIR/$_d/$BIN_NAME" ] && BIN="$BONSAI_DIR/$_d/$BIN_NAME" && break
done

if [ -z "$BIN" ]; then
    BIN=$(command -v "$BIN_NAME" 2>/dev/null || echo "")
fi

if [ -z "$BIN" ]; then
    err "$BIN_NAME not found. Run Bonsai-demo setup first:"
    echo "  cd $BONSAI_DIR && ./setup.sh"
    exit 1
fi

# ── Library path for shared libs ──
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── GPU detection ──
NGL=$(bonsai_llama_ngl)

# ── Model info ──
MODEL_SIZE=$(du -h "$MODEL" | cut -f1)
echo ""
echo "=== Crystal GGUF Test ==="
echo "  Model:   $MODEL ($MODEL_SIZE)"
echo "  Binary:  $BIN"
echo "  GPU layers: $NGL"

if $SERVER_MODE; then
    # ── Server mode ──
    # Strip --server from args passed to llama-server
    FILTERED_ARGS=()
    for arg in "$@"; do
        [ "$arg" != "--server" ] && FILTERED_ARGS+=("$arg")
    done

    HOST="0.0.0.0"
    PORT=8080

    if curl -s --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        warn "Server already running on port $PORT."
        echo "  Stop it: kill \$(lsof -ti TCP:$PORT)"
        exit 1
    fi

    echo "  Mode:    Server (OpenAI-compatible)"
    echo ""
    echo "  Chat UI:  http://localhost:$PORT"
    echo "  API:      http://localhost:$PORT/v1/chat/completions"
    echo "  Press Ctrl+C to stop."
    echo ""

    exec "$BIN" -m "$MODEL" --host "$HOST" --port "$PORT" -ngl "$NGL" -c "$CTX_SIZE_DEFAULT" \
        --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0 \
        --reasoning-budget 0 --reasoning-format none \
        --chat-template-kwargs '{"enable_thinking": false}' \
        "${FILTERED_ARGS[@]}"
else
    # ── CLI mode (interactive by default) ──
    HAS_PROMPT=false
    for arg in "$@"; do
        [ "$arg" = "-p" ] && HAS_PROMPT=true && break
    done

    if $HAS_PROMPT; then
        echo "  Mode:    Single prompt"
    else
        echo "  Mode:    Interactive conversation"
    fi
    echo ""

    "$BIN" -m "$MODEL" -ngl "$NGL" -c "$CTX_SIZE_DEFAULT" --log-disable \
        --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0 \
        --reasoning-budget 0 --reasoning-format none \
        --chat-template-kwargs '{"enable_thinking": false}' \
        "$@" 2>/dev/null \
    || {
        CTX_SIZE=$(get_context_size_fallback)
        warn "Auto-fit not supported, falling back to -c $CTX_SIZE"
        "$BIN" -m "$MODEL" -ngl "$NGL" -c "$CTX_SIZE" --log-disable \
            --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0 \
            --reasoning-budget 0 --reasoning-format none \
            --chat-template-kwargs '{"enable_thinking": false}' \
            "$@" 2>/dev/null
    }
fi
