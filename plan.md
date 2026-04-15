# Crystal Quantize - Plan

## Goal

Build a GUI application for quantizing GGUF models using Crystal's ternary quantization, with support for:
1. Discovering Ollama models and showing human-readable names
2. Adding multiple models to an ensemble for quantization
3. Testing resulting GGUF files with llama.cpp

## Progress Log

### 2025-04-14

**Completed:**
- Fixed model discovery to read Ollama manifests and show proper names (e.g., "qwen3.5:0.8b" instead of hash)
- Fixed blob prefix detection for both `sha256:` and `sha256-` formats
- Added UI buttons: "Add to Ensemble" and "Add Custom GGUF to Ensemble"
- Added debug messages to pipeline.cpp and calibration.cpp
- Fixed duplicate model bug in GUI (filters duplicates before passing to pipeline)
- Fixed progress bar signal connection (now properly calls onQuantizeFinished slot)
- Cloned Bonsai-demo repo for testing at `/home/chris/ai/Bonsai-demo`
- Created test script `/home/chris/ai/crystal/test_custom_gguf.sh` for testing quantized models

**Known Issues:**
- Progress bar stays at 30% during importance computation step (Step 3) - no incremental progress updates
- No callback mechanism for progress updates during quantization

### 2025-04-13

**Completed:**
- Initial GUI with Ollama model discovery
- Basic quantization pipeline
- Ensemble support

## Relevant Files

- `apps/crystal_quantize_gui/main.cpp` - Main GUI code
- `src/quantize/pipeline.cpp` - Pipeline with debug output
- `src/quantize/calibration.cpp` - Importance computation
- `test_custom_gguf.sh` - Test script for quantized models

## Testing

Use the test script to verify quantized models work with llama.cpp:

```bash
./test_custom_gguf.sh /path/to/quantized.gguf "Your prompt"
```
