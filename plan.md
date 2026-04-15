# Crystal — Modern C++20 Computational Intelligence Library

## Overview

Crystal is a cross-platform C++20 library for fuzzy logic, neural networks, quantization, and GGUF model I/O. It includes CLI and Qt6 GUI applications. Originally ported from a ~114K-line Borland-era C++ codebase (cyrlib).

**Build:** `cmake --preset debug && cmake --build build/debug`
**Test:** `ctest --test-dir build/debug --output-on-failure`
**Tests:** 10/10 passing

---

## Project Structure

```
crystal/
├── include/crystal/
│   ├── core/           # TernaryWeight, TernaryGroup, concepts, random
│   ├── fuzzy/          # FuzzySet<T>, fuzzy operators (&, |, ^, ~)
│   ├── nn/             # BackpropNetwork<T>, BlobNetwork, layers, activation
│   ├── io/             # GGUF binary format reader/writer
│   ├── math/           # HugeInt (stub)
│   ├── quantize/       # LLM ternary quantization pipeline
│   └── patterns/       # Factory, Observer, Strategy
├── src/                # Implementations for all modules above
├── apps/
│   ├── crystal_quantize/       # CLI: ternary quantization tool
│   ├── crystal_quantize_gui/   # Qt6 GUI: quantization with Ollama picker
│   ├── fuzzy_builder/          # Qt6 GUI: visual fuzzy logic editor
│   └── class_builder/          # Qt6 GUI: C++ class code generator
├── tests/              # 10 test executables (assert-based)
├── benchmarks/         # Empty — not yet implemented
├── cmake/              # CompilerWarnings.cmake
└── .github/workflows/  # CI (Ubuntu/macOS/Windows × g++/clang++/cl)
```

---

## Module Status

### Core Library — Complete

| Module | Status | Key Types |
|--------|--------|-----------|
| core | Done | `TernaryWeight`, `TernaryGroup`, `WeightTraits<T>`, C++20 concepts |
| fuzzy | Done | `FuzzySet<T>` with evaluate, and/or/xor/complement, optimize, increase_samples, equality, JSON I/O |
| nn | Done | `BackpropNetwork<T>` (double/float/int/char/ternary), `BlobNetwork` (simulated annealing) |
| io | Done | GGUF reader/writer (F32, F16, Q8_0, TERNARY_B158) |
| math | Stub | `HugeInt` uses `unsigned long long`, not Boost.Multiprecision |
| patterns | Done | Factory, Observer, Strategy |
| quantize | Done | Full pipeline: model_reader → calibration → ternary_converter → gguf_writer |

### Quantization Pipeline — Complete

| Component | Description |
|-----------|-------------|
| model_reader | Reads GGUF tensors via `gguf_init_from_file()`, dequantizes to F32 |
| calibration | Entropy-based importance via llama.cpp inference; weight-magnitude fallback |
| ternary_converter | Absmean quantization to {-1, 0, +1} in TernaryGroup blocks |
| gguf_writer | Writes quantized GGUF with metadata via llama.cpp API |
| ensemble | Weight averaging across multiple models |
| pipeline | Orchestrates all steps with options and progress reporting |

llama.cpp linked via FetchContent (tag b8784). API note: uses `llama_memory_clear()` not `llama_kv_cache_clear()`.

**Compression:** ~6.6% of original size (F32 → ternary, ~15:1 ratio)

### Applications — Complete

| App | Type | Description |
|-----|------|-------------|
| crystal-quantize | CLI | `crystal-quantize output.gguf dataset.txt model1.gguf [model2.gguf ...]` |
| crystal_quantize_gui | Qt6 GUI | Ollama model picker, ensemble support, progress bar, log output |
| fuzzy_builder | Qt6 GUI | Visual logic block editor (In/Out/Or/Xor/And/Fuzzy), connection drawing, membership chart, simulation, save/load in original format, C header export |
| class_builder | Qt6 GUI | C++ class generator with crystal base types, member variables, getters/setters, JSON save/load |

### Tests — 10/10 Passing

| Test | What it covers |
|------|---------------|
| test_fuzzy_set | Sorted insert, interpolation, AND, JSON, optimize, increase_samples, equality, indexed access |
| test_backprop_double | BackpropNetwork<double> XOR training |
| test_backprop_float | BackpropNetwork<float> (8+ hidden, 10K+ epochs for XOR) |
| test_backprop_int | BackpropNetwork<int> integer arithmetic |
| test_backprop_ternary | BackpropNetwork<TernaryWeight> shadow weights |
| test_quantization | TernaryGroup packing/unpacking, FP16 conversion |
| test_gguf_io | GGUF read/write round-trip |
| test_blob_network | BlobNetwork simulated annealing convergence |
| test_reflection | Type introspection utilities |
| test_quantize_pipeline | End-to-end model → ternary → GGUF |

---

## Known Issues

- Progress bar in quantize GUI stays at 30% during importance computation (no incremental callback)
- CI matrix needs exclude rules for invalid OS/compiler combos (e.g., g++ on Windows)
- HugeInt is a stub — not usable for real big-integer math

---

## Remaining Work

### Phase 10: Benchmarks & Polish

- [ ] Google Benchmark integration for core operations (fuzzy eval, backprop forward/backward, quantization)
- [ ] README with API examples
- [ ] API documentation
- [ ] Fix CI matrix exclusions
- [ ] HugeInt: decide on Boost.Multiprecision or drop

### Potential Improvements

- [ ] Quantize GUI: incremental progress during calibration step
- [ ] FuzzyBuilder: undo/redo, copy/paste blocks, zoom-to-fit
- [ ] ClassBuilder: template parameter support, inheritance chains
- [ ] Calibration: hook into llama.cpp imatrix for true per-weight importance (currently uses magnitude proxy)

---

## Usage

### CLI Quantization
```bash
cmake --preset debug && cmake --build build/debug

./build/debug/apps/crystal_quantize/crystal-quantize \
  --verbose \
  output.gguf \
  /dev/null \
  ~/.ollama/models/blobs/sha256:YOUR_MODEL_HASH
```

### GUI Apps
```bash
./build/debug/apps/crystal_quantize_gui/crystal_quantize_gui
./build/debug/apps/fuzzy_builder/fuzzy_builder
./build/debug/apps/class_builder/class_builder
```

### Ollama Model Discovery
```bash
ls -lS ~/.ollama/models/blobs/  # Large files (>100MB) are GGUF models
```

### Testing Quantized Models
```bash
./test_custom_gguf.sh /path/to/quantized.gguf "Your prompt"
```
