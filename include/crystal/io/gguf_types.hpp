#pragma once

#include <cstdint>

namespace crystal {

/// GGUF quantization types (matching ggml spec)
enum class GGUFQuantType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    TERNARY_B158 = 1024, // Custom: BitNet b1.58 ternary
};

/// GGUF metadata value types
enum class GGUFMetaType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

/// Block sizes for quantization types
inline uint32_t gguf_block_size(GGUFQuantType type) {
    switch (type) {
        case GGUFQuantType::F32: return 1;
        case GGUFQuantType::F16: return 1;
        case GGUFQuantType::Q8_0: return 32;
        case GGUFQuantType::TERNARY_B158: return 128; // TernaryGroup::NUM_WEIGHTS
        default: return 1;
    }
}

/// Bytes per element/block for quantization types
inline uint32_t gguf_type_size(GGUFQuantType type) {
    switch (type) {
        case GGUFQuantType::F32: return 4;
        case GGUFQuantType::F16: return 2;
        case GGUFQuantType::Q8_0: return 34; // 32 bytes data + 2 bytes scale
        case GGUFQuantType::TERNARY_B158: return 34; // 32 bytes packed + 2 bytes FP16 scale
        default: return 0;
    }
}

constexpr uint32_t GGUF_MAGIC = 0x46475547; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;
constexpr uint32_t GGUF_ALIGNMENT = 32;

} // namespace crystal
