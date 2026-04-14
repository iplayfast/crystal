#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace crystal {

/// Ternary weight value: exactly {-1, 0, +1}
struct TernaryWeight {
    int8_t value{0};

    constexpr TernaryWeight() = default;
    constexpr explicit TernaryWeight(int8_t v) : value(v < 0 ? int8_t(-1) : (v > 0 ? int8_t(1) : int8_t(0))) {}

    constexpr bool operator==(const TernaryWeight&) const = default;
    constexpr auto operator<=>(const TernaryWeight&) const = default;

    /// Convert to arithmetic for accumulation
    constexpr explicit operator float() const { return static_cast<float>(value); }
    constexpr explicit operator double() const { return static_cast<double>(value); }
    constexpr explicit operator int() const { return static_cast<int>(value); }
};

/// Pack 128 ternary weights into 32 bytes (2 bits each) + FP16 scale
/// Encoding: 00 = 0, 01 = +1, 11 = -1 (10 unused)
struct TernaryGroup {
    static constexpr size_t NUM_WEIGHTS = 128;
    static constexpr size_t PACKED_BYTES = 32; // 128 * 2 bits = 256 bits = 32 bytes

    std::array<uint8_t, PACKED_BYTES> packed{};
    uint16_t scale_fp16{0}; // FP16 scale factor

    /// Pack 128 ternary weights and a scale into this group
    void pack(const TernaryWeight* weights, float scale) {
        packed.fill(0);
        for (size_t i = 0; i < NUM_WEIGHTS; ++i) {
            uint8_t code = 0;
            if (weights[i].value == 1) code = 0b01;
            else if (weights[i].value == -1) code = 0b11;
            // 0 stays 0b00

            size_t byte_idx = (i * 2) / 8;
            size_t bit_idx = (i * 2) % 8;
            packed[byte_idx] |= static_cast<uint8_t>(code << bit_idx);
        }
        scale_fp16 = float_to_fp16(scale);
    }

    /// Unpack 128 ternary weights and the scale from this group
    void unpack(TernaryWeight* weights, float& scale) const {
        for (size_t i = 0; i < NUM_WEIGHTS; ++i) {
            size_t byte_idx = (i * 2) / 8;
            size_t bit_idx = (i * 2) % 8;
            uint8_t code = (packed[byte_idx] >> bit_idx) & 0b11;

            if (code == 0b01) weights[i].value = 1;
            else if (code == 0b11) weights[i].value = -1;
            else weights[i].value = 0;
        }
        scale = fp16_to_float(scale_fp16);
    }

    /// Total size in bytes: 32 packed + 2 scale = 34
    static constexpr size_t total_bytes() { return PACKED_BYTES + sizeof(uint16_t); }

private:
    static uint16_t float_to_fp16(float f) {
        // IEEE 754 float to half conversion
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));

        uint32_t sign = (bits >> 31) & 1;
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
        uint32_t mant = bits & 0x7FFFFF;

        uint16_t h_sign = static_cast<uint16_t>(sign << 15);

        if (exp > 15) {
            return static_cast<uint16_t>(h_sign | 0x7C00); // inf
        }
        if (exp < -14) {
            return h_sign; // zero/denorm → zero for simplicity
        }

        uint16_t h_exp = static_cast<uint16_t>((exp + 15) << 10);
        uint16_t h_mant = static_cast<uint16_t>(mant >> 13);
        return static_cast<uint16_t>(h_sign | h_exp | h_mant);
    }

    static float fp16_to_float(uint16_t h) {
        uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) {
                float f;
                std::memcpy(&f, &sign, sizeof(f));
                return f;
            }
            // Denormalized
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
        } else if (exp == 31) {
            uint32_t bits = sign | 0x7F800000 | (mant << 13);
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }

        uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

/// WeightTraits: defines accumulator, shadow, and quantization properties per weight type
template <typename T>
struct WeightTraits;

template <>
struct WeightTraits<double> {
    using weight_type = double;
    using accumulator_type = double;
    using shadow_type = double;
    static constexpr bool is_quantized = false;
};

template <>
struct WeightTraits<float> {
    using weight_type = float;
    using accumulator_type = float;
    using shadow_type = float;
    static constexpr bool is_quantized = false;
};

template <>
struct WeightTraits<int> {
    using weight_type = int;
    using accumulator_type = int64_t;
    using shadow_type = int;
    static constexpr bool is_quantized = false;
};

template <>
struct WeightTraits<char> {
    using weight_type = char;
    using accumulator_type = int32_t;
    using shadow_type = char;
    static constexpr bool is_quantized = false;
};

template <>
struct WeightTraits<TernaryWeight> {
    using weight_type = TernaryWeight;
    using accumulator_type = float;
    using shadow_type = float;
    static constexpr bool is_quantized = true;
};

} // namespace crystal
