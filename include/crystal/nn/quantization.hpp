#pragma once

#include "crystal/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace crystal {

/// Absmean quantizer: scale = mean(|w|), then round(w/scale) clamped to {-1, 0, +1}
struct AbsmeanQuantizer {
    /// Compute scale as mean of absolute values
    static float compute_scale(std::span<const float> weights);

    /// Quantize shadow weights to ternary using absmean
    static void quantize(std::span<const float> shadow_weights,
                        std::span<TernaryWeight> out_weights,
                        float& out_scale);
};

/// Absmax activation quantizer: scale = max(|act|), q = round(act * 127 / scale)
struct AbsmaxActivationQuantizer {
    /// Quantize activations to int8
    static float quantize(std::span<const float> activations,
                         std::span<int8_t> out_quantized);

    /// Dequantize int8 activations back to float
    static void dequantize(std::span<const int8_t> quantized,
                          float scale,
                          std::span<float> out_activations);
};

/// Straight-Through Estimator: gradient passes through quantizer unchanged
struct STE {
    /// Apply gradient to shadow weights (gradient flows through as-is)
    static void apply(std::span<const float> gradients,
                     std::span<float> shadow_weights,
                     float learning_rate);
};

} // namespace crystal
