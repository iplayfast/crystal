#include "crystal/nn/quantization.hpp"

#include <cmath>

namespace crystal {

float AbsmeanQuantizer::compute_scale(std::span<const float> weights) {
    if (weights.empty()) return 0.0f;
    float sum = 0.0f;
    for (float w : weights) {
        sum += std::abs(w);
    }
    return sum / static_cast<float>(weights.size());
}

void AbsmeanQuantizer::quantize(std::span<const float> shadow_weights,
                                std::span<TernaryWeight> out_weights,
                                float& out_scale) {
    out_scale = compute_scale(shadow_weights);
    if (out_scale == 0.0f) {
        for (size_t i = 0; i < shadow_weights.size(); ++i) {
            out_weights[i].value = 0;
        }
        return;
    }

    for (size_t i = 0; i < shadow_weights.size(); ++i) {
        float normalized = shadow_weights[i] / out_scale;
        int rounded = static_cast<int>(std::round(normalized));
        rounded = std::clamp(rounded, -1, 1);
        out_weights[i].value = static_cast<int8_t>(rounded);
    }
}

float AbsmaxActivationQuantizer::quantize(std::span<const float> activations,
                                          std::span<int8_t> out_quantized) {
    if (activations.empty()) return 0.0f;

    float max_abs = 0.0f;
    for (float a : activations) {
        max_abs = std::max(max_abs, std::abs(a));
    }

    if (max_abs == 0.0f) {
        for (size_t i = 0; i < activations.size(); ++i) {
            out_quantized[i] = 0;
        }
        return 0.0f;
    }

    for (size_t i = 0; i < activations.size(); ++i) {
        float scaled = activations[i] * 127.0f / max_abs;
        int rounded = static_cast<int>(std::round(scaled));
        rounded = std::clamp(rounded, -127, 127);
        out_quantized[i] = static_cast<int8_t>(rounded);
    }

    return max_abs;
}

void AbsmaxActivationQuantizer::dequantize(std::span<const int8_t> quantized,
                                           float scale,
                                           std::span<float> out_activations) {
    for (size_t i = 0; i < quantized.size(); ++i) {
        out_activations[i] = static_cast<float>(quantized[i]) * scale / 127.0f;
    }
}

void STE::apply(std::span<const float> gradients,
               std::span<float> shadow_weights,
               float learning_rate) {
    for (size_t i = 0; i < gradients.size(); ++i) {
        shadow_weights[i] -= learning_rate * gradients[i];
    }
}

} // namespace crystal
