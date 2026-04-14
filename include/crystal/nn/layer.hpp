#pragma once

#include "crystal/core/concepts.hpp"
#include "crystal/core/types.hpp"
#include "crystal/core/random.hpp"

#include <vector>

namespace crystal {

/// A single layer in a neural network
template <NetworkWeight T>
class Layer {
public:
    using Traits = WeightTraits<T>;
    using accumulator_type = typename Traits::accumulator_type;
    using shadow_type = typename Traits::shadow_type;

    Layer() = default;
    Layer(size_t input_size, size_t output_size);

    /// Initialize weights (Xavier for float/double, uniform for int/char, zero shadow for ternary)
    void initialize();

    [[nodiscard]] size_t input_size() const { return input_size_; }
    [[nodiscard]] size_t output_size() const { return output_size_; }

    /// Weight matrix: [output_size x input_size] stored row-major
    std::vector<T> weights;
    /// Bias vector: [output_size]
    std::vector<T> biases;

    /// Shadow weights for quantized types (float precision during training)
    std::vector<shadow_type> shadow_weights;
    std::vector<shadow_type> shadow_biases;

    /// Per-layer quantization scale (for ternary)
    float quant_scale{0.0f};

    /// Output activations (set during forward pass)
    std::vector<accumulator_type> activations;
    /// Pre-activation values (set during forward pass)
    std::vector<accumulator_type> pre_activations;
    /// Error deltas (set during backward pass)
    std::vector<accumulator_type> deltas;

    /// Previous weight updates (for momentum)
    std::vector<accumulator_type> prev_weight_updates;
    std::vector<accumulator_type> prev_bias_updates;

private:
    size_t input_size_{0};
    size_t output_size_{0};
};

} // namespace crystal
