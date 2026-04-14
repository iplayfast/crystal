#pragma once

#include "crystal/core/concepts.hpp"
#include "crystal/core/types.hpp"
#include "crystal/core/random.hpp"
#include "crystal/nn/activation.hpp"
#include "crystal/nn/quantization.hpp"
#include "crystal/nn/layer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <span>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace crystal {

/// Training configuration
struct TrainingConfig {
    double learning_rate = 0.25;
    double momentum = 0.9;
    double gain = 1.0;
    int max_epochs = 5000;
    double error_threshold = 0.01;
    bool early_stopping = true;
    double early_stopping_factor = 1.2; // stop if error > factor * best_error
};

/// Training result
struct TrainingResult {
    int epochs_run{0};
    double final_error{0.0};
    bool converged{false};
};

/// Templated backpropagation neural network
///
/// Supports double, float, int, char, and TernaryWeight.
/// For quantized types (TernaryWeight), shadow weights are maintained in float
/// precision during training, and weights are quantized on each forward pass.
template <NetworkWeight T>
class BackpropNetwork {
public:
    using Traits = WeightTraits<T>;
    using accumulator_type = typename Traits::accumulator_type;
    using shadow_type = typename Traits::shadow_type;

    BackpropNetwork() = default;

    /// Add a layer with the given number of neurons
    void add_layer(size_t size);

    /// Number of layers
    [[nodiscard]] size_t num_layers() const { return layers_.size(); }

    /// Get layer sizes
    [[nodiscard]] std::vector<size_t> layer_sizes() const;

    /// Initialize weights randomly
    void randomize_weights();

    /// Forward pass: compute outputs from inputs
    void forward(std::span<const accumulator_type> input);

    /// Get output of the network (after forward pass)
    [[nodiscard]] std::span<const accumulator_type> output() const;

    /// Compute output error against target
    [[nodiscard]] double compute_error(std::span<const accumulator_type> target);

    /// Backward pass: propagate error and adjust weights
    void backward(std::span<const accumulator_type> target, const TrainingConfig& config);

    /// Run a single sample: forward + error + backward (if training)
    double simulate(std::span<const accumulator_type> input,
                   std::span<const accumulator_type> target,
                   bool training,
                   const TrainingConfig& config);

    /// Train on a dataset for a number of epochs
    TrainingResult train(std::span<const accumulator_type> inputs,
                        std::span<const accumulator_type> targets,
                        size_t num_samples,
                        const TrainingConfig& config);

    /// Train with early stopping (like the original STTrainNet)
    TrainingResult train_early_stopping(std::span<const accumulator_type> inputs,
                                       std::span<const accumulator_type> targets,
                                       size_t num_samples,
                                       const TrainingConfig& config);

    /// Save/restore best weights
    void save_weights();
    void restore_weights();

    /// JSON serialization
    [[nodiscard]] nlohmann::json to_json() const;
    static BackpropNetwork from_json(const nlohmann::json& j);

    /// Training progress callback
    using ProgressCallback = std::function<void(int epoch, double error)>;
    void set_progress_callback(ProgressCallback cb) { progress_callback_ = std::move(cb); }

    /// Access layers (for GGUF export, etc.)
    [[nodiscard]] const std::vector<Layer<T>>& layers() const { return layers_; }
    [[nodiscard]] std::vector<Layer<T>>& layers() { return layers_; }

private:
    std::vector<Layer<T>> layers_;
    std::vector<Layer<T>> saved_layers_;  // for save/restore
    ProgressCallback progress_callback_;

    /// Quantize shadow weights to ternary (only for TernaryWeight)
    void quantize_layer(Layer<T>& layer);
};

// Extern template declarations
extern template class BackpropNetwork<double>;
extern template class BackpropNetwork<float>;
extern template class BackpropNetwork<int>;
extern template class BackpropNetwork<char>;
extern template class BackpropNetwork<TernaryWeight>;

} // namespace crystal
