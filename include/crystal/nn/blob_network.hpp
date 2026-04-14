#pragma once

#include "crystal/core/concepts.hpp"
#include "crystal/core/types.hpp"
#include "crystal/core/random.hpp"

#include <cstdint>
#include <functional>
#include <vector>
#include <nlohmann/json.hpp>

namespace crystal {

/// A "blob" node in the unstructured neural network
template <NetworkWeight T>
struct Blob {
    using acc_t = typename WeightTraits<T>::accumulator_type;

    acc_t count{};         // accumulated weighted input
    acc_t trigger_point{}; // activation threshold
    acc_t weight{};        // output weight
    acc_t value{};         // output value
    size_t index{0};       // destination node index
    bool used{false};      // cycle detection flag
};

/// BlobNetwork: simulated-annealing graph neural network
///
/// No layers — just a flat "blob" of interconnected nodes.
/// Training uses simulated annealing: randomly mutate, keep if better.
template <NetworkWeight T>
class BlobNetwork {
public:
    using acc_t = typename WeightTraits<T>::accumulator_type;

    BlobNetwork() = default;
    BlobNetwork(size_t input_size, size_t output_size, size_t complexity = 100);

    /// Randomize all blob weights and connections
    void randomize();

    /// Run a forward pass with the given inputs
    void forward(const acc_t* input, acc_t* output);

    /// Get the current error level
    [[nodiscard]] double error_level() const { return error_level_; }

    /// Train on a dataset using simulated annealing
    struct TrainingResult {
        int iterations{0};
        double final_error{0.0};
        bool converged{false};
    };

    TrainingResult train(const acc_t* inputs, const acc_t* targets,
                        size_t num_samples, int max_iterations,
                        double target_error = 0.01);

    /// JSON serialization
    [[nodiscard]] nlohmann::json to_json() const;
    static BlobNetwork from_json(const nlohmann::json& j);

    /// Progress callback
    using ProgressCallback = std::function<void(int iteration, double error)>;
    void set_progress_callback(ProgressCallback cb) { progress_callback_ = std::move(cb); }

    [[nodiscard]] size_t input_size() const { return input_size_; }
    [[nodiscard]] size_t output_size() const { return output_size_; }
    [[nodiscard]] size_t blob_size() const { return blobs_.size(); }

private:
    std::vector<Blob<T>> blobs_;
    std::vector<Blob<T>> temp_blobs_;  // for annealing
    std::vector<size_t> input_indexes_;
    std::vector<size_t> output_indexes_;
    std::vector<size_t> temp_input_indexes_;
    size_t input_size_{0};
    size_t output_size_{0};
    double error_level_{0.0};
    ProgressCallback progress_callback_;

    /// Trace inputs through the blob graph
    void trace_inputs(const acc_t* input);

    /// Compute output from blob state
    void compute_output(acc_t* output);

    /// Compute error against target
    double compute_error(const acc_t* output, const acc_t* target);

    /// Mutate the temp blob randomly
    void mutate();
};

// Extern template declarations
extern template class BlobNetwork<double>;
extern template class BlobNetwork<float>;

} // namespace crystal
