#include "crystal/nn/blob_network.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace crystal {

template <NetworkWeight T>
BlobNetwork<T>::BlobNetwork(size_t input_size, size_t output_size, size_t complexity)
    : input_size_(input_size), output_size_(output_size) {
    size_t blob_count = (input_size + output_size) * complexity;
    blobs_.resize(blob_count);
    temp_blobs_.resize(blob_count);
    input_indexes_.resize(input_size);
    output_indexes_.resize(output_size);
    temp_input_indexes_.resize(input_size);
    randomize();
}

template <NetworkWeight T>
void BlobNetwork<T>::randomize() {
    size_t n = blobs_.size();
    for (size_t i = 0; i < n; ++i) {
        auto& b = blobs_[i];
        b.count = acc_t{};
        b.trigger_point = static_cast<acc_t>(Random::uniform(-1.0, 1.0));
        b.weight = static_cast<acc_t>(Random::uniform(-1.0, 1.0));
        b.value = static_cast<acc_t>(Random::uniform(-1.0, 1.0));
        b.index = static_cast<size_t>(Random::uniform_int(size_t{0}, n - 1));
        b.used = false;
    }
    for (size_t i = 0; i < input_size_; ++i) {
        input_indexes_[i] = static_cast<size_t>(Random::uniform_int(size_t{0}, n - 1));
    }
    for (size_t i = 0; i < output_size_; ++i) {
        output_indexes_[i] = static_cast<size_t>(Random::uniform_int(size_t{0}, n - 1));
    }
}

template <NetworkWeight T>
void BlobNetwork<T>::trace_inputs(const acc_t* input) {
    // Reset used flags
    for (auto& b : blobs_) {
        b.used = false;
        b.count = acc_t{};
    }

    // Feed inputs into their assigned nodes
    for (size_t i = 0; i < input_size_; ++i) {
        size_t idx = input_indexes_[i];
        blobs_[idx].count += input[i];
    }

    // Propagate through the graph (bounded to prevent infinite loops)
    bool changed = true;
    int max_steps = static_cast<int>(blobs_.size());
    while (changed && max_steps-- > 0) {
        changed = false;
        for (auto& b : blobs_) {
            if (!b.used && static_cast<double>(b.count) >= static_cast<double>(b.trigger_point)) {
                b.used = true;
                b.value = static_cast<acc_t>(
                    static_cast<double>(b.weight) * static_cast<double>(b.count)
                );
                // Forward to destination
                auto& dest = blobs_[b.index];
                dest.count += b.value;
                changed = true;
            }
        }
    }
}

template <NetworkWeight T>
void BlobNetwork<T>::compute_output(acc_t* output) {
    for (size_t i = 0; i < output_size_; ++i) {
        output[i] = blobs_[output_indexes_[i]].value;
    }
}

template <NetworkWeight T>
double BlobNetwork<T>::compute_error(const acc_t* output, const acc_t* target) {
    double error = 0.0;
    for (size_t i = 0; i < output_size_; ++i) {
        double diff = static_cast<double>(output[i]) - static_cast<double>(target[i]);
        error += diff * diff;
    }
    return error;
}

template <NetworkWeight T>
void BlobNetwork<T>::forward(const acc_t* input, acc_t* output) {
    trace_inputs(input);
    compute_output(output);
}

template <NetworkWeight T>
void BlobNetwork<T>::mutate() {
    temp_blobs_ = blobs_;
    temp_input_indexes_ = input_indexes_;

    size_t n = blobs_.size();

    // Mutate a random subset of blobs
    size_t mutations = std::max(size_t{1}, n / 20);
    for (size_t m = 0; m < mutations; ++m) {
        size_t idx = static_cast<size_t>(Random::uniform_int(size_t{0}, n - 1));
        auto& b = temp_blobs_[idx];

        int what = Random::uniform_int(0, 4);
        switch (what) {
            case 0: b.trigger_point = static_cast<acc_t>(Random::uniform(-1.0, 1.0)); break;
            case 1: b.weight = static_cast<acc_t>(Random::uniform(-1.0, 1.0)); break;
            case 2: b.value = static_cast<acc_t>(Random::uniform(-1.0, 1.0)); break;
            case 3: b.index = static_cast<size_t>(Random::uniform_int(size_t{0}, n - 1)); break;
            case 4:
                if (!input_indexes_.empty()) {
                    size_t ii = static_cast<size_t>(Random::uniform_int(size_t{0}, input_size_ - 1));
                    temp_input_indexes_[ii] = static_cast<size_t>(Random::uniform_int(size_t{0}, n - 1));
                }
                break;
        }
    }
}

template <NetworkWeight T>
typename BlobNetwork<T>::TrainingResult
BlobNetwork<T>::train(const acc_t* inputs, const acc_t* targets,
                     size_t num_samples, int max_iterations,
                     double target_error) {
    TrainingResult result;
    double best_error = std::numeric_limits<double>::max();
    std::vector<acc_t> output(output_size_);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Evaluate current network
        double total_error = 0.0;
        for (size_t s = 0; s < num_samples; ++s) {
            forward(inputs + s * input_size_, output.data());
            total_error += compute_error(output.data(), targets + s * output_size_);
        }
        error_level_ = total_error;

        if (total_error < best_error) {
            best_error = total_error;
        }

        if (progress_callback_ && (iter % 100 == 0)) {
            progress_callback_(iter, total_error);
        }

        if (total_error <= target_error) {
            result.iterations = iter + 1;
            result.final_error = total_error;
            result.converged = true;
            return result;
        }

        // Try a mutation
        auto saved_blobs = blobs_;
        auto saved_inputs = input_indexes_;
        mutate();
        blobs_ = temp_blobs_;
        input_indexes_ = temp_input_indexes_;

        // Evaluate mutated network
        double mutated_error = 0.0;
        for (size_t s = 0; s < num_samples; ++s) {
            forward(inputs + s * input_size_, output.data());
            mutated_error += compute_error(output.data(), targets + s * output_size_);
        }

        // Simulated annealing acceptance: always accept improvements,
        // accept worse solutions with probability exp(-delta/temperature)
        double temperature = static_cast<double>(max_iterations - iter) / static_cast<double>(max_iterations);
        temperature = std::max(temperature, 0.001); // floor to avoid div-by-zero

        if (mutated_error < total_error) {
            // Always accept improvements
        } else {
            double delta = mutated_error - total_error;
            double accept_prob = std::exp(-delta / (temperature * best_error + 1e-10));
            if (Random::uniform(0.0, 1.0) > accept_prob) {
                // Reject: revert mutation
                blobs_ = saved_blobs;
                input_indexes_ = saved_inputs;
            }
        }
    }

    result.iterations = max_iterations;
    result.final_error = error_level_;
    result.converged = false;
    return result;
}

template <NetworkWeight T>
nlohmann::json BlobNetwork<T>::to_json() const {
    nlohmann::json j;
    j["input_size"] = input_size_;
    j["output_size"] = output_size_;
    j["blob_count"] = blobs_.size();
    j["input_indexes"] = input_indexes_;
    j["output_indexes"] = output_indexes_;

    j["blobs"] = nlohmann::json::array();
    for (auto& b : blobs_) {
        j["blobs"].push_back({
            {"trigger_point", static_cast<double>(b.trigger_point)},
            {"weight", static_cast<double>(b.weight)},
            {"value", static_cast<double>(b.value)},
            {"index", b.index}
        });
    }
    return j;
}

template <NetworkWeight T>
BlobNetwork<T> BlobNetwork<T>::from_json(const nlohmann::json& j) {
    size_t input_size = j.at("input_size").get<size_t>();
    size_t output_size = j.at("output_size").get<size_t>();
    size_t blob_count = j.at("blob_count").get<size_t>();
    size_t complexity = blob_count / (input_size + output_size);

    BlobNetwork net(input_size, output_size, complexity);
    net.input_indexes_ = j.at("input_indexes").get<std::vector<size_t>>();
    net.output_indexes_ = j.at("output_indexes").get<std::vector<size_t>>();

    auto& blobs_json = j.at("blobs");
    for (size_t i = 0; i < blob_count && i < blobs_json.size(); ++i) {
        auto& bj = blobs_json[i];
        net.blobs_[i].trigger_point = static_cast<acc_t>(bj.at("trigger_point").get<double>());
        net.blobs_[i].weight = static_cast<acc_t>(bj.at("weight").get<double>());
        net.blobs_[i].value = static_cast<acc_t>(bj.at("value").get<double>());
        net.blobs_[i].index = bj.at("index").get<size_t>();
    }

    return net;
}

// Explicit instantiations
template class BlobNetwork<double>;
template class BlobNetwork<float>;

} // namespace crystal
