#include "crystal/nn/backprop.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace crystal {

// --- Layer implementation ---

template <NetworkWeight T>
Layer<T>::Layer(size_t input_size, size_t output_size)
    : input_size_(input_size), output_size_(output_size) {
    size_t num_weights = output_size * input_size;
    weights.resize(num_weights);
    biases.resize(output_size);
    activations.resize(output_size);
    pre_activations.resize(output_size);
    deltas.resize(output_size);

    prev_weight_updates.resize(num_weights, accumulator_type{});
    prev_bias_updates.resize(output_size, accumulator_type{});

    if constexpr (Traits::is_quantized) {
        shadow_weights.resize(num_weights);
        shadow_biases.resize(output_size);
    }
}

template <NetworkWeight T>
void Layer<T>::initialize() {
    if constexpr (Traits::is_quantized) {
        // Initialize shadow weights, quantize later
        for (size_t i = 0; i < shadow_weights.size(); ++i) {
            shadow_weights[i] = Random::xavier<float>(input_size_, output_size_);
        }
        for (size_t i = 0; i < shadow_biases.size(); ++i) {
            shadow_biases[i] = 0.0f;
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        using FP = T;
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = Random::uniform(static_cast<FP>(-1), static_cast<FP>(1));
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = FP{0};
        }
    } else {
        // Integer types
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<T>(Random::uniform_int(-100, 100));
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = T{0};
        }
    }
}

// --- BackpropNetwork implementation ---

template <NetworkWeight T>
void BackpropNetwork<T>::add_layer(size_t size) {
    size_t prev_size = layers_.empty() ? 0 : layers_.back().output_size();
    layers_.emplace_back(prev_size, size);
}

template <NetworkWeight T>
std::vector<size_t> BackpropNetwork<T>::layer_sizes() const {
    std::vector<size_t> sizes;
    sizes.reserve(layers_.size());
    for (auto& l : layers_) {
        sizes.push_back(l.output_size());
    }
    return sizes;
}

template <NetworkWeight T>
void BackpropNetwork<T>::randomize_weights() {
    for (auto& layer : layers_) {
        layer.initialize();
    }
}

template <NetworkWeight T>
void BackpropNetwork<T>::quantize_layer(Layer<T>& layer) {
    if constexpr (Traits::is_quantized) {
        AbsmeanQuantizer::quantize(
            std::span<const float>(layer.shadow_weights),
            std::span<TernaryWeight>(layer.weights),
            layer.quant_scale
        );
    }
}

template <NetworkWeight T>
void BackpropNetwork<T>::forward(std::span<const accumulator_type> input) {
    if (layers_.empty()) return;

    // Set input layer activations
    auto& input_layer = layers_[0];
    for (size_t i = 0; i < input_layer.activations.size() && i < input.size(); ++i) {
        input_layer.activations[i] = input[i];
    }

    // Propagate through hidden and output layers
    for (size_t l = 1; l < layers_.size(); ++l) {
        auto& prev = layers_[l - 1];
        auto& curr = layers_[l];

        // Quantize weights if ternary
        if constexpr (Traits::is_quantized) {
            quantize_layer(curr);
        }

        for (size_t i = 0; i < curr.output_size(); ++i) {
            accumulator_type sum{};

            if constexpr (Traits::is_quantized) {
                // Ternary: MAC is just add/subtract
                for (size_t j = 0; j < prev.output_size(); ++j) {
                    int8_t w = curr.weights[i * prev.output_size() + j].value;
                    if (w == 1) sum += prev.activations[j];
                    else if (w == -1) sum -= prev.activations[j];
                }
                sum *= curr.quant_scale;
            } else {
                for (size_t j = 0; j < prev.output_size(); ++j) {
                    if constexpr (std::is_floating_point_v<T>) {
                        sum += static_cast<accumulator_type>(curr.weights[i * prev.output_size() + j])
                             * prev.activations[j];
                    } else {
                        sum += static_cast<accumulator_type>(curr.weights[i * prev.output_size() + j])
                             * prev.activations[j];
                    }
                }
            }

            // Add bias
            if constexpr (Traits::is_quantized) {
                sum += static_cast<accumulator_type>(curr.shadow_biases[i]);
            } else {
                sum += static_cast<accumulator_type>(curr.biases[i]);
            }

            curr.pre_activations[i] = sum;
            curr.activations[i] = Sigmoid<accumulator_type>::forward(sum);
        }
    }
}

template <NetworkWeight T>
std::span<const typename BackpropNetwork<T>::accumulator_type> BackpropNetwork<T>::output() const {
    if (layers_.empty()) {
        return {};
    }
    return std::span<const accumulator_type>(layers_.back().activations);
}

template <NetworkWeight T>
double BackpropNetwork<T>::compute_error(std::span<const accumulator_type> target) {
    if (layers_.empty()) return 0.0;

    auto& output_layer = layers_.back();
    double error = 0.0;

    for (size_t i = 0; i < output_layer.output_size() && i < target.size(); ++i) {
        double out = static_cast<double>(output_layer.activations[i]);
        double tgt = static_cast<double>(target[i]);
        double err = tgt - out;
        output_layer.deltas[i] = static_cast<accumulator_type>(
            static_cast<double>(Sigmoid<accumulator_type>::derivative(output_layer.activations[i]))
            * err
        );
        error += 0.5 * err * err;
    }

    return error;
}

template <NetworkWeight T>
void BackpropNetwork<T>::backward(std::span<const accumulator_type> /*target*/, const TrainingConfig& config) {
    if (layers_.size() < 2) return;

    // Backpropagate errors
    for (size_t l = layers_.size() - 1; l > 1; --l) {
        auto& curr = layers_[l];
        auto& prev = layers_[l - 1];

        for (size_t j = 0; j < prev.output_size(); ++j) {
            accumulator_type err{};
            for (size_t i = 0; i < curr.output_size(); ++i) {
                if constexpr (Traits::is_quantized) {
                    err += static_cast<accumulator_type>(
                        static_cast<float>(curr.weights[i * prev.output_size() + j].value)
                        * curr.quant_scale
                    ) * curr.deltas[i];
                } else {
                    err += static_cast<accumulator_type>(curr.weights[i * prev.output_size() + j])
                         * curr.deltas[i];
                }
            }
            prev.deltas[j] = static_cast<accumulator_type>(
                static_cast<double>(Sigmoid<accumulator_type>::derivative(prev.activations[j]))
                * static_cast<double>(err)
            );
        }
    }

    // Adjust weights with momentum
    auto lr = static_cast<accumulator_type>(config.learning_rate);
    auto alpha = static_cast<accumulator_type>(config.momentum);

    for (size_t l = 1; l < layers_.size(); ++l) {
        auto& prev = layers_[l - 1];
        auto& curr = layers_[l];

        if constexpr (Traits::is_quantized) {
            // STE: update shadow weights with momentum
            float flr = static_cast<float>(config.learning_rate);
            float falpha = static_cast<float>(config.momentum);
            for (size_t i = 0; i < curr.output_size(); ++i) {
                for (size_t j = 0; j < prev.output_size(); ++j) {
                    size_t idx = i * prev.output_size() + j;
                    float grad = static_cast<float>(curr.deltas[i])
                               * static_cast<float>(prev.activations[j]);
                    float update = flr * grad + falpha * static_cast<float>(curr.prev_weight_updates[idx]);
                    curr.shadow_weights[idx] += update;
                    curr.prev_weight_updates[idx] = static_cast<accumulator_type>(update);
                }
                float bias_update = flr * static_cast<float>(curr.deltas[i])
                                  + falpha * static_cast<float>(curr.prev_bias_updates[i]);
                curr.shadow_biases[i] += bias_update;
                curr.prev_bias_updates[i] = static_cast<accumulator_type>(bias_update);
            }
        } else {
            for (size_t i = 0; i < curr.output_size(); ++i) {
                for (size_t j = 0; j < prev.output_size(); ++j) {
                    size_t idx = i * prev.output_size() + j;
                    auto grad = lr * curr.deltas[i] * prev.activations[j];
                    auto update = grad + alpha * curr.prev_weight_updates[idx];
                    curr.weights[idx] += static_cast<T>(update);
                    curr.prev_weight_updates[idx] = update;
                }
                auto bias_grad = lr * curr.deltas[i];
                auto bias_update = bias_grad + alpha * curr.prev_bias_updates[i];
                curr.biases[i] += static_cast<T>(bias_update);
                curr.prev_bias_updates[i] = bias_update;
            }
        }
    }
}

template <NetworkWeight T>
double BackpropNetwork<T>::simulate(std::span<const accumulator_type> input,
                                   std::span<const accumulator_type> target,
                                   bool training,
                                   const TrainingConfig& config) {
    forward(input);
    double error = compute_error(target);
    if (training) {
        backward(target, config);
    }
    return error;
}

template <NetworkWeight T>
TrainingResult BackpropNetwork<T>::train(std::span<const accumulator_type> inputs,
                                        std::span<const accumulator_type> targets,
                                        size_t num_samples,
                                        const TrainingConfig& config) {
    if (layers_.empty() || num_samples == 0) return {};

    size_t input_size = layers_.front().output_size();
    size_t output_size = layers_.back().output_size();

    TrainingResult result;

    for (int epoch = 0; epoch < config.max_epochs; ++epoch) {
        // Pick random sample
        size_t sample = static_cast<size_t>(Random::uniform_int(0, static_cast<int>(num_samples) - 1));
        auto input = inputs.subspan(sample * input_size, input_size);
        auto target = targets.subspan(sample * output_size, output_size);

        simulate(input, target, true, config);
        result.epochs_run = epoch + 1;
    }

    // Compute final average error across all samples
    double total_error = 0.0;
    for (size_t s = 0; s < num_samples; ++s) {
        auto input = inputs.subspan(s * input_size, input_size);
        auto target = targets.subspan(s * output_size, output_size);
        forward(input);
        total_error += compute_error(target);
    }
    result.final_error = total_error / static_cast<double>(num_samples);
    result.converged = result.final_error < config.error_threshold;
    return result;
}

template <NetworkWeight T>
TrainingResult BackpropNetwork<T>::train_early_stopping(
    std::span<const accumulator_type> inputs,
    std::span<const accumulator_type> targets,
    size_t num_samples,
    const TrainingConfig& config) {

    TrainingResult result;
    double min_error = std::numeric_limits<double>::max();
    int round = 0;

    while (true) {
        auto round_result = train(inputs, targets, num_samples, config);
        double error = round_result.final_error;
        ++round;

        if (progress_callback_) {
            progress_callback_(round, error);
        }

        if (error < min_error) {
            min_error = error;
            save_weights();
        } else {
            if ((error < 1.0) || (error > config.early_stopping_factor * min_error)) {
                restore_weights();
                result.epochs_run = round * config.max_epochs;
                result.final_error = min_error;
                result.converged = (min_error < config.error_threshold);
                return result;
            }
        }
    }
}

template <NetworkWeight T>
void BackpropNetwork<T>::save_weights() {
    saved_layers_ = layers_;
}

template <NetworkWeight T>
void BackpropNetwork<T>::restore_weights() {
    if (!saved_layers_.empty()) {
        layers_ = saved_layers_;
    }
}

template <NetworkWeight T>
nlohmann::json BackpropNetwork<T>::to_json() const {
    nlohmann::json j;
    j["type"] = Traits::is_quantized ? "ternary" :
                std::is_same_v<T, double> ? "double" :
                std::is_same_v<T, float> ? "float" :
                std::is_same_v<T, int> ? "int" : "char";

    j["layers"] = nlohmann::json::array();
    for (auto& layer : layers_) {
        nlohmann::json lj;
        lj["input_size"] = layer.input_size();
        lj["output_size"] = layer.output_size();

        if constexpr (Traits::is_quantized) {
            lj["shadow_weights"] = layer.shadow_weights;
            lj["shadow_biases"] = layer.shadow_biases;
            lj["quant_scale"] = layer.quant_scale;
        } else {
            // Convert weights to a serializable form
            if constexpr (std::is_floating_point_v<T>) {
                lj["weights"] = layer.weights;
                lj["biases"] = layer.biases;
            } else {
                std::vector<int> w(layer.weights.begin(), layer.weights.end());
                std::vector<int> b(layer.biases.begin(), layer.biases.end());
                lj["weights"] = w;
                lj["biases"] = b;
            }
        }

        j["layers"].push_back(lj);
    }
    return j;
}

template <NetworkWeight T>
BackpropNetwork<T> BackpropNetwork<T>::from_json(const nlohmann::json& j) {
    BackpropNetwork net;

    for (auto& lj : j.at("layers")) {
        size_t out_size = lj.at("output_size").get<size_t>();
        net.add_layer(out_size);

        auto& layer = net.layers_.back();

        if constexpr (Traits::is_quantized) {
            layer.shadow_weights = lj.at("shadow_weights").get<std::vector<float>>();
            layer.shadow_biases = lj.at("shadow_biases").get<std::vector<float>>();
            layer.quant_scale = lj.at("quant_scale").get<float>();
        } else {
            if constexpr (std::is_floating_point_v<T>) {
                layer.weights = lj.at("weights").get<std::vector<T>>();
                layer.biases = lj.at("biases").get<std::vector<T>>();
            } else {
                auto w = lj.at("weights").get<std::vector<int>>();
                auto b = lj.at("biases").get<std::vector<int>>();
                layer.weights.assign(w.begin(), w.end());
                layer.biases.assign(b.begin(), b.end());
            }
        }
    }
    return net;
}

// Explicit instantiations
template class Layer<double>;
template class Layer<float>;
template class Layer<int>;
template class Layer<char>;
template class Layer<TernaryWeight>;

template class BackpropNetwork<double>;
template class BackpropNetwork<float>;
template class BackpropNetwork<int>;
template class BackpropNetwork<char>;
template class BackpropNetwork<TernaryWeight>;

} // namespace crystal
