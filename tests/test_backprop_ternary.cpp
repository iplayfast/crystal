#include "crystal/nn/backprop.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace crystal;

void test_ternary_creation() {
    BackpropNetwork<TernaryWeight> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    assert(net.num_layers() == 3);
    // Shadow weights should be initialized
    assert(!net.layers()[1].shadow_weights.empty());
    std::cout << "test_ternary_creation PASSED\n";
}

void test_ternary_training() {
    Random::seed(42);

    BackpropNetwork<TernaryWeight> net;
    net.add_layer(2);
    net.add_layer(8); // More hidden units for ternary
    net.add_layer(1);
    net.randomize_weights();

    // XOR dataset (accumulator_type is float for TernaryWeight)
    std::vector<float> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<float> targets = {0, 1, 1, 0};

    TrainingConfig config;
    config.learning_rate = 0.5;
    config.max_epochs = 10000;

    auto result = net.train(inputs, targets, 4, config);
    std::cout << "Ternary XOR: " << result.epochs_run << " epochs, error: " << result.final_error << "\n";

    // Verify weights are ternary {-1, 0, +1}
    for (size_t l = 1; l < net.layers().size(); ++l) {
        for (auto& w : net.layers()[l].weights) {
            assert(w.value == -1 || w.value == 0 || w.value == 1);
        }
    }
    std::cout << "test_ternary_training PASSED (all weights are ternary)\n";
}

void test_ternary_json_roundtrip() {
    Random::seed(42);

    BackpropNetwork<TernaryWeight> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    auto json = net.to_json();
    auto net2 = BackpropNetwork<TernaryWeight>::from_json(json);

    assert(net2.num_layers() == 3);
    assert(net2.layers()[1].shadow_weights.size() == net.layers()[1].shadow_weights.size());
    std::cout << "test_ternary_json_roundtrip PASSED\n";
}

int main() {
    test_ternary_creation();
    test_ternary_training();
    test_ternary_json_roundtrip();
    std::cout << "All ternary backprop tests PASSED\n";
    return 0;
}
