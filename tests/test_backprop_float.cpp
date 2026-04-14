#include "crystal/nn/backprop.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace crystal;

void test_xor_float() {
    Random::seed(42);

    BackpropNetwork<float> net;
    net.add_layer(2);
    net.add_layer(8);
    net.add_layer(1);
    net.randomize_weights();

    std::vector<float> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<float> targets = {0, 1, 1, 0};

    TrainingConfig config;
    config.learning_rate = 0.5;
    config.momentum = 0.9;
    config.max_epochs = 10000;

    auto result = net.train(inputs, targets, 4, config);
    std::cout << "Float XOR: " << result.epochs_run << " epochs, error: " << result.final_error << "\n";

    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        std::span<const float> in(inputs.data() + i * 2, 2);
        net.forward(in);
        float out = net.output()[0];
        int predicted = (out > 0.5f) ? 1 : 0;
        int expected = static_cast<int>(targets[static_cast<size_t>(i)]);
        std::cout << "  Input: " << inputs[static_cast<size_t>(i * 2)] << "," << inputs[static_cast<size_t>(i * 2 + 1)]
                  << " -> " << out << " (expected " << targets[static_cast<size_t>(i)] << ")\n";
        if (predicted == expected) ++correct;
    }
    assert(correct >= 3);
    std::cout << "test_xor_float PASSED (" << correct << "/4 correct)\n";
}

int main() {
    test_xor_float();
    std::cout << "All float backprop tests PASSED\n";
    return 0;
}
