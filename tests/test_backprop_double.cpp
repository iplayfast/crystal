#include "crystal/nn/backprop.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace crystal;

void test_xor() {
    Random::seed(42);

    BackpropNetwork<double> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    // XOR dataset: 4 samples, 2 inputs each, 1 output each
    std::vector<double> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<double> targets = {0, 1, 1, 0};

    TrainingConfig config;
    config.learning_rate = 0.5;
    config.momentum = 0.9;
    config.max_epochs = 5000;

    auto result = net.train(inputs, targets, 4, config);
    std::cout << "XOR training: " << result.epochs_run << " epochs, error: " << result.final_error << "\n";

    // Verify outputs
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        std::span<const double> in(inputs.data() + i * 2, 2);
        net.forward(in);
        double out = net.output()[0];
        double expected = targets[static_cast<size_t>(i)];
        int predicted = (out > 0.5) ? 1 : 0;
        int target_val = static_cast<int>(expected);
        std::cout << "  Input: " << inputs[static_cast<size_t>(i * 2)] << "," << inputs[static_cast<size_t>(i * 2 + 1)]
                  << " -> " << out << " (expected " << expected << ")\n";
        if (predicted == target_val) ++correct;
    }
    assert(correct >= 3); // Should get at least 3/4 correct after training
    std::cout << "test_xor PASSED (" << correct << "/4 correct)\n";
}

void test_json_roundtrip() {
    Random::seed(123);

    BackpropNetwork<double> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    // Train briefly
    std::vector<double> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<double> targets = {0, 1, 1, 0};
    TrainingConfig config;
    config.max_epochs = 100;
    net.train(inputs, targets, 4, config);

    // Serialize and deserialize
    auto json = net.to_json();
    auto net2 = BackpropNetwork<double>::from_json(json);

    // Verify same outputs
    for (int i = 0; i < 4; ++i) {
        std::span<const double> in(inputs.data() + i * 2, 2);
        net.forward(in);
        double out1 = net.output()[0];
        net2.forward(in);
        double out2 = net2.output()[0];
        assert(std::abs(out1 - out2) < 1e-10);
    }
    std::cout << "test_json_roundtrip PASSED\n";
}

void test_save_restore() {
    Random::seed(99);

    BackpropNetwork<double> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    // Get output before saving
    std::vector<double> input = {1.0, 0.0};
    net.forward(std::span<const double>(input));
    double out_before = net.output()[0];

    net.save_weights();

    // Train to change weights
    std::vector<double> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<double> targets = {0, 1, 1, 0};
    TrainingConfig config;
    config.max_epochs = 500;
    net.train(inputs, targets, 4, config);

    // Restore and verify same output
    net.restore_weights();
    net.forward(std::span<const double>(input));
    double out_after = net.output()[0];

    assert(std::abs(out_before - out_after) < 1e-10);
    std::cout << "test_save_restore PASSED\n";
}

int main() {
    test_xor();
    test_json_roundtrip();
    test_save_restore();
    std::cout << "All double backprop tests PASSED\n";
    return 0;
}
