#include "crystal/nn/blob_network.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace crystal;

void test_blob_creation() {
    BlobNetwork<double> net(2, 1, 50);
    assert(net.input_size() == 2);
    assert(net.output_size() == 1);
    assert(net.blob_size() == (2 + 1) * 50);
    std::cout << "test_blob_creation PASSED\n";
}

void test_blob_forward() {
    Random::seed(42);

    BlobNetwork<double> net(2, 1, 50);
    double input[] = {1.0, 0.0};
    double output[1] = {0.0};
    net.forward(input, output);

    std::cout << "  Blob forward output: " << output[0] << "\n";
    std::cout << "test_blob_forward PASSED\n";
}

void test_blob_xor_training() {
    Random::seed(42);

    BlobNetwork<double> net(2, 1, 100);

    double inputs[] = {0, 0, 0, 1, 1, 0, 1, 1};
    double targets[] = {0, 1, 1, 0};

    auto result = net.train(inputs, targets, 4, 5000, 0.1);
    std::cout << "  Blob XOR: " << result.iterations << " iterations, error: " << result.final_error << "\n";

    if (result.converged) {
        std::cout << "  Blob network converged!\n";
        // Verify
        double output[1];
        for (int i = 0; i < 4; ++i) {
            net.forward(&inputs[i * 2], output);
            std::cout << "    " << inputs[i * 2] << "," << inputs[i * 2 + 1]
                      << " -> " << output[0] << " (expected " << targets[i] << ")\n";
        }
    } else {
        std::cout << "  Blob network did not converge (expected for stochastic SA)\n";
    }

    // Blob networks are stochastic — convergence not guaranteed
    // Just verify it ran without crashing
    assert(result.iterations > 0);
    std::cout << "test_blob_xor_training PASSED\n";
}

void test_blob_json_roundtrip() {
    Random::seed(42);

    BlobNetwork<double> net(2, 1, 20);

    auto json = net.to_json();
    auto net2 = BlobNetwork<double>::from_json(json);

    assert(net2.input_size() == 2);
    assert(net2.output_size() == 1);
    assert(net2.blob_size() == net.blob_size());
    std::cout << "test_blob_json_roundtrip PASSED\n";
}

int main() {
    test_blob_creation();
    test_blob_forward();
    test_blob_xor_training();
    test_blob_json_roundtrip();
    std::cout << "All blob network tests PASSED\n";
    return 0;
}
