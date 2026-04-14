#include "crystal/nn/quantization.hpp"
#include "crystal/core/types.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace crystal;

void test_absmean_quantizer() {
    // Known input: [0.5, -0.3, 0.8, -0.1, 0.0]
    std::vector<float> weights = {0.5f, -0.3f, 0.8f, -0.1f, 0.0f};
    float expected_scale = (0.5f + 0.3f + 0.8f + 0.1f + 0.0f) / 5.0f; // 0.34

    float scale = AbsmeanQuantizer::compute_scale(weights);
    assert(std::abs(scale - expected_scale) < 1e-5f);
    std::cout << "  Absmean scale: " << scale << " (expected " << expected_scale << ")\n";

    std::vector<TernaryWeight> quantized(weights.size());
    float out_scale;
    AbsmeanQuantizer::quantize(weights, quantized, out_scale);

    // 0.5/0.34 ≈ 1.47 → 1, -0.3/0.34 ≈ -0.88 → -1, 0.8/0.34 ≈ 2.35 → 1,
    // -0.1/0.34 ≈ -0.29 → 0, 0.0/0.34 = 0 → 0
    assert(quantized[0].value == 1);
    assert(quantized[1].value == -1);
    assert(quantized[2].value == 1);
    assert(quantized[3].value == 0);
    assert(quantized[4].value == 0);
    std::cout << "test_absmean_quantizer PASSED\n";
}

void test_absmax_activation_quantizer() {
    std::vector<float> activations = {1.0f, -0.5f, 0.25f, -1.0f};
    std::vector<int8_t> quantized(4);

    float scale = AbsmaxActivationQuantizer::quantize(activations, quantized);
    assert(std::abs(scale - 1.0f) < 1e-5f);

    // 1.0 * 127 / 1.0 = 127, -0.5 * 127 = -63.5 → -64,
    // 0.25 * 127 = 31.75 → 32, -1.0 * 127 = -127
    assert(quantized[0] == 127);
    assert(quantized[1] == -64 || quantized[1] == -63); // rounding
    assert(quantized[3] == -127);
    std::cout << "test_absmax_activation_quantizer PASSED\n";

    // Dequantize
    std::vector<float> dequantized(4);
    AbsmaxActivationQuantizer::dequantize(quantized, scale, dequantized);
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(dequantized[static_cast<size_t>(i)] - activations[static_cast<size_t>(i)]) < 0.02f);
    }
    std::cout << "  Dequantization error within tolerance\n";
}

void test_ternary_group_pack_unpack() {
    // Create 128 known ternary weights
    std::vector<TernaryWeight> weights(128);
    for (size_t i = 0; i < 128; ++i) {
        if (i % 3 == 0) weights[i].value = 1;
        else if (i % 3 == 1) weights[i].value = -1;
        else weights[i].value = 0;
    }

    float scale_in = 0.42f;
    TernaryGroup group;
    group.pack(weights.data(), scale_in);

    // Unpack and verify
    std::vector<TernaryWeight> unpacked(128);
    float scale_out;
    group.unpack(unpacked.data(), scale_out);

    for (size_t i = 0; i < 128; ++i) {
        assert(unpacked[i].value == weights[i].value);
    }

    // FP16 round-trip loses some precision
    assert(std::abs(scale_out - scale_in) < 0.01f);
    std::cout << "test_ternary_group_pack_unpack PASSED (scale: " << scale_in << " -> " << scale_out << ")\n";
}

void test_ste() {
    std::vector<float> gradients = {0.1f, -0.2f, 0.05f};
    std::vector<float> shadow = {1.0f, 2.0f, 3.0f};
    float lr = 0.5f;

    STE::apply(gradients, shadow, lr);

    assert(std::abs(shadow[0] - 0.95f) < 1e-5f);  // 1.0 - 0.5 * 0.1
    assert(std::abs(shadow[1] - 2.1f) < 1e-5f);   // 2.0 - 0.5 * (-0.2)
    assert(std::abs(shadow[2] - 2.975f) < 1e-5f);  // 3.0 - 0.5 * 0.05
    std::cout << "test_ste PASSED\n";
}

int main() {
    test_absmean_quantizer();
    test_absmax_activation_quantizer();
    test_ternary_group_pack_unpack();
    test_ste();
    std::cout << "All quantization tests PASSED\n";
    return 0;
}
