#include "crystal/quantize/model_reader.hpp"
#include "crystal/quantize/ternary_converter.hpp"
#include "crystal/quantize/ensemble.hpp"
#include "crystal/quantize/calibration.hpp"
#include "crystal/quantize/pipeline.hpp"
#include "crystal/core/types.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

void test_quantize_tensor_basic() {
    std::cout << "test_quantize_tensor_basic... ";

    // Create a simple weight vector
    std::vector<float> weights = {
        0.5f, -0.3f, 0.0f, 0.8f, -1.0f, 0.1f, -0.05f, 0.6f,
        0.4f, -0.7f, 0.2f, -0.9f, 0.0f, 0.3f, -0.4f, 0.15f
    };

    // Pad to at least 128 for a full TernaryGroup
    weights.resize(128, 0.0f);

    std::vector<uint64_t> shape = {128};

    auto result = crystal::quantize_tensor("test_tensor", weights, {}, shape);

    assert(result.name == "test_tensor");
    assert(result.shape.size() == 1);
    assert(result.shape[0] == 128);
    assert(result.groups.size() == 1);  // 128 weights = 1 group
    assert(result.scale > 0.0f);

    // Unpack and verify
    crystal::TernaryWeight unpacked[128];
    float scale;
    result.groups[0].unpack(unpacked, scale);

    // The large positive weight (0.8) should map to +1
    // The large negative weight (-1.0) should map to -1
    // Near-zero weights should map to 0
    // (exact mapping depends on scale = absmean)
    assert(scale > 0.0f);

    std::cout << "PASSED (scale=" << result.scale << ")\n";
}

void test_quantize_tensor_with_importance() {
    std::cout << "test_quantize_tensor_with_importance... ";

    std::vector<float> weights(256, 0.0f);
    std::vector<float> importance(256, 0.0f);

    // Set some weights
    weights[0] = 0.5f;
    weights[1] = -0.5f;
    weights[128] = 0.3f;
    weights[129] = -0.3f;

    // High importance for first group, low for second
    for (int i = 0; i < 128; ++i) importance[i] = 1.0f;
    for (int i = 128; i < 256; ++i) importance[i] = 0.0f;

    std::vector<uint64_t> shape = {256};

    auto result = crystal::quantize_tensor("test_imp", weights, importance, shape);

    assert(result.groups.size() == 2);  // 256 / 128 = 2 groups
    assert(result.scale > 0.0f);

    std::cout << "PASSED\n";
}

void test_quantize_model() {
    std::cout << "test_quantize_model... ";

    crystal::ModelTensors model;
    model.path = "test_model";

    // Create a tensor that should be quantized
    crystal::TensorData td;
    td.name = "blk.0.attn_q.weight";
    td.shape = {128, 128};
    td.type = crystal::GGUFQuantType::F32;
    td.data.resize(128 * 128);
    for (size_t i = 0; i < td.data.size(); ++i) {
        td.data[i] = static_cast<float>(i % 7 - 3) * 0.1f;  // some varied weights
    }
    model.tensors.push_back(td);

    // Create a tensor that should be kept (matches "embed|output")
    crystal::TensorData embed;
    embed.name = "token_embd.weight";
    embed.shape = {128, 64};
    embed.type = crystal::GGUFQuantType::F32;
    embed.data.resize(128 * 64, 0.5f);
    model.tensors.push_back(embed);

    crystal::QuantizeOptions opts;
    opts.keep_layers_regex = "embd|output";
    opts.verbose = false;

    auto result = crystal::quantize_model(model, opts);

    assert(result.tensors.size() == 1);  // only attn_q quantized
    assert(result.tensors[0].name == "blk.0.attn_q.weight");
    assert(result.original_size_bytes > 0);
    assert(result.quantized_size_bytes > 0);
    assert(result.compression_ratio > 0.0);
    assert(result.compression_ratio < 1.0);  // should be smaller

    std::cout << "PASSED (compression=" << (result.compression_ratio * 100.0) << "%)\n";
}

void test_ensemble_average() {
    std::cout << "test_ensemble_average... ";

    crystal::ModelTensors model1;
    model1.path = "model1";
    crystal::TensorData t1;
    t1.name = "weight";
    t1.shape = {4};
    t1.type = crystal::GGUFQuantType::F32;
    t1.data = {1.0f, 2.0f, 3.0f, 4.0f};
    model1.tensors.push_back(t1);

    crystal::ModelTensors model2;
    model2.path = "model2";
    crystal::TensorData t2;
    t2.name = "weight";
    t2.shape = {4};
    t2.type = crystal::GGUFQuantType::F32;
    t2.data = {3.0f, 4.0f, 5.0f, 6.0f};
    model2.tensors.push_back(t2);

    std::vector<crystal::ModelTensors> models = {model1, model2};
    auto averaged = crystal::ensemble_average(models);

    assert(averaged.tensors.size() == 1);
    assert(averaged.tensors[0].name == "weight");
    assert(averaged.tensors[0].data.size() == 4);

    // Average of [1,2,3,4] and [3,4,5,6] = [2,3,4,5]
    assert(std::abs(averaged.tensors[0].data[0] - 2.0f) < 1e-5f);
    assert(std::abs(averaged.tensors[0].data[1] - 3.0f) < 1e-5f);
    assert(std::abs(averaged.tensors[0].data[2] - 4.0f) < 1e-5f);
    assert(std::abs(averaged.tensors[0].data[3] - 5.0f) < 1e-5f);

    std::cout << "PASSED\n";
}

void test_ensemble_single_model() {
    std::cout << "test_ensemble_single_model... ";

    crystal::ModelTensors model;
    model.path = "model";
    crystal::TensorData t;
    t.name = "w";
    t.shape = {2};
    t.type = crystal::GGUFQuantType::F32;
    t.data = {1.0f, 2.0f};
    model.tensors.push_back(t);

    std::vector<crystal::ModelTensors> models = {model};
    auto result = crystal::ensemble_average(models);

    assert(result.tensors.size() == 1);
    assert(std::abs(result.tensors[0].data[0] - 1.0f) < 1e-5f);
    assert(std::abs(result.tensors[0].data[1] - 2.0f) < 1e-5f);

    std::cout << "PASSED\n";
}

void test_importance_from_weights() {
    std::cout << "test_importance_from_weights... ";

    crystal::ModelTensors model;
    model.path = "test";

    crystal::TensorData td;
    td.name = "layer.weight";
    td.shape = {4};
    td.type = crystal::GGUFQuantType::F32;
    td.data = {0.5f, -1.0f, 0.0f, 0.25f};
    model.tensors.push_back(td);

    auto imatrix = crystal::compute_importance_from_weights(model);

    assert(imatrix.tensor_importance.count("layer.weight") == 1);
    auto& imp = imatrix.tensor_importance["layer.weight"];
    assert(imp.size() == 4);

    // max_abs = 1.0, so importance = |w| / 1.0
    assert(std::abs(imp[0] - 0.5f) < 1e-5f);
    assert(std::abs(imp[1] - 1.0f) < 1e-5f);
    assert(std::abs(imp[2] - 0.0f) < 1e-5f);
    assert(std::abs(imp[3] - 0.25f) < 1e-5f);

    std::cout << "PASSED\n";
}

void test_ternary_group_roundtrip() {
    std::cout << "test_ternary_group_roundtrip... ";

    crystal::TernaryWeight weights[128];
    for (int i = 0; i < 128; ++i) {
        weights[i] = crystal::TernaryWeight(static_cast<int8_t>((i % 3) - 1));
    }
    float scale = 0.42f;

    crystal::TernaryGroup group;
    group.pack(weights, scale);

    crystal::TernaryWeight unpacked[128];
    float unpacked_scale;
    group.unpack(unpacked, unpacked_scale);

    for (int i = 0; i < 128; ++i) {
        assert(unpacked[i].value == weights[i].value);
    }
    // FP16 roundtrip may lose some precision
    assert(std::abs(unpacked_scale - scale) < 0.01f);

    std::cout << "PASSED\n";
}

void test_pipeline_validation() {
    std::cout << "test_pipeline_validation... ";

    // Empty input should fail gracefully
    crystal::PipelineOptions opts;
    auto result = crystal::run_pipeline(opts);
    assert(!result.success);
    assert(!result.error_message.empty());

    // No output path should fail
    opts.input_models.push_back("nonexistent.gguf");
    opts.output_path = "";
    result = crystal::run_pipeline(opts);
    assert(!result.success);

    std::cout << "PASSED\n";
}

void test_quantize_compression_ratio() {
    std::cout << "test_quantize_compression_ratio... ";

    // Create a large-ish tensor to test compression
    crystal::ModelTensors model;
    model.path = "test";

    crystal::TensorData td;
    td.name = "big_weight";
    td.shape = {1024, 1024};
    td.type = crystal::GGUFQuantType::F32;
    td.data.resize(1024 * 1024);
    for (size_t i = 0; i < td.data.size(); ++i) {
        td.data[i] = static_cast<float>(i % 100 - 50) * 0.01f;
    }
    model.tensors.push_back(td);

    crystal::QuantizeOptions opts;
    opts.keep_layers_regex = "^$";  // keep nothing
    auto result = crystal::quantize_model(model, opts);

    // F32: 1024*1024*4 = 4MB
    // Ternary: ~1024*1024*2/8 ≈ 256KB (plus scale overhead)
    // Compression should be roughly 1/16 ≈ 6.25%
    assert(result.compression_ratio < 0.15);  // should be well under 15%
    assert(result.compression_ratio > 0.01);  // but not zero

    std::cout << "PASSED (compression=" << (result.compression_ratio * 100.0) << "%)\n";
}

}  // namespace

int main() {
    std::cout << "=== Quantize Pipeline Tests ===\n\n";

    test_ternary_group_roundtrip();
    test_quantize_tensor_basic();
    test_quantize_tensor_with_importance();
    test_quantize_model();
    test_ensemble_average();
    test_ensemble_single_model();
    test_importance_from_weights();
    test_pipeline_validation();
    test_quantize_compression_ratio();

    std::cout << "\nAll quantize pipeline tests passed!\n";
    return 0;
}
