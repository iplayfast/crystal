#pragma once

#include "crystal/nn/quantization.hpp"
#include "crystal/quantize/model_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace crystal {

struct QuantizedTensor {
    std::string name;
    std::vector<uint64_t> shape;
    std::vector<TernaryGroup> groups;
    float scale;
};

struct QuantizationResult {
    std::vector<QuantizedTensor> tensors;
    size_t original_size_bytes = 0;
    size_t quantized_size_bytes = 0;
    double compression_ratio = 0.0;
};

struct QuantizeOptions {
    std::string keep_layers_regex = "embed|output";  // layers to keep at F16
    bool use_importance = true;
    bool verbose = false;
};

QuantizedTensor quantize_tensor(
    const std::string& name,
    std::span<const float> weights,
    std::span<const float> importance,
    const std::vector<uint64_t>& shape);

QuantizationResult quantize_model(
    const ModelTensors& model,
    const QuantizeOptions& options = {});

}  // namespace crystal