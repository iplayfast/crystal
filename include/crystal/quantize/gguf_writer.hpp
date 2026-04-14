#pragma once

#include "crystal/quantize/ternary_converter.hpp"
#include "crystal/quantize/model_reader.hpp"

#include <filesystem>
#include <string>

namespace crystal {

bool write_quantized_gguf(
    const std::filesystem::path& output_path,
    const ModelTensors& model,
    const std::vector<QuantizedTensor>& quantized_tensors,
    const QuantizeOptions& options);

}  // namespace crystal
