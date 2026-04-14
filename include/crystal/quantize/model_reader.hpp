#pragma once

#include "crystal/io/gguf_types.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace crystal {

struct TensorData {
    std::string name;
    std::vector<uint64_t> shape;
    GGUFQuantType type;
    std::vector<float> data;  // always F32 after dequantization
};

struct ModelTensors {
    std::string path;
    std::vector<TensorData> tensors;
    std::map<std::string, std::string> metadata;
};

ModelTensors read_model(const std::filesystem::path& gguf_path);

}  // namespace crystal