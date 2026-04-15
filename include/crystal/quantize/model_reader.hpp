#pragma once

#include "crystal/io/gguf_types.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace crystal {

// Use the gguf_type enum from gguf.h
// We'll use uint32_t to store the type to avoid needing the full enum

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
    std::map<std::string, uint32_t> metadata_types;  // gguf_type as uint32_t
};

ModelTensors read_model(const std::filesystem::path& gguf_path);

}  // namespace crystal