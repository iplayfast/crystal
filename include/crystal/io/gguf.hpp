#pragma once

#include "crystal/core/types.hpp"
#include "crystal/io/gguf_types.hpp"
#include "crystal/nn/backprop.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace crystal {

/// A tensor descriptor in a GGUF file
struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> dimensions;
    GGUFQuantType type{GGUFQuantType::F32};
    uint64_t offset{0}; // offset into data section
    std::vector<uint8_t> data;
};

/// Metadata value (subset of types we support)
using GGUFMetaValue = std::variant<
    uint32_t, int32_t, float, bool, std::string, uint64_t, int64_t, double
>;

/// GGUF file reader/writer
class GGUFFile {
public:
    GGUFFile() = default;

    /// Read a GGUF file from disk
    static GGUFFile read(const std::filesystem::path& path);

    /// Write this GGUF file to disk
    void write(const std::filesystem::path& path) const;

    /// Metadata access
    void set_metadata(const std::string& key, GGUFMetaValue value);
    [[nodiscard]] const GGUFMetaValue* get_metadata(const std::string& key) const;
    [[nodiscard]] const std::map<std::string, GGUFMetaValue>& metadata() const { return metadata_; }

    /// Tensor access
    void add_tensor(GGUFTensor tensor);
    [[nodiscard]] const GGUFTensor* get_tensor(const std::string& name) const;
    [[nodiscard]] const std::vector<GGUFTensor>& tensors() const { return tensors_; }

    /// Convert a BackpropNetwork to GGUF tensors
    template <NetworkWeight T>
    static GGUFFile from_network(const class BackpropNetwork<T>& net);

    /// Reconstruct a BackpropNetwork from GGUF tensors
    template <NetworkWeight T>
    BackpropNetwork<T> to_network() const;

private:
    std::map<std::string, GGUFMetaValue> metadata_;
    std::vector<GGUFTensor> tensors_;
};

} // namespace crystal
