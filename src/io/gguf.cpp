#include "crystal/io/gguf.hpp"
#include "crystal/nn/backprop.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace crystal {

namespace {

void write_u32(std::ostream& os, uint32_t v) { os.write(reinterpret_cast<const char*>(&v), 4); }
void write_u64(std::ostream& os, uint64_t v) { os.write(reinterpret_cast<const char*>(&v), 8); }
void write_string(std::ostream& os, const std::string& s) {
    write_u64(os, s.size());
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

uint32_t read_u32(std::istream& is) { uint32_t v; is.read(reinterpret_cast<char*>(&v), 4); return v; }
uint64_t read_u64(std::istream& is) { uint64_t v; is.read(reinterpret_cast<char*>(&v), 8); return v; }
std::string read_string(std::istream& is) {
    uint64_t len = read_u64(is);
    std::string s(len, '\0');
    is.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}

void align_to(std::ostream& os, uint32_t alignment) {
    auto pos = os.tellp();
    auto pad = (alignment - (pos % alignment)) % alignment;
    for (int64_t i = 0; i < pad; ++i) {
        char zero = 0;
        os.write(&zero, 1);
    }
}

void align_to(std::istream& is, uint32_t alignment) {
    auto pos = is.tellg();
    auto pad = (alignment - (pos % alignment)) % alignment;
    is.seekg(pad, std::ios_base::cur);
}

void write_meta_value(std::ostream& os, const GGUFMetaValue& val) {
    std::visit([&os](const auto& v) {
        using V = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<V, uint32_t>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::UINT32));
            write_u32(os, v);
        } else if constexpr (std::is_same_v<V, int32_t>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::INT32));
            int32_t val = v; os.write(reinterpret_cast<const char*>(&val), 4);
        } else if constexpr (std::is_same_v<V, float>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::FLOAT32));
            float val = v; os.write(reinterpret_cast<const char*>(&val), 4);
        } else if constexpr (std::is_same_v<V, bool>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::BOOL));
            uint8_t val = v ? 1 : 0; os.write(reinterpret_cast<const char*>(&val), 1);
        } else if constexpr (std::is_same_v<V, std::string>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::STRING));
            write_string(os, v);
        } else if constexpr (std::is_same_v<V, uint64_t>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::UINT64));
            write_u64(os, v);
        } else if constexpr (std::is_same_v<V, int64_t>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::INT64));
            int64_t val = v; os.write(reinterpret_cast<const char*>(&val), 8);
        } else if constexpr (std::is_same_v<V, double>) {
            write_u32(os, static_cast<uint32_t>(GGUFMetaType::FLOAT64));
            double val = v; os.write(reinterpret_cast<const char*>(&val), 8);
        }
    }, val);
}

GGUFMetaValue read_meta_value(std::istream& is) {
    auto type = static_cast<GGUFMetaType>(read_u32(is));
    switch (type) {
        case GGUFMetaType::UINT32: return read_u32(is);
        case GGUFMetaType::INT32: { int32_t v; is.read(reinterpret_cast<char*>(&v), 4); return v; }
        case GGUFMetaType::FLOAT32: { float v; is.read(reinterpret_cast<char*>(&v), 4); return v; }
        case GGUFMetaType::BOOL: { uint8_t v; is.read(reinterpret_cast<char*>(&v), 1); return v != 0; }
        case GGUFMetaType::STRING: return read_string(is);
        case GGUFMetaType::UINT64: return read_u64(is);
        case GGUFMetaType::INT64: { int64_t v; is.read(reinterpret_cast<char*>(&v), 8); return v; }
        case GGUFMetaType::FLOAT64: { double v; is.read(reinterpret_cast<char*>(&v), 8); return v; }
        default: throw std::runtime_error("Unsupported GGUF metadata type");
    }
}

} // anonymous namespace

GGUFFile GGUFFile::read(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open GGUF file: " + path.string());

    uint32_t magic = read_u32(ifs);
    if (magic != GGUF_MAGIC) throw std::runtime_error("Invalid GGUF magic number");

    uint32_t version = read_u32(ifs);
    if (version < 2 || version > 3) throw std::runtime_error("Unsupported GGUF version");

    uint64_t num_tensors = read_u64(ifs);
    uint64_t num_metadata = read_u64(ifs);

    GGUFFile file;

    // Read metadata
    for (uint64_t i = 0; i < num_metadata; ++i) {
        std::string key = read_string(ifs);
        GGUFMetaValue val = read_meta_value(ifs);
        file.metadata_[key] = std::move(val);
    }

    // Read tensor descriptors
    std::vector<GGUFTensor> tensor_descs;
    for (uint64_t i = 0; i < num_tensors; ++i) {
        GGUFTensor t;
        t.name = read_string(ifs);
        uint32_t n_dims = read_u32(ifs);
        t.dimensions.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            t.dimensions[d] = read_u64(ifs);
        }
        t.type = static_cast<GGUFQuantType>(read_u32(ifs));
        t.offset = read_u64(ifs);
        tensor_descs.push_back(std::move(t));
    }

    // Align to data section
    align_to(ifs, GGUF_ALIGNMENT);
    auto data_start = ifs.tellg();

    // Read tensor data
    for (auto& t : tensor_descs) {
        ifs.seekg(static_cast<std::streamoff>(data_start) + static_cast<std::streamoff>(t.offset));

        // Compute total data size
        uint64_t num_elements = 1;
        for (auto dim : t.dimensions) num_elements *= dim;

        uint32_t block_sz = gguf_block_size(t.type);
        uint32_t type_sz = gguf_type_size(t.type);
        uint64_t num_blocks = (num_elements + block_sz - 1) / block_sz;
        uint64_t data_size = num_blocks * type_sz;

        t.data.resize(data_size);
        ifs.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(data_size));

        file.tensors_.push_back(std::move(t));
    }

    return file;
}

void GGUFFile::write(const std::filesystem::path& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot create GGUF file: " + path.string());

    write_u32(ofs, GGUF_MAGIC);
    write_u32(ofs, GGUF_VERSION);
    write_u64(ofs, tensors_.size());
    write_u64(ofs, metadata_.size());

    // Write metadata
    for (auto& [key, val] : metadata_) {
        write_string(ofs, key);
        write_meta_value(ofs, val);
    }

    // Compute tensor data offsets
    uint64_t data_offset = 0;
    std::vector<uint64_t> offsets;
    for (auto& t : tensors_) {
        offsets.push_back(data_offset);
        data_offset += t.data.size();
        // Align each tensor
        data_offset = (data_offset + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;
    }

    // Write tensor descriptors
    for (size_t i = 0; i < tensors_.size(); ++i) {
        auto& t = tensors_[i];
        write_string(ofs, t.name);
        write_u32(ofs, static_cast<uint32_t>(t.dimensions.size()));
        for (auto dim : t.dimensions) {
            write_u64(ofs, dim);
        }
        write_u32(ofs, static_cast<uint32_t>(t.type));
        write_u64(ofs, offsets[i]);
    }

    // Align to data section
    align_to(ofs, GGUF_ALIGNMENT);

    // Write tensor data
    for (size_t i = 0; i < tensors_.size(); ++i) {
        auto& t = tensors_[i];
        ofs.write(reinterpret_cast<const char*>(t.data.data()),
                 static_cast<std::streamsize>(t.data.size()));
        align_to(ofs, GGUF_ALIGNMENT);
    }
}

void GGUFFile::set_metadata(const std::string& key, GGUFMetaValue value) {
    metadata_[key] = std::move(value);
}

const GGUFMetaValue* GGUFFile::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? &it->second : nullptr;
}

void GGUFFile::add_tensor(GGUFTensor tensor) {
    tensors_.push_back(std::move(tensor));
}

const GGUFTensor* GGUFFile::get_tensor(const std::string& name) const {
    for (auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// --- Network conversion ---

template <>
GGUFFile GGUFFile::from_network(const BackpropNetwork<double>& net) {
    GGUFFile file;
    file.set_metadata("crystal.network_type", std::string("backprop"));
    file.set_metadata("crystal.weight_type", std::string("f64"));
    file.set_metadata("crystal.num_layers", static_cast<uint32_t>(net.num_layers()));

    auto sizes = net.layer_sizes();
    for (size_t i = 0; i < sizes.size(); ++i) {
        file.set_metadata("crystal.layer_size." + std::to_string(i),
                         static_cast<uint32_t>(sizes[i]));
    }

    for (size_t l = 1; l < net.layers().size(); ++l) {
        auto& layer = net.layers()[l];
        // Weights tensor
        GGUFTensor wt;
        wt.name = "layer." + std::to_string(l) + ".weights";
        wt.dimensions = {layer.output_size(), layer.input_size()};
        wt.type = GGUFQuantType::F32; // Store as F32 for compatibility
        wt.data.resize(layer.weights.size() * sizeof(float));
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            float f = static_cast<float>(layer.weights[i]);
            std::memcpy(wt.data.data() + i * sizeof(float), &f, sizeof(float));
        }
        file.add_tensor(std::move(wt));

        // Biases tensor
        GGUFTensor bt;
        bt.name = "layer." + std::to_string(l) + ".biases";
        bt.dimensions = {layer.output_size()};
        bt.type = GGUFQuantType::F32;
        bt.data.resize(layer.biases.size() * sizeof(float));
        for (size_t i = 0; i < layer.biases.size(); ++i) {
            float f = static_cast<float>(layer.biases[i]);
            std::memcpy(bt.data.data() + i * sizeof(float), &f, sizeof(float));
        }
        file.add_tensor(std::move(bt));
    }

    return file;
}

template <>
GGUFFile GGUFFile::from_network(const BackpropNetwork<float>& net) {
    GGUFFile file;
    file.set_metadata("crystal.network_type", std::string("backprop"));
    file.set_metadata("crystal.weight_type", std::string("f32"));
    file.set_metadata("crystal.num_layers", static_cast<uint32_t>(net.num_layers()));

    auto sizes = net.layer_sizes();
    for (size_t i = 0; i < sizes.size(); ++i) {
        file.set_metadata("crystal.layer_size." + std::to_string(i),
                         static_cast<uint32_t>(sizes[i]));
    }

    for (size_t l = 1; l < net.layers().size(); ++l) {
        auto& layer = net.layers()[l];

        GGUFTensor wt;
        wt.name = "layer." + std::to_string(l) + ".weights";
        wt.dimensions = {layer.output_size(), layer.input_size()};
        wt.type = GGUFQuantType::F32;
        wt.data.resize(layer.weights.size() * sizeof(float));
        std::memcpy(wt.data.data(), layer.weights.data(), wt.data.size());
        file.add_tensor(std::move(wt));

        GGUFTensor bt;
        bt.name = "layer." + std::to_string(l) + ".biases";
        bt.dimensions = {layer.output_size()};
        bt.type = GGUFQuantType::F32;
        bt.data.resize(layer.biases.size() * sizeof(float));
        std::memcpy(bt.data.data(), layer.biases.data(), bt.data.size());
        file.add_tensor(std::move(bt));
    }

    return file;
}

template <>
GGUFFile GGUFFile::from_network(const BackpropNetwork<TernaryWeight>& net) {
    GGUFFile file;
    file.set_metadata("crystal.network_type", std::string("backprop"));
    file.set_metadata("crystal.weight_type", std::string("ternary_b158"));
    file.set_metadata("crystal.num_layers", static_cast<uint32_t>(net.num_layers()));

    auto sizes = net.layer_sizes();
    for (size_t i = 0; i < sizes.size(); ++i) {
        file.set_metadata("crystal.layer_size." + std::to_string(i),
                         static_cast<uint32_t>(sizes[i]));
    }

    for (size_t l = 1; l < net.layers().size(); ++l) {
        auto& layer = net.layers()[l];
        size_t num_weights = layer.weights.size();

        // Pack ternary weights into groups
        size_t num_groups = (num_weights + TernaryGroup::NUM_WEIGHTS - 1) / TernaryGroup::NUM_WEIGHTS;

        GGUFTensor wt;
        wt.name = "layer." + std::to_string(l) + ".weights";
        wt.dimensions = {layer.output_size(), layer.input_size()};
        wt.type = GGUFQuantType::TERNARY_B158;
        wt.data.resize(num_groups * TernaryGroup::total_bytes());

        for (size_t g = 0; g < num_groups; ++g) {
            TernaryGroup group;
            size_t start = g * TernaryGroup::NUM_WEIGHTS;
            size_t count = std::min(TernaryGroup::NUM_WEIGHTS, num_weights - start);

            // Pad with zeros if needed
            std::vector<TernaryWeight> padded(TernaryGroup::NUM_WEIGHTS);
            for (size_t i = 0; i < count; ++i) {
                padded[i] = layer.weights[start + i];
            }

            group.pack(padded.data(), layer.quant_scale);
            std::memcpy(wt.data.data() + g * TernaryGroup::total_bytes(),
                       &group, TernaryGroup::total_bytes());
        }
        file.add_tensor(std::move(wt));

        // Store per-layer quant scale as metadata
        file.set_metadata("crystal.layer_scale." + std::to_string(l),
                         static_cast<float>(layer.quant_scale));

        // Shadow weights as F32
        GGUFTensor st;
        st.name = "layer." + std::to_string(l) + ".shadow_weights";
        st.dimensions = {layer.output_size(), layer.input_size()};
        st.type = GGUFQuantType::F32;
        st.data.resize(layer.shadow_weights.size() * sizeof(float));
        std::memcpy(st.data.data(), layer.shadow_weights.data(), st.data.size());
        file.add_tensor(std::move(st));

        // Shadow biases as F32
        GGUFTensor sb;
        sb.name = "layer." + std::to_string(l) + ".shadow_biases";
        sb.dimensions = {layer.output_size()};
        sb.type = GGUFQuantType::F32;
        sb.data.resize(layer.shadow_biases.size() * sizeof(float));
        std::memcpy(sb.data.data(), layer.shadow_biases.data(), sb.data.size());
        file.add_tensor(std::move(sb));
    }

    return file;
}

template <>
BackpropNetwork<double> GGUFFile::to_network() const {
    BackpropNetwork<double> net;

    auto* num_layers_ptr = get_metadata("crystal.num_layers");
    if (!num_layers_ptr) throw std::runtime_error("Missing crystal.num_layers metadata");
    uint32_t num_layers = std::get<uint32_t>(*num_layers_ptr);

    for (uint32_t i = 0; i < num_layers; ++i) {
        auto* size_ptr = get_metadata("crystal.layer_size." + std::to_string(i));
        if (!size_ptr) throw std::runtime_error("Missing layer size metadata");
        net.add_layer(std::get<uint32_t>(*size_ptr));
    }

    for (uint32_t l = 1; l < num_layers; ++l) {
        auto& layer = net.layers()[l];
        auto* wt = get_tensor("layer." + std::to_string(l) + ".weights");
        if (wt) {
            for (size_t i = 0; i < layer.weights.size(); ++i) {
                float f;
                std::memcpy(&f, wt->data.data() + i * sizeof(float), sizeof(float));
                layer.weights[i] = static_cast<double>(f);
            }
        }
        auto* bt = get_tensor("layer." + std::to_string(l) + ".biases");
        if (bt) {
            for (size_t i = 0; i < layer.biases.size(); ++i) {
                float f;
                std::memcpy(&f, bt->data.data() + i * sizeof(float), sizeof(float));
                layer.biases[i] = static_cast<double>(f);
            }
        }
    }

    return net;
}

template <>
BackpropNetwork<float> GGUFFile::to_network() const {
    BackpropNetwork<float> net;

    auto* num_layers_ptr = get_metadata("crystal.num_layers");
    if (!num_layers_ptr) throw std::runtime_error("Missing crystal.num_layers metadata");
    uint32_t num_layers = std::get<uint32_t>(*num_layers_ptr);

    for (uint32_t i = 0; i < num_layers; ++i) {
        auto* size_ptr = get_metadata("crystal.layer_size." + std::to_string(i));
        if (!size_ptr) throw std::runtime_error("Missing layer size metadata");
        net.add_layer(std::get<uint32_t>(*size_ptr));
    }

    for (uint32_t l = 1; l < num_layers; ++l) {
        auto& layer = net.layers()[l];
        auto* wt = get_tensor("layer." + std::to_string(l) + ".weights");
        if (wt) {
            std::memcpy(layer.weights.data(), wt->data.data(),
                       layer.weights.size() * sizeof(float));
        }
        auto* bt = get_tensor("layer." + std::to_string(l) + ".biases");
        if (bt) {
            std::memcpy(layer.biases.data(), bt->data.data(),
                       layer.biases.size() * sizeof(float));
        }
    }

    return net;
}

template <>
BackpropNetwork<TernaryWeight> GGUFFile::to_network() const {
    BackpropNetwork<TernaryWeight> net;

    auto* num_layers_ptr = get_metadata("crystal.num_layers");
    if (!num_layers_ptr) throw std::runtime_error("Missing crystal.num_layers metadata");
    uint32_t num_layers = std::get<uint32_t>(*num_layers_ptr);

    for (uint32_t i = 0; i < num_layers; ++i) {
        auto* size_ptr = get_metadata("crystal.layer_size." + std::to_string(i));
        if (!size_ptr) throw std::runtime_error("Missing layer size metadata");
        net.add_layer(std::get<uint32_t>(*size_ptr));
    }

    for (uint32_t l = 1; l < num_layers; ++l) {
        auto& layer = net.layers()[l];

        // Unpack ternary weights
        auto* wt = get_tensor("layer." + std::to_string(l) + ".weights");
        if (wt) {
            size_t num_weights = layer.weights.size();
            size_t num_groups = (num_weights + TernaryGroup::NUM_WEIGHTS - 1) / TernaryGroup::NUM_WEIGHTS;

            for (size_t g = 0; g < num_groups; ++g) {
                TernaryGroup group;
                std::memcpy(&group, wt->data.data() + g * TernaryGroup::total_bytes(),
                           TernaryGroup::total_bytes());

                std::vector<TernaryWeight> unpacked(TernaryGroup::NUM_WEIGHTS);
                float scale;
                group.unpack(unpacked.data(), scale);

                size_t start = g * TernaryGroup::NUM_WEIGHTS;
                size_t count = std::min(TernaryGroup::NUM_WEIGHTS, num_weights - start);
                for (size_t i = 0; i < count; ++i) {
                    layer.weights[start + i] = unpacked[i];
                }
            }
        }

        // Restore per-layer quant scale from metadata (not from per-group packed scale)
        auto* scale_ptr = get_metadata("crystal.layer_scale." + std::to_string(l));
        if (scale_ptr) {
            layer.quant_scale = std::get<float>(*scale_ptr);
        }

        // Shadow weights
        auto* st = get_tensor("layer." + std::to_string(l) + ".shadow_weights");
        if (st) {
            std::memcpy(layer.shadow_weights.data(), st->data.data(),
                       layer.shadow_weights.size() * sizeof(float));
        }

        // Shadow biases
        auto* sb = get_tensor("layer." + std::to_string(l) + ".shadow_biases");
        if (sb) {
            std::memcpy(layer.shadow_biases.data(), sb->data.data(),
                       layer.shadow_biases.size() * sizeof(float));
        }
    }

    return net;
}

} // namespace crystal
