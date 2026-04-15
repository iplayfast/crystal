#include "crystal/quantize/gguf_writer.hpp"
#include "crystal/quantize/model_reader.hpp"
#include "crystal/quantize/ternary_converter.hpp"

#include <ggml.h>
#include <gguf.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

namespace crystal {

bool write_quantized_gguf(
    const std::filesystem::path& output_path,
    const ModelTensors& model,
    const std::vector<QuantizedTensor>& quantized_tensors,
    const QuantizeOptions& options) {

    struct gguf_context* gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        std::cerr << "Failed to create GGUF context\n";
        return false;
    }

    // Copy original metadata with correct types
    for (const auto& [key, value] : model.metadata) {
        auto it = model.metadata_types.find(key);
        uint32_t type = it != model.metadata_types.end() ? it->second : GGUF_TYPE_STRING;
        
        switch (type) {
            case GGUF_TYPE_STRING:
                gguf_set_val_str(gguf_ctx, key.c_str(), value.c_str());
                break;
            case GGUF_TYPE_UINT32: {
                uint32_t v = static_cast<uint32_t>(std::stoul(value));
                gguf_set_val_u32(gguf_ctx, key.c_str(), v);
                break;
            }
            case GGUF_TYPE_INT32: {
                int32_t v = static_cast<int32_t>(std::stol(value));
                gguf_set_val_i32(gguf_ctx, key.c_str(), v);
                break;
            }
            case GGUF_TYPE_UINT64: {
                uint64_t v = static_cast<uint64_t>(std::stoull(value));
                gguf_set_val_u64(gguf_ctx, key.c_str(), v);
                break;
            }
            case GGUF_TYPE_INT64: {
                int64_t v = std::stoll(value);
                gguf_set_val_i64(gguf_ctx, key.c_str(), v);
                break;
            }
            case GGUF_TYPE_FLOAT32: {
                float v = std::stof(value);
                gguf_set_val_f32(gguf_ctx, key.c_str(), v);
                break;
            }
            case GGUF_TYPE_BOOL: {
                bool v = (value == "true");
                gguf_set_val_bool(gguf_ctx, key.c_str(), v);
                break;
            }
            default:
                gguf_set_val_str(gguf_ctx, key.c_str(), value.c_str());
                break;
        }
    }

    // Add crystal-specific metadata
    gguf_set_val_str(gguf_ctx, "general.name", "crystal-quantized-ternary");
    gguf_set_val_str(gguf_ctx, "crystal.quantization_method", "absmean_b158");
    gguf_set_val_u32(gguf_ctx, "general.file_type", 34);

    // Allocate ggml context for tensor descriptors
    // We need enough space for tensor metadata (not data)
    size_t n_total_tensors = quantized_tensors.size() + model.tensors.size();
    size_t ctx_size = n_total_tensors * ggml_tensor_overhead() + ggml_graph_overhead();

    struct ggml_init_params ggml_params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context* ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) {
        std::cerr << "Failed to initialize ggml context\n";
        gguf_free(gguf_ctx);
        return false;
    }

    // Build set of quantized tensor names for quick lookup
    std::unordered_set<std::string> quantized_names;
    for (const auto& qt : quantized_tensors) {
        quantized_names.insert(qt.name);
    }

    // Track tensor data and order for writing
    struct TensorWriteInfo {
        const void* data;
        size_t size;
    };
    std::vector<TensorWriteInfo> write_order;

    // Add quantized tensors first
    for (const auto& qt : quantized_tensors) {
        // Store ternary data as a 1D blob of bytes
        size_t data_size = qt.groups.size() * sizeof(TernaryGroup);

        // Create tensor with the actual byte count as shape
        // Use F32 type but encode the real data size
        // (GGUF doesn't have a ternary type in standard ggml)
        int64_t ne = static_cast<int64_t>(data_size / sizeof(float));
        if (data_size % sizeof(float) != 0) {
            ne++;  // round up
        }

        struct ggml_tensor* tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, ne);
        if (!tensor) {
            std::cerr << "Failed to create tensor: " << qt.name << "\n";
            continue;
        }

        strncpy(tensor->name, qt.name.c_str(), GGML_MAX_NAME - 1);
        tensor->name[GGML_MAX_NAME - 1] = '\0';

        gguf_add_tensor(gguf_ctx, tensor);
        write_order.push_back({qt.groups.data(), data_size});
    }

    // Add non-quantized tensors (kept at F16 or original type)
    for (const auto& tensor : model.tensors) {
        if (quantized_names.count(tensor.name)) {
            continue;
        }

        // Create tensor with proper dimensions
        std::vector<int64_t> ne(tensor.shape.size());
        for (size_t d = 0; d < tensor.shape.size(); ++d) {
            ne[d] = static_cast<int64_t>(tensor.shape[d]);
        }

        // Keep as F16 for preserved layers
        enum ggml_type type = GGML_TYPE_F16;

        struct ggml_tensor* t = nullptr;
        if (ne.size() == 1) {
            t = ggml_new_tensor_1d(ggml_ctx, type, ne[0]);
        } else if (ne.size() == 2) {
            t = ggml_new_tensor_2d(ggml_ctx, type, ne[0], ne[1]);
        } else if (ne.size() == 3) {
            t = ggml_new_tensor_3d(ggml_ctx, type, ne[0], ne[1], ne[2]);
        } else {
            t = ggml_new_tensor_1d(ggml_ctx, type, static_cast<int64_t>(tensor.data.size()));
        }

        if (!t) {
            std::cerr << "Failed to create tensor: " << tensor.name << "\n";
            continue;
        }

        strncpy(t->name, tensor.name.c_str(), GGML_MAX_NAME - 1);
        t->name[GGML_MAX_NAME - 1] = '\0';

        gguf_add_tensor(gguf_ctx, t);

        // Convert F32 data to F16 for writing
        // Store the conversion temporarily
        write_order.push_back({tensor.data.data(), tensor.data.size() * sizeof(float)});
    }

    // Write GGUF header (metadata only)
    if (!gguf_write_to_file(gguf_ctx, output_path.string().c_str(), /*only_meta=*/true)) {
        std::cerr << "Failed to write GGUF header: " << output_path << "\n";
        ggml_free(ggml_ctx);
        gguf_free(gguf_ctx);
        return false;
    }

    // Append tensor data
    std::ofstream out_file(output_path.string(), std::ios::binary | std::ios::app);
    if (!out_file.is_open()) {
        std::cerr << "Failed to open output for tensor data: " << output_path << "\n";
        ggml_free(ggml_ctx);
        gguf_free(gguf_ctx);
        return false;
    }

    // Write quantized tensor data
    for (const auto& qt : quantized_tensors) {
        size_t data_size = qt.groups.size() * sizeof(TernaryGroup);
        out_file.write(reinterpret_cast<const char*>(qt.groups.data()), data_size);

        // Pad to F32 element boundary if needed
        size_t padded_size = ((data_size + sizeof(float) - 1) / sizeof(float)) * sizeof(float);
        if (padded_size > data_size) {
            std::vector<char> padding(padded_size - data_size, 0);
            out_file.write(padding.data(), padding.size());
        }
    }

    // Write non-quantized tensor data as F16
    for (const auto& tensor : model.tensors) {
        if (quantized_names.count(tensor.name)) {
            continue;
        }

        // Convert F32 to F16 and write
        size_t n_elements = tensor.data.size();
        std::vector<ggml_fp16_t> fp16_data(n_elements);
        ggml_fp32_to_fp16_row(tensor.data.data(), fp16_data.data(),
                              static_cast<int64_t>(n_elements));
        out_file.write(reinterpret_cast<const char*>(fp16_data.data()),
                      n_elements * sizeof(ggml_fp16_t));
    }

    out_file.close();
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);

    if (options.verbose) {
        std::cout << "Wrote quantized model to: " << output_path << "\n";
    }

    return true;
}

}  // namespace crystal
