#include "crystal/quantize/model_reader.hpp"
#include "crystal/io/gguf_types.hpp"

#include <ggml.h>
#include <gguf.h>

#include <cstring>
#include <iostream>

namespace crystal {

namespace {

void dequantize_to_f32(const struct ggml_tensor* tensor, std::vector<float>& output) {
    int64_t n_elements = ggml_nelements(tensor);
    output.resize(n_elements);

    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(output.data(), tensor->data, n_elements * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(
            static_cast<const ggml_fp16_t*>(tensor->data),
            output.data(), n_elements);
    } else {
        // Use ggml's type traits for block-quantized types (Q4_0, Q8_0, etc.)
        const struct ggml_type_traits* traits = ggml_get_type_traits(tensor->type);
        if (traits && traits->to_float) {
            traits->to_float(tensor->data, output.data(), n_elements);
        } else {
            std::cerr << "Warning: no dequantizer for tensor '" << tensor->name
                      << "' (type " << tensor->type << "), zeroing\n";
            std::fill(output.begin(), output.end(), 0.0f);
        }
    }
}

}  // namespace

ModelTensors read_model(const std::filesystem::path& gguf_path) {
    ModelTensors result;
    result.path = gguf_path.string();

    // Open GGUF with tensor data allocated into a ggml context
    struct ggml_context* ggml_ctx = nullptr;
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx      = &ggml_ctx,
    };
    struct gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.string().c_str(), params);
    if (!gguf_ctx) {
        std::cerr << "Failed to read GGUF file: " << gguf_path << "\n";
        return result;
    }

    std::cerr << "GGUF version: " << gguf_get_version(gguf_ctx) << "\n";

    // Read string metadata
    const int64_t n_meta = gguf_get_n_kv(gguf_ctx);
    for (int64_t i = 0; i < n_meta; ++i) {
        if (gguf_get_kv_type(gguf_ctx, i) == GGUF_TYPE_STRING) {
            const char* key = gguf_get_key(gguf_ctx, i);
            const char* value = gguf_get_val_str(gguf_ctx, i);
            if (key && value) {
                result.metadata[key] = value;
            }
        }
    }

    // Iterate tensors using ggml context (has proper data and dimensions)
    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    std::cerr << "Tensor count: " << n_tensors << "\n";

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        if (!name) continue;

        // Look up the tensor in the ggml context to get real data + dimensions
        struct ggml_tensor* tensor = ggml_get_tensor(ggml_ctx, name);
        if (!tensor) {
            std::cerr << "Warning: tensor '" << name << "' not found in ggml context\n";
            continue;
        }

        TensorData td;
        td.name = name;
        td.type = static_cast<GGUFQuantType>(gguf_get_tensor_type(gguf_ctx, i));

        // Get real dimensions from ggml tensor
        int n_dims = ggml_n_dims(tensor);
        for (int d = 0; d < n_dims; ++d) {
            td.shape.push_back(static_cast<uint64_t>(tensor->ne[d]));
        }

        // Dequantize tensor data to F32
        dequantize_to_f32(tensor, td.data);

        result.tensors.push_back(std::move(td));
    }

    std::cerr << "Read " << result.tensors.size() << " tensors from " << gguf_path << "\n";

    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);

    return result;
}

}  // namespace crystal
