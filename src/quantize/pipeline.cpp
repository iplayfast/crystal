#include "crystal/quantize/pipeline.hpp"
#include "crystal/quantize/calibration.hpp"
#include "crystal/quantize/ensemble.hpp"
#include "crystal/quantize/gguf_writer.hpp"

#include <iostream>
#include <regex>

namespace crystal {

namespace {

bool should_keep_at_f16(const std::string& tensor_name, const std::string& regex_pattern) {
    try {
        std::regex re(regex_pattern, std::regex_constants::icase);
        return std::regex_search(tensor_name, re);
    } catch (...) {
        return false;
    }
}

}  // namespace

PipelineResult run_pipeline(const PipelineOptions& options) {
    PipelineResult result;

    if (options.input_models.empty()) {
        result.error_message = "No input models specified";
        return result;
    }

    if (options.output_path.empty()) {
        result.error_message = "No output path specified";
        return result;
    }

    // Step 1: Load all models
    std::vector<ModelTensors> models;
    for (const auto& path : options.input_models) {
        if (options.verbose) {
            std::cout << "Loading model: " << path << "\n";
        }

        auto model = read_model(path);

        if (model.tensors.empty()) {
            result.error_message = "Failed to load model: " + path;
            return result;
        }

        models.push_back(std::move(model));
    }

    if (options.verbose) {
        std::cout << "Loaded " << models.size() << " model(s)\n";
    }

    // Step 2: Ensemble if multiple models
    ModelTensors model;
    if (models.size() > 1) {
        if (options.verbose) {
            std::cout << "Ensembling " << models.size() << " models...\n";
        }
        model = ensemble_average(models);
    } else {
        model = std::move(models[0]);
    }

    if (options.verbose) {
        std::cout << "Model has " << model.tensors.size() << " tensors\n";
    }

    // Step 3: Compute importance matrix
    ImportanceMatrix imatrix;
    if (!options.no_calibrate && !options.dataset_path.empty()) {
        if (options.verbose) {
            std::cout << "Running calibration...\n";
        }
        imatrix = compute_importance(
            options.input_models[0],  // calibrate on first model
            options.dataset_path,
            options.num_chunks,
            options.verbose);
    }

    // Fall back to weight-magnitude importance if calibration didn't produce per-weight data
    if (imatrix.tensor_importance.empty()) {
        if (options.verbose) {
            std::cout << "Using weight-magnitude importance\n";
        }
        imatrix = compute_importance_from_weights(model);
    }

    // Step 4: Quantize
    if (options.verbose) {
        std::cout << "Quantizing model...\n";
    }

    QuantizeOptions qopts;
    qopts.keep_layers_regex = options.keep_layers_regex;
    qopts.use_importance = true;
    qopts.verbose = options.verbose;

    // Quantize with importance data
    QuantizationResult quant_result;

    for (const auto& tensor : model.tensors) {
        // Calculate original size
        size_t tensor_elements = 1;
        for (auto dim : tensor.shape) {
            tensor_elements *= dim;
        }
        quant_result.original_size_bytes += tensor_elements * 4;  // F32

        if (options.verbose) {
            std::cout << "Processing: " << tensor.name << " (";
            for (size_t i = 0; i < tensor.shape.size(); ++i) {
                if (i > 0) std::cout << "x";
                std::cout << tensor.shape[i];
            }
            std::cout << ")\n";
        }

        // Check if we should keep this at F16
        if (should_keep_at_f16(tensor.name, qopts.keep_layers_regex)) {
            if (options.verbose) {
                std::cout << "  Keeping at F16 (matched keep-layers regex)\n";
            }
            result.tensors_skipped++;
            continue;
        }

        // Get importance for this tensor
        std::span<const float> importance;
        auto it = imatrix.tensor_importance.find(tensor.name);
        if (it != imatrix.tensor_importance.end() && qopts.use_importance) {
            importance = it->second;
        }

        // Quantize to ternary
        auto quantized = quantize_tensor(
            tensor.name,
            tensor.data,
            importance,
            tensor.shape);

        quant_result.quantized_size_bytes += quantized.groups.size() * sizeof(TernaryGroup);
        quant_result.tensors.push_back(std::move(quantized));
    }

    if (quant_result.original_size_bytes > 0) {
        quant_result.compression_ratio = static_cast<double>(quant_result.quantized_size_bytes) /
                                         static_cast<double>(quant_result.original_size_bytes);
    }

    result.original_size_bytes = quant_result.original_size_bytes;
    result.quantized_size_bytes = quant_result.quantized_size_bytes;
    result.compression_ratio = quant_result.compression_ratio;
    result.tensors_quantized = static_cast<int>(quant_result.tensors.size());

    // Step 5: Write output GGUF
    if (options.verbose) {
        std::cout << "Writing output GGUF...\n";
    }

    bool write_ok = write_quantized_gguf(options.output_path, model, quant_result.tensors, qopts);

    if (!write_ok) {
        result.error_message = "Failed to write output GGUF file";
        return result;
    }

    result.success = true;

    if (options.verbose) {
        std::cout << "Quantization complete:\n";
        std::cout << "  Original size: " << result.original_size_bytes << " bytes\n";
        std::cout << "  Quantized size: " << result.quantized_size_bytes << " bytes\n";
        std::cout << "  Compression: " << (result.compression_ratio * 100.0) << "%\n";
        std::cout << "  Tensors quantized: " << result.tensors_quantized << "\n";
        std::cout << "  Tensors skipped (F16): " << result.tensors_skipped << "\n";
    }

    return result;
}

}  // namespace crystal
