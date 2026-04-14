#include "crystal/quantize/ternary_converter.hpp"
#include "crystal/quantize/model_reader.hpp"

#include <algorithm>
#include <cmath>
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

QuantizedTensor quantize_tensor(
    const std::string& name,
    std::span<const float> weights,
    std::span<const float> importance,
    const std::vector<uint64_t>& shape) {
    
    QuantizedTensor result;
    result.name = name;
    result.shape = shape;
    
    // Compute absmean scale
    float abs_sum = 0.0f;
    for (float w : weights) {
        abs_sum += std::abs(w);
    }
    float scale = abs_sum / static_cast<float>(weights.size());
    
    // Avoid division by zero
    if (scale < 1e-8f) {
        scale = 1.0f;
    }
    
    result.scale = scale;
    
    // Calculate number of groups
    size_t num_weights = weights.size();
    size_t num_groups = (num_weights + TernaryGroup::NUM_WEIGHTS - 1) / TernaryGroup::NUM_WEIGHTS;
    result.groups.resize(num_groups);
    
    // Quantize each group
    for (size_t g = 0; g < num_groups; ++g) {
        size_t start = g * TernaryGroup::NUM_WEIGHTS;
        size_t end = std::min(start + TernaryGroup::NUM_WEIGHTS, num_weights);
        
        TernaryWeight ternary_weights[TernaryGroup::NUM_WEIGHTS] = {};
        
        for (size_t i = start; i < end; ++i) {
            float w = weights[i];
            
            // Apply importance weighting if available
            if (!importance.empty()) {
                w *= (1.0f + importance[i]);
            }
            
            float scaled = w / scale;
            float rounded = std::round(scaled);
            float clamped = std::clamp(rounded, -1.0f, 1.0f);
            
            ternary_weights[i - start] = TernaryWeight(static_cast<int8_t>(static_cast<int>(clamped)));
        }
        
        result.groups[g].pack(ternary_weights, scale);
    }
    
    return result;
}

QuantizationResult quantize_model(
    const ModelTensors& model,
    const QuantizeOptions& options) {
    
    QuantizationResult result;
    
    // Calculate original size
    for (const auto& tensor : model.tensors) {
        size_t tensor_size = 1;
        for (auto dim : tensor.shape) {
            tensor_size *= dim;
        }
        result.original_size_bytes += tensor_size * 4;  // assume F32
    }
    
    // Process each tensor
    for (const auto& tensor : model.tensors) {
        if (options.verbose) {
            std::cout << "Processing: " << tensor.name << " (";
            for (size_t i = 0; i < tensor.shape.size(); ++i) {
                if (i > 0) std::cout << "x";
                std::cout << tensor.shape[i];
            }
            std::cout << ")\n";
        }
        
        // Check if we should keep this at F16
        bool keep_f16 = should_keep_at_f16(tensor.name, options.keep_layers_regex);
        
        if (keep_f16) {
            if (options.verbose) {
                std::cout << "  Keeping at F16 (matched keep-layers regex)\n";
            }
            // For now, just skip - we'll handle F16 separately
            continue;
        }
        
        // Quantize to ternary
        auto quantized = quantize_tensor(
            tensor.name,
            tensor.data,
            {},  // no importance yet
            tensor.shape
        );
        
        result.tensors.push_back(std::move(quantized));
        
        // Calculate quantized size
        result.quantized_size_bytes += result.tensors.back().groups.size() * sizeof(TernaryGroup);
    }
    
    if (result.original_size_bytes > 0) {
        result.compression_ratio = static_cast<double>(result.quantized_size_bytes) / 
                                   static_cast<double>(result.original_size_bytes);
    }
    
    return result;
}

}  // namespace crystal