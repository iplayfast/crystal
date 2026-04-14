#include "crystal/quantize/ensemble.hpp"
#include "crystal/quantize/model_reader.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace crystal {

ModelTensors ensemble_average(std::span<const ModelTensors> models) {
    if (models.empty()) {
        return {};
    }
    
    if (models.size() == 1) {
        return models[0];
    }
    
    ModelTensors result;
    result.path = "ensemble";
    result.metadata = models[0].metadata;
    
    // Build tensor index by name
    std::unordered_map<std::string, size_t> tensor_idx;
    for (size_t i = 0; i < models[0].tensors.size(); ++i) {
        tensor_idx[models[0].tensors[i].name] = i;
    }
    
    // Average each tensor
    for (const auto& tensor : models[0].tensors) {
        const std::string& name = tensor.name;
        
        // Check if all models have this tensor
        bool all_have = true;
        for (size_t m = 1; m < models.size() && all_have; ++m) {
            auto it = std::find_if(models[m].tensors.begin(), models[m].tensors.end(),
                [&name](const auto& t) { return t.name == name; });
            if (it == models[m].tensors.end()) {
                all_have = false;
            }
        }
        
        if (!all_have) {
            std::cout << "Warning: tensor " << name << " not found in all models, skipping\n";
            continue;
        }
        
        // Get tensors from all models
        std::span<const float> first_data = tensor.data;
        
        // Calculate average
        TensorData averaged;
        averaged.name = name;
        averaged.shape = tensor.shape;
        averaged.type = tensor.type;
        averaged.data.resize(first_data.size());
        
        for (size_t i = 0; i < first_data.size(); ++i) {
            float sum = 0.0f;
            for (size_t m = 0; m < models.size(); ++m) {
                auto it = std::find_if(models[m].tensors.begin(), models[m].tensors.end(),
                    [&name](const auto& t) { return t.name == name; });
                sum += it->data[i];
            }
            averaged.data[i] = sum / static_cast<float>(models.size());
        }
        
        result.tensors.push_back(std::move(averaged));
    }
    
    std::cout << "Ensemble averaged " << result.tensors.size() << " tensors from " 
              << models.size() << " models\n";
    
    return result;
}

}  // namespace crystal