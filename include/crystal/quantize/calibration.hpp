#pragma once

#include "crystal/quantize/model_reader.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace crystal {

struct ImportanceMatrix {
    // Per-tensor: vector of per-weight importance scores in [0, 1]
    std::map<std::string, std::vector<float>> tensor_importance;
};

/// Compute importance matrix by running calibration dataset through the model.
/// Uses llama.cpp to load the model, tokenize the dataset, and run inference.
/// Accumulates squared input activations per weight position.
ImportanceMatrix compute_importance(
    const std::filesystem::path& model_path,
    const std::filesystem::path& dataset_path,
    int num_chunks = 100,
    bool verbose = false);

/// Compute a weight-magnitude-based importance proxy (no inference needed).
/// importance[i] = |w[i]| / max(|w|) for each tensor.
ImportanceMatrix compute_importance_from_weights(const ModelTensors& model);

}  // namespace crystal
