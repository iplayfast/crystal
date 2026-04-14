#pragma once

#include "crystal/quantize/model_reader.hpp"
#include "crystal/quantize/ternary_converter.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace crystal {

struct PipelineOptions {
    std::string output_path;
    std::string dataset_path;
    std::vector<std::string> input_models;
    int num_chunks = 100;
    std::string keep_layers_regex = "embed|output";
    bool no_calibrate = false;
    bool verbose = false;
};

struct PipelineResult {
    bool success = false;
    std::string error_message;
    size_t original_size_bytes = 0;
    size_t quantized_size_bytes = 0;
    double compression_ratio = 0.0;
    int tensors_quantized = 0;
    int tensors_skipped = 0;
};

PipelineResult run_pipeline(const PipelineOptions& options);

}  // namespace crystal