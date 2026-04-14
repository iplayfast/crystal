#include "crystal/quantize/pipeline.hpp"
#include <iostream>
#include <cstring>

void print_help(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options] output.gguf dataset.txt model1.gguf [model2.gguf ...]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --chunks N         Number of calibration chunks (default: 100)\n";
    std::cout << "  --keep-layers RE   Regex for layers to keep at F16 (default: \"embed|output\")\n";
    std::cout << "  --no-calibrate     Skip calibration, use weight-only statistics\n";
    std::cout << "  --verbose          Print per-layer quantization stats\n";
    std::cout << "  --help             Show this help\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }
    
    crystal::PipelineOptions options;
    int i = 1;
    
    while (i < argc) {
        if (strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            options.verbose = true;
        } else if (strcmp(argv[i], "--no-calibrate") == 0) {
            options.no_calibrate = true;
        } else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) {
            options.num_chunks = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--keep-layers") == 0 && i + 1 < argc) {
            options.keep_layers_regex = argv[++i];
        } else if (argv[i][0] != '-') {
            // Non-option argument
            if (options.output_path.empty()) {
                options.output_path = argv[i];
            } else if (options.dataset_path.empty()) {
                options.dataset_path = argv[i];
            } else {
                options.input_models.push_back(argv[i]);
            }
        }
        ++i;
    }
    
    if (options.input_models.empty()) {
        std::cerr << "Error: No input models specified\n\n";
        print_help(argv[0]);
        return 1;
    }
    
    auto result = crystal::run_pipeline(options);
    
    if (!result.success) {
        std::cerr << "Error: " << result.error_message << "\n";
        return 1;
    }
    
    std::cout << "\nSummary:\n";
    std::cout << "  Original size: " << result.original_size_bytes << " bytes\n";
    std::cout << "  Quantized size: " << result.quantized_size_bytes << " bytes\n";
    std::cout << "  Compression: " << (result.compression_ratio * 100.0) << "%\n";
    std::cout << "  Tensors quantized: " << result.tensors_quantized << "\n";
    std::cout << "  Tensors skipped: " << result.tensors_skipped << "\n";
    
    return 0;
}