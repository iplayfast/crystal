#pragma once

/// Crystal: Modern C++20 computational intelligence library

// Core
#include "core/concepts.hpp"
#include "core/types.hpp"
#include "core/random.hpp"

// Fuzzy logic
#include "fuzzy/fuzzy_set.hpp"
#include "fuzzy/fuzzy_ops.hpp"

// Neural networks
#include "nn/activation.hpp"
#include "nn/quantization.hpp"
#include "nn/layer.hpp"
#include "nn/backprop.hpp"
#include "nn/blob_network.hpp"
#include "nn/training.hpp"

// I/O
#include "io/gguf_types.hpp"
#include "io/gguf.hpp"

// Math
#include "math/huge_int.hpp"

// Patterns
#include "patterns/strategy.hpp"
#include "patterns/observer.hpp"
#include "patterns/factory.hpp"
