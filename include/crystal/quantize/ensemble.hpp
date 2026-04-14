#pragma once

#include "crystal/quantize/model_reader.hpp"

#include <span>
#include <vector>

namespace crystal {

ModelTensors ensemble_average(std::span<const ModelTensors> models);

}  // namespace crystal