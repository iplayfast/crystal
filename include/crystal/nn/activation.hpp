#pragma once

#include "crystal/core/concepts.hpp"

#include <algorithm>
#include <cmath>

namespace crystal {

/// Sigmoid activation function
template <typename T>
struct Sigmoid {
    static T forward(T x) {
        if constexpr (std::is_floating_point_v<T>) {
            return T{1} / (T{1} + std::exp(-x));
        } else {
            // Integer approximation: scale to fixed point
            double fx = 1.0 / (1.0 + std::exp(-static_cast<double>(x)));
            return static_cast<T>(fx);
        }
    }

    static T derivative(T output) {
        if constexpr (std::is_floating_point_v<T>) {
            return output * (T{1} - output);
        } else {
            double o = static_cast<double>(output);
            return static_cast<T>(o * (1.0 - o));
        }
    }
};

/// ReLU activation function
template <typename T>
struct ReLU {
    static T forward(T x) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::max(T{0}, x);
        } else {
            return x > T{0} ? x : T{0};
        }
    }

    static T derivative(T output) {
        return output > T{0} ? T{1} : T{0};
    }
};

} // namespace crystal
