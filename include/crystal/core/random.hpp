#pragma once

#include <random>
#include <type_traits>

namespace crystal {

/// Thread-local modern RNG wrapper
class Random {
public:
    /// Get the thread-local engine (seeded once per thread)
    static std::mt19937& engine() {
        thread_local std::mt19937 gen{std::random_device{}()};
        return gen;
    }

    /// Seed the thread-local engine explicitly
    static void seed(uint32_t s) {
        engine().seed(s);
    }

    /// Uniform real in [lo, hi)
    template <std::floating_point T>
    static T uniform(T lo, T hi) {
        std::uniform_real_distribution<T> dist(lo, hi);
        return dist(engine());
    }

    /// Uniform integer in [lo, hi]
    template <std::integral T>
    static T uniform_int(T lo, T hi) {
        std::uniform_int_distribution<T> dist(lo, hi);
        return dist(engine());
    }

    /// Normal distribution
    template <std::floating_point T>
    static T normal(T mean, T stddev) {
        std::normal_distribution<T> dist(mean, stddev);
        return dist(engine());
    }

    /// Xavier/Glorot initialization range for a layer
    template <std::floating_point T>
    static T xavier(size_t fan_in, size_t fan_out) {
        T limit = std::sqrt(static_cast<T>(6) / static_cast<T>(fan_in + fan_out));
        return uniform(-limit, limit);
    }
};

} // namespace crystal
