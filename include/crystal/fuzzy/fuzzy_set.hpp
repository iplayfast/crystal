#pragma once

#include "crystal/core/concepts.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace crystal {

/// A point in a fuzzy set: (x, membership)
template <FloatingPoint T>
struct FuzzyPoint {
    T x{};
    T membership{};

    constexpr bool operator<(const FuzzyPoint& other) const { return x < other.x; }
    constexpr bool operator==(const FuzzyPoint& other) const = default;
};

/// FuzzySet<T> — sorted point table with linear interpolation and fuzzy operations
template <FloatingPoint T>
class FuzzySet {
public:
    using point_type = FuzzyPoint<T>;

    FuzzySet() = default;
    explicit FuzzySet(std::string name) : name_(std::move(name)) {}

    /// Add a point, maintaining sorted order by x
    void add_point(T x, T membership);

    /// Remove all points
    void clear();

    /// Number of points
    [[nodiscard]] size_t size() const { return points_.size(); }

    /// Check if empty
    [[nodiscard]] bool empty() const { return points_.empty(); }

    /// Get membership value at x via linear interpolation
    [[nodiscard]] T evaluate(T x) const;

    /// Fuzzy AND (intersection): min of memberships
    [[nodiscard]] FuzzySet and_with(const FuzzySet& other) const;

    /// Fuzzy OR (union): max of memberships
    [[nodiscard]] FuzzySet or_with(const FuzzySet& other) const;

    /// Fuzzy NOT (complement): 1 - membership
    [[nodiscard]] FuzzySet complement() const;

    /// Fuzzy XOR: (A OR B) AND NOT (A AND B)
    [[nodiscard]] FuzzySet xor_with(const FuzzySet& other) const;

    /// Normalize memberships to [0, 1]
    void normalize();

    /// Scale x-range to [new_min, new_max]
    void scale_range(T new_min, T new_max);

    /// Inhibit: clamp memberships to at most `ceiling`
    void inhibit(T ceiling);

    /// Name
    [[nodiscard]] const std::string& name() const { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    /// Access points
    [[nodiscard]] const std::vector<point_type>& points() const { return points_; }

    /// JSON serialization
    [[nodiscard]] nlohmann::json to_json() const;
    static FuzzySet from_json(const nlohmann::json& j);

private:
    std::string name_;
    std::vector<point_type> points_;
};

// Extern template declarations for common types
extern template class FuzzySet<float>;
extern template class FuzzySet<double>;

} // namespace crystal
