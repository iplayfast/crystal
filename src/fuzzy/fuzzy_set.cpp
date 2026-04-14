#include "crystal/fuzzy/fuzzy_set.hpp"

namespace crystal {

template <FloatingPoint T>
void FuzzySet<T>::add_point(T x, T membership) {
    point_type pt{x, membership};
    auto it = std::lower_bound(points_.begin(), points_.end(), pt);
    // If a point with same x exists, update its membership
    if (it != points_.end() && it->x == x) {
        it->membership = membership;
    } else {
        points_.insert(it, pt);
    }
}

template <FloatingPoint T>
void FuzzySet<T>::clear() {
    points_.clear();
}

template <FloatingPoint T>
T FuzzySet<T>::evaluate(T x) const {
    if (points_.empty()) return T{0};
    if (points_.size() == 1) return points_[0].membership;

    // Below first point
    if (x <= points_.front().x) return points_.front().membership;
    // Above last point
    if (x >= points_.back().x) return points_.back().membership;

    // Binary search for interval
    point_type key{x, T{0}};
    auto it = std::lower_bound(points_.begin(), points_.end(), key);

    if (it == points_.begin()) return it->membership;

    auto prev = std::prev(it);
    // Linear interpolation
    T range = it->x - prev->x;
    if (range == T{0}) return prev->membership;
    T t = (x - prev->x) / range;
    return prev->membership + t * (it->membership - prev->membership);
}

template <FloatingPoint T>
FuzzySet<T> FuzzySet<T>::and_with(const FuzzySet& other) const {
    FuzzySet result;
    // Collect all x values from both sets
    std::vector<T> xs;
    xs.reserve(points_.size() + other.points_.size());
    for (auto& p : points_) xs.push_back(p.x);
    for (auto& p : other.points_) xs.push_back(p.x);
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    for (T x : xs) {
        result.add_point(x, std::min(evaluate(x), other.evaluate(x)));
    }
    return result;
}

template <FloatingPoint T>
FuzzySet<T> FuzzySet<T>::or_with(const FuzzySet& other) const {
    FuzzySet result;
    std::vector<T> xs;
    xs.reserve(points_.size() + other.points_.size());
    for (auto& p : points_) xs.push_back(p.x);
    for (auto& p : other.points_) xs.push_back(p.x);
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    for (T x : xs) {
        result.add_point(x, std::max(evaluate(x), other.evaluate(x)));
    }
    return result;
}

template <FloatingPoint T>
FuzzySet<T> FuzzySet<T>::complement() const {
    FuzzySet result;
    for (auto& p : points_) {
        result.add_point(p.x, T{1} - p.membership);
    }
    return result;
}

template <FloatingPoint T>
FuzzySet<T> FuzzySet<T>::xor_with(const FuzzySet& other) const {
    return or_with(other).and_with(and_with(other).complement());
}

template <FloatingPoint T>
void FuzzySet<T>::normalize() {
    if (points_.empty()) return;
    T max_val = T{0};
    for (auto& p : points_) {
        max_val = std::max(max_val, std::abs(p.membership));
    }
    if (max_val > T{0}) {
        for (auto& p : points_) {
            p.membership /= max_val;
        }
    }
}

template <FloatingPoint T>
void FuzzySet<T>::scale_range(T new_min, T new_max) {
    if (points_.size() < 2) return;
    T old_min = points_.front().x;
    T old_max = points_.back().x;
    T old_range = old_max - old_min;
    if (old_range == T{0}) return;
    T new_range = new_max - new_min;

    for (auto& p : points_) {
        p.x = new_min + (p.x - old_min) / old_range * new_range;
    }
}

template <FloatingPoint T>
void FuzzySet<T>::inhibit(T ceiling) {
    for (auto& p : points_) {
        p.membership = std::min(p.membership, ceiling);
    }
}

template <FloatingPoint T>
nlohmann::json FuzzySet<T>::to_json() const {
    nlohmann::json j;
    j["name"] = name_;
    j["points"] = nlohmann::json::array();
    for (auto& p : points_) {
        j["points"].push_back({{"x", p.x}, {"m", p.membership}});
    }
    return j;
}

template <FloatingPoint T>
FuzzySet<T> FuzzySet<T>::from_json(const nlohmann::json& j) {
    FuzzySet fs(j.value("name", ""));
    for (auto& pt : j.at("points")) {
        fs.add_point(pt.at("x").get<T>(), pt.at("m").get<T>());
    }
    return fs;
}

// Explicit instantiations
template class FuzzySet<float>;
template class FuzzySet<double>;

} // namespace crystal
