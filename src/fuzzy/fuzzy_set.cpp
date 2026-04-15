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

template <FloatingPoint T>
void FuzzySet<T>::optimize(T tolerance) {
    if (points_.size() < 3) return;
    std::vector<point_type> optimized;
    optimized.push_back(points_.front());
    for (size_t i = 1; i + 1 < points_.size(); ++i) {
        auto& prev = optimized.back();
        auto& curr = points_[i];
        auto& next = points_[i + 1];
        // Check if curr lies on the line from prev to next
        T range = next.x - prev.x;
        if (range == T{0}) continue;
        T t = (curr.x - prev.x) / range;
        T expected = prev.membership + t * (next.membership - prev.membership);
        if (std::abs(curr.membership - expected) > tolerance) {
            optimized.push_back(curr);
        }
    }
    optimized.push_back(points_.back());
    points_ = std::move(optimized);
}

template <FloatingPoint T>
void FuzzySet<T>::increase_samples(int factor) {
    if (points_.size() < 2 || factor < 2) return;
    std::vector<point_type> expanded;
    expanded.reserve((points_.size() - 1) * factor + 1);
    for (size_t i = 0; i + 1 < points_.size(); ++i) {
        expanded.push_back(points_[i]);
        for (int j = 1; j < factor; ++j) {
            T t = static_cast<T>(j) / static_cast<T>(factor);
            T x = points_[i].x + t * (points_[i + 1].x - points_[i].x);
            T m = points_[i].membership + t * (points_[i + 1].membership - points_[i].membership);
            expanded.push_back({x, m});
        }
    }
    expanded.push_back(points_.back());
    points_ = std::move(expanded);
}

template <FloatingPoint T>
T FuzzySet<T>::equality(const FuzzySet& other, int samples) const {
    if (points_.empty() && other.points_.empty()) return T{1};
    if (points_.empty() || other.points_.empty()) return T{0};

    T x_min = std::min(points_.front().x, other.points_.front().x);
    T x_max = std::max(points_.back().x, other.points_.back().x);
    if (x_max == x_min) {
        return T{1} - std::abs(evaluate(x_min) - other.evaluate(x_min));
    }

    T sum_diff = T{0};
    for (int i = 0; i <= samples; ++i) {
        T x = x_min + (x_max - x_min) * static_cast<T>(i) / static_cast<T>(samples);
        sum_diff += std::abs(evaluate(x) - other.evaluate(x));
    }
    T avg_diff = sum_diff / static_cast<T>(samples + 1);
    return T{1} - avg_diff;
}

// Explicit instantiations
template class FuzzySet<float>;
template class FuzzySet<double>;

} // namespace crystal
