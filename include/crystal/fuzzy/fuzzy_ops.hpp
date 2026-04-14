#pragma once

#include "crystal/fuzzy/fuzzy_set.hpp"

namespace crystal {

/// Convenience operators for fuzzy sets
template <FloatingPoint T>
FuzzySet<T> operator&(const FuzzySet<T>& a, const FuzzySet<T>& b) {
    return a.and_with(b);
}

template <FloatingPoint T>
FuzzySet<T> operator|(const FuzzySet<T>& a, const FuzzySet<T>& b) {
    return a.or_with(b);
}

template <FloatingPoint T>
FuzzySet<T> operator^(const FuzzySet<T>& a, const FuzzySet<T>& b) {
    return a.xor_with(b);
}

template <FloatingPoint T>
FuzzySet<T> operator~(const FuzzySet<T>& a) {
    return a.complement();
}

} // namespace crystal
