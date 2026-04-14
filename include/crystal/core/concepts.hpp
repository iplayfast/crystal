#pragma once

#include <concepts>
#include <type_traits>

namespace crystal {

/// A type that can be used as a network weight
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/// Forward-declare TernaryWeight so concepts can reference it
struct TernaryWeight;

/// True if T is the special ternary quantized type
template <typename T>
constexpr bool is_ternary_v = std::is_same_v<T, TernaryWeight>;

/// A type usable as a network weight (arithmetic or ternary)
template <typename T>
concept NetworkWeight = Arithmetic<T> || is_ternary_v<T>;

/// A floating-point type for continuous math
template <typename T>
concept FloatingPoint = std::floating_point<T>;

/// An integer type (includes char)
template <typename T>
concept IntegerWeight = std::is_integral_v<T> && !std::is_same_v<T, bool>;

} // namespace crystal
