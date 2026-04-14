#pragma once

#include <functional>

namespace crystal {

template <typename... Args>
using Strategy = std::function<void(Args...)>;

}