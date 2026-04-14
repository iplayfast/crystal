#pragma once

#include <functional>
#include <vector>

namespace crystal {

template <typename Event>
class Observer {
public:
    using callback_type = std::function<void(const Event&)>;

    void subscribe(callback_type callback) {
        callbacks_.push_back(std::move(callback));
    }

    void notify(const Event& event) const {
        for (const auto& cb : callbacks_) {
            cb(event);
        }
    }

private:
    std::vector<callback_type> callbacks_;
};

}