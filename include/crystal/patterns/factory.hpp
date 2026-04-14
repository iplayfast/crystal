#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace crystal {

/// Simple factory using registered creator functions
template <typename Base>
class Factory {
public:
    using CreatorFunc = std::function<std::unique_ptr<Base>()>;

    void register_type(const std::string& name, CreatorFunc creator) {
        creators_[name] = std::move(creator);
    }

    [[nodiscard]] std::unique_ptr<Base> create(const std::string& name) const {
        auto it = creators_.find(name);
        if (it != creators_.end()) {
            return it->second();
        }
        return nullptr;
    }

    [[nodiscard]] bool has_type(const std::string& name) const {
        return creators_.contains(name);
    }

private:
    std::unordered_map<std::string, CreatorFunc> creators_;
};

} // namespace crystal
