#include "crystal/patterns/strategy.hpp"
#include "crystal/patterns/observer.hpp"
#include "crystal/patterns/factory.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace crystal;

void test_strategy() {
    int result = 0;
    Strategy<int&, int> add = [](int& r, int v) { r += v; };
    add(result, 5);
    assert(result == 5);
    add(result, 3);
    assert(result == 8);
    std::cout << "test_strategy PASSED\n";
}

void test_observer() {
    Observer<std::string> obs;
    std::vector<std::string> received;

    obs.subscribe([&received](const std::string& msg) { received.push_back(msg); });
    obs.subscribe([&received](const std::string& msg) { received.push_back("copy:" + msg); });

    obs.notify("hello");
    assert(received.size() == 2);
    assert(received[0] == "hello");
    assert(received[1] == "copy:hello");
    std::cout << "test_observer PASSED\n";
}

struct Shape {
    virtual ~Shape() = default;
    virtual std::string name() const = 0;
};

struct Circle : Shape {
    std::string name() const override { return "circle"; }
};

struct Square : Shape {
    std::string name() const override { return "square"; }
};

void test_factory() {
    Factory<Shape> factory;
    factory.register_type("circle", [] { return std::make_unique<Circle>(); });
    factory.register_type("square", [] { return std::make_unique<Square>(); });

    assert(factory.has_type("circle"));
    assert(!factory.has_type("triangle"));

    auto c = factory.create("circle");
    assert(c && c->name() == "circle");

    auto s = factory.create("square");
    assert(s && s->name() == "square");

    auto n = factory.create("triangle");
    assert(n == nullptr);

    std::cout << "test_factory PASSED\n";
}

int main() {
    test_strategy();
    test_observer();
    test_factory();
    std::cout << "All reflection/pattern tests PASSED\n";
    return 0;
}
