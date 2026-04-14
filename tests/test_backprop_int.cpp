#include "crystal/nn/backprop.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace crystal;

void test_int_network_creation() {
    BackpropNetwork<int> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    assert(net.num_layers() == 3);
    auto sizes = net.layer_sizes();
    assert(sizes[0] == 2);
    assert(sizes[1] == 4);
    assert(sizes[2] == 1);
    std::cout << "test_int_network_creation PASSED\n";
}

void test_int_forward() {
    Random::seed(42);

    BackpropNetwork<int> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    // Integer inputs
    std::vector<int64_t> input = {100, 50};
    net.forward(std::span<const int64_t>(input));

    auto out = net.output();
    assert(out.size() == 1);
    // Sigmoid on integer produces 0 or 1 (since Sigmoid<int64_t> casts from double)
    std::cout << "  Int forward output: " << out[0] << "\n";
    std::cout << "test_int_forward PASSED\n";
}

void test_int_json_roundtrip() {
    Random::seed(42);

    BackpropNetwork<int> net;
    net.add_layer(2);
    net.add_layer(3);
    net.add_layer(1);
    net.randomize_weights();

    auto json = net.to_json();
    auto net2 = BackpropNetwork<int>::from_json(json);

    assert(net2.num_layers() == 3);
    std::cout << "test_int_json_roundtrip PASSED\n";
}

int main() {
    test_int_network_creation();
    test_int_forward();
    test_int_json_roundtrip();
    std::cout << "All int backprop tests PASSED\n";
    return 0;
}
