#include "crystal/io/gguf.hpp"
#include "crystal/nn/backprop.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace crystal;

void test_metadata_roundtrip() {
    GGUFFile file;
    file.set_metadata("test.string", std::string("hello"));
    file.set_metadata("test.uint32", uint32_t(42));
    file.set_metadata("test.float", 3.14f);
    file.set_metadata("test.bool", true);

    std::filesystem::path path = "/tmp/crystal_test_meta.gguf";
    file.write(path);

    auto loaded = GGUFFile::read(path);

    auto* s = loaded.get_metadata("test.string");
    assert(s && std::get<std::string>(*s) == "hello");

    auto* u = loaded.get_metadata("test.uint32");
    assert(u && std::get<uint32_t>(*u) == 42);

    auto* f = loaded.get_metadata("test.float");
    assert(f && std::abs(std::get<float>(*f) - 3.14f) < 1e-5f);

    auto* b = loaded.get_metadata("test.bool");
    assert(b && std::get<bool>(*b) == true);

    std::filesystem::remove(path);
    std::cout << "test_metadata_roundtrip PASSED\n";
}

void test_f32_tensor_roundtrip() {
    GGUFFile file;
    file.set_metadata("test", std::string("f32_tensor"));

    GGUFTensor t;
    t.name = "weights";
    t.dimensions = {2, 3};
    t.type = GGUFQuantType::F32;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t.data.resize(data.size() * sizeof(float));
    std::memcpy(t.data.data(), data.data(), t.data.size());
    file.add_tensor(std::move(t));

    std::filesystem::path path = "/tmp/crystal_test_f32.gguf";
    file.write(path);

    auto loaded = GGUFFile::read(path);
    auto* tensor = loaded.get_tensor("weights");
    assert(tensor != nullptr);
    assert(tensor->dimensions.size() == 2);
    assert(tensor->dimensions[0] == 2);
    assert(tensor->dimensions[1] == 3);
    assert(tensor->type == GGUFQuantType::F32);

    for (size_t i = 0; i < 6; ++i) {
        float v;
        std::memcpy(&v, tensor->data.data() + i * sizeof(float), sizeof(float));
        assert(std::abs(v - data[i]) < 1e-5f);
    }

    std::filesystem::remove(path);
    std::cout << "test_f32_tensor_roundtrip PASSED\n";
}

void test_magic_validation() {
    // Write garbage and try to read
    std::filesystem::path path = "/tmp/crystal_test_bad.gguf";
    {
        std::ofstream ofs(path, std::ios::binary);
        uint32_t bad_magic = 0xDEADBEEF;
        ofs.write(reinterpret_cast<const char*>(&bad_magic), 4);
    }

    bool caught = false;
    try {
        GGUFFile::read(path);
    } catch (const std::runtime_error& e) {
        caught = true;
        std::cout << "  Caught expected error: " << e.what() << "\n";
    }
    assert(caught);

    std::filesystem::remove(path);
    std::cout << "test_magic_validation PASSED\n";
}

void test_double_network_gguf_roundtrip() {
    Random::seed(42);

    BackpropNetwork<double> net;
    net.add_layer(2);
    net.add_layer(4);
    net.add_layer(1);
    net.randomize_weights();

    // Train briefly
    std::vector<double> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<double> targets = {0, 1, 1, 0};
    TrainingConfig config;
    config.max_epochs = 500;
    config.learning_rate = 0.5;
    net.train(inputs, targets, 4, config);

    // Export to GGUF
    auto gguf = GGUFFile::from_network(net);
    std::filesystem::path path = "/tmp/crystal_test_net_double.gguf";
    gguf.write(path);

    // Import back
    auto loaded = GGUFFile::read(path);
    auto net2 = loaded.to_network<double>();

    // Compare outputs
    for (int i = 0; i < 4; ++i) {
        std::span<const double> in(inputs.data() + i * 2, 2);
        net.forward(in);
        double out1 = net.output()[0];
        net2.forward(in);
        double out2 = net2.output()[0];
        // F32 storage loses some precision from double
        assert(std::abs(out1 - out2) < 0.01);
    }

    std::filesystem::remove(path);
    std::cout << "test_double_network_gguf_roundtrip PASSED\n";
}

void test_ternary_network_gguf_roundtrip() {
    Random::seed(42);

    BackpropNetwork<TernaryWeight> net;
    net.add_layer(2);
    net.add_layer(8);
    net.add_layer(1);
    net.randomize_weights();

    // Train briefly
    std::vector<float> inputs = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<float> targets = {0, 1, 1, 0};
    TrainingConfig config;
    config.max_epochs = 1000;
    config.learning_rate = 0.5;
    net.train(inputs, targets, 4, config);

    // Export to GGUF
    auto gguf = GGUFFile::from_network(net);
    std::filesystem::path path = "/tmp/crystal_test_net_ternary.gguf";
    gguf.write(path);

    // Import back
    auto loaded = GGUFFile::read(path);
    auto net2 = loaded.to_network<TernaryWeight>();

    // Verify ternary weights match
    for (size_t l = 1; l < net.layers().size(); ++l) {
        for (size_t w = 0; w < net.layers()[l].weights.size(); ++w) {
            assert(net.layers()[l].weights[w].value == net2.layers()[l].weights[w].value);
        }
    }

    // Compare outputs
    for (int i = 0; i < 4; ++i) {
        std::span<const float> in(inputs.data() + i * 2, 2);
        net.forward(in);
        float out1 = net.output()[0];
        net2.forward(in);
        float out2 = net2.output()[0];
        assert(std::abs(out1 - out2) < 0.01f);
    }

    std::filesystem::remove(path);
    std::cout << "test_ternary_network_gguf_roundtrip PASSED\n";
}

int main() {
    test_metadata_roundtrip();
    test_f32_tensor_roundtrip();
    test_magic_validation();
    test_double_network_gguf_roundtrip();
    test_ternary_network_gguf_roundtrip();
    std::cout << "All GGUF I/O tests PASSED\n";
    return 0;
}
