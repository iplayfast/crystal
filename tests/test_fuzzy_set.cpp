#include "crystal/fuzzy/fuzzy_set.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace crystal;

void test_sorted_insertion() {
    FuzzySet<float> fs;
    fs.add_point(3.0f, 0.3f);
    fs.add_point(1.0f, 0.1f);
    fs.add_point(2.0f, 0.2f);
    
    const auto& points = fs.points();
    assert(points.size() == 3);
    assert(points[0].x == 1.0f);
    assert(points[1].x == 2.0f);
    assert(points[2].x == 3.0f);
    std::cout << "test_sorted_insertion PASSED\n";
}

void test_interpolation() {
    FuzzySet<float> fs;
    fs.add_point(0.0f, 0.0f);
    fs.add_point(50.0f, 0.5f);
    fs.add_point(100.0f, 1.0f);
    
    assert(std::abs(fs.evaluate(25.0f) - 0.25f) < 0.001f);
    assert(std::abs(fs.evaluate(75.0f) - 0.75f) < 0.001f);
    std::cout << "test_interpolation PASSED\n";
}

void test_and_operation() {
    FuzzySet<float> a;
    a.add_point(0.0f, 0.0f);
    a.add_point(50.0f, 0.5f);
    a.add_point(100.0f, 1.0f);
    
    FuzzySet<float> b;
    b.add_point(0.0f, 1.0f);
    b.add_point(50.0f, 0.5f);
    b.add_point(100.0f, 0.0f);
    
    auto c = a.and_with(b);
    assert(std::abs(c.evaluate(50.0f) - 0.5f) < 0.001f);
    std::cout << "test_and_operation PASSED\n";
}

void test_json_roundtrip() {
    FuzzySet<float> fs("test");
    fs.add_point(0.0f, 0.0f);
    fs.add_point(100.0f, 1.0f);
    
    auto json = fs.to_json();
    FuzzySet<float> fs2 = FuzzySet<float>::from_json(json);
    
    assert(fs.evaluate(50.0f) == fs2.evaluate(50.0f));
    std::cout << "test_json_roundtrip PASSED\n";
}

int main() {
    test_sorted_insertion();
    test_interpolation();
    test_and_operation();
    test_json_roundtrip();
    std::cout << "All fuzzy set tests PASSED\n";
    return 0;
}