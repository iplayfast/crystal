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

void test_optimize() {
    FuzzySet<float> fs;
    // Three collinear points: (0,0), (50,0.5), (100,1.0)
    fs.add_point(0.0f, 0.0f);
    fs.add_point(50.0f, 0.5f);
    fs.add_point(100.0f, 1.0f);

    fs.optimize(0.01f);
    // Middle point is collinear — should be removed
    assert(fs.size() == 2);
    assert(fs.points()[0].x == 0.0f);
    assert(fs.points()[1].x == 100.0f);

    // Non-collinear points should be kept
    FuzzySet<float> fs2;
    fs2.add_point(0.0f, 0.0f);
    fs2.add_point(50.0f, 1.0f);
    fs2.add_point(100.0f, 0.0f);
    fs2.optimize(0.01f);
    assert(fs2.size() == 3);
    std::cout << "test_optimize PASSED\n";
}

void test_increase_samples() {
    FuzzySet<float> fs;
    fs.add_point(0.0f, 0.0f);
    fs.add_point(100.0f, 1.0f);

    fs.increase_samples(2);
    assert(fs.size() == 3);
    assert(std::abs(fs.points()[1].x - 50.0f) < 0.001f);
    assert(std::abs(fs.points()[1].membership - 0.5f) < 0.001f);
    std::cout << "test_increase_samples PASSED\n";
}

void test_equality() {
    FuzzySet<float> a;
    a.add_point(0.0f, 0.0f);
    a.add_point(100.0f, 1.0f);

    FuzzySet<float> b;
    b.add_point(0.0f, 0.0f);
    b.add_point(100.0f, 1.0f);

    float eq = a.equality(b);
    assert(std::abs(eq - 1.0f) < 0.001f);

    // Different sets should have lower equality
    FuzzySet<float> c;
    c.add_point(0.0f, 1.0f);
    c.add_point(100.0f, 0.0f);

    float eq2 = a.equality(c);
    assert(eq2 < 0.6f); // opposite slopes
    std::cout << "test_equality PASSED\n";
}

void test_indexed_access() {
    FuzzySet<float> fs;
    fs.add_point(10.0f, 0.3f);
    fs.add_point(20.0f, 0.7f);

    assert(fs.x_at(0) == 10.0f);
    assert(fs.membership_at(0) == 0.3f);
    assert(fs.x_at(1) == 20.0f);
    assert(fs.membership_at(1) == 0.7f);
    std::cout << "test_indexed_access PASSED\n";
}

int main() {
    test_sorted_insertion();
    test_interpolation();
    test_and_operation();
    test_json_roundtrip();
    test_optimize();
    test_increase_samples();
    test_equality();
    test_indexed_access();
    std::cout << "All fuzzy set tests PASSED\n";
    return 0;
}