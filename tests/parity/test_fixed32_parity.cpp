/**
 * @file test_fixed32_parity.cpp
 * @brief Parity test: Fixed32 arithmetic vs float reference within tolerance.
 *
 * Verifies that Fixed32 maintains deterministic results and matches
 * floating-point reference values within the expected precision bounds.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lpl/core/Log.hpp>
#include <lpl/math/FixedPoint.hpp>

using namespace lpl;

static int failures = 0;

static void check(const char *label, float expected, float actual, float tolerance)
{
    const float diff = std::fabs(expected - actual);
    if (diff > tolerance)
    {
        std::printf("  FAIL: %-40s expected=%.6f actual=%.6f diff=%.6f\n", label, expected, actual, diff);
        ++failures;
    }
    else
    {
        std::printf("  PASS: %-40s (diff=%.6f)\n", label, diff);
    }
}

int main()
{
    core::Log::info("=== Fixed32 Parity Test ===");

    // Addition
    {
        auto a = math::Fixed32::fromFloat(3.14f);
        auto b = math::Fixed32::fromFloat(2.71f);
        auto c = a + b;
        check("3.14 + 2.71", 5.85f, c.toFloat(), 0.001f);
    }

    // Subtraction
    {
        auto a = math::Fixed32::fromFloat(10.0f);
        auto b = math::Fixed32::fromFloat(3.5f);
        auto c = a - b;
        check("10.0 - 3.5", 6.5f, c.toFloat(), 0.001f);
    }

    // Multiplication
    {
        auto a = math::Fixed32::fromFloat(2.5f);
        auto b = math::Fixed32::fromFloat(4.0f);
        auto c = a * b;
        check("2.5 * 4.0", 10.0f, c.toFloat(), 0.001f);
    }

    // Division
    {
        auto a = math::Fixed32::fromFloat(10.0f);
        auto b = math::Fixed32::fromFloat(3.0f);
        auto c = a / b;
        check("10.0 / 3.0", 3.333f, c.toFloat(), 0.01f);
    }

    // Negative values
    {
        auto a = math::Fixed32::fromFloat(-5.0f);
        auto b = math::Fixed32::fromFloat(2.0f);
        auto c = a + b;
        check("-5.0 + 2.0", -3.0f, c.toFloat(), 0.001f);
    }

    // Determinism: same operation repeated produces identical results
    {
        auto a = math::Fixed32::fromFloat(7.777f);
        auto b = math::Fixed32::fromFloat(3.333f);
        auto r1 = (a * b) + (a - b);
        auto r2 = (a * b) + (a - b);
        if (r1.raw() == r2.raw())
        {
            std::printf("  PASS: Determinism (identical raw values)\n");
        }
        else
        {
            std::printf("  FAIL: Determinism (raw1=%d raw2=%d)\n", r1.raw(), r2.raw());
            ++failures;
        }
    }

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
