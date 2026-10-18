#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lpl/core/Log.hpp>
#include <lpl/math/FixedPoint.hpp>

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

/**
 * @brief Addition: a Fixed32 sum matches the float reference.
 */
static void checkAdditionMatchesFloat()
{
    auto a = lpl::math::Fixed32::fromFloat(3.14f);
    auto b = lpl::math::Fixed32::fromFloat(2.71f);
    auto c = a + b;
    check("3.14 + 2.71", 5.85f, c.toFloat(), 0.001f);
}

/**
 * @brief Subtraction: a Fixed32 difference matches the float reference.
 */
static void checkSubtractionMatchesFloat()
{
    auto a = lpl::math::Fixed32::fromFloat(10.0f);
    auto b = lpl::math::Fixed32::fromFloat(3.5f);
    auto c = a - b;
    check("10.0 - 3.5", 6.5f, c.toFloat(), 0.001f);
}

/**
 * @brief Multiplication: a Fixed32 product matches the float reference.
 */
static void checkMultiplicationMatchesFloat()
{
    auto a = lpl::math::Fixed32::fromFloat(2.5f);
    auto b = lpl::math::Fixed32::fromFloat(4.0f);
    auto c = a * b;
    check("2.5 * 4.0", 10.0f, c.toFloat(), 0.001f);
}

/**
 * @brief Division: a Fixed32 quotient matches the float reference.
 */
static void checkDivisionMatchesFloat()
{
    auto a = lpl::math::Fixed32::fromFloat(10.0f);
    auto b = lpl::math::Fixed32::fromFloat(3.0f);
    auto c = a / b;
    check("10.0 / 3.0", 3.333f, c.toFloat(), 0.01f);
}

/**
 * @brief Negative values: a Fixed32 sum with a negative operand matches the float reference.
 */
static void checkNegativeValuesMatchFloat()
{
    auto a = lpl::math::Fixed32::fromFloat(-5.0f);
    auto b = lpl::math::Fixed32::fromFloat(2.0f);
    auto c = a + b;
    check("-5.0 + 2.0", -3.0f, c.toFloat(), 0.001f);
}

/**
 * @brief Determinism: the same operation repeated produces identical results.
 */
static void checkRepeatedOperationIsDeterministic()
{
    auto a = lpl::math::Fixed32::fromFloat(7.777f);
    auto b = lpl::math::Fixed32::fromFloat(3.333f);
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

/**
 * @brief Parity test: Fixed32 arithmetic vs float reference within tolerance.
 *
 * Verifies that Fixed32 maintains deterministic results and matches
 * floating-point reference values within the expected precision bounds.
 */
int main()
{
    lpl::core::Log::info("=== Fixed32 Parity Test ===");

    checkAdditionMatchesFloat();

    checkSubtractionMatchesFloat();

    checkMultiplicationMatchesFloat();

    checkDivisionMatchesFloat();

    checkNegativeValuesMatchFloat();

    checkRepeatedOperationIsDeterministic();

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
