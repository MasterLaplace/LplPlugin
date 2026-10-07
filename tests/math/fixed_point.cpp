#include <lpl/math/Cordic.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(fixed_point);

namespace {

[[nodiscard]] lpl::core::i32 distance(lpl::core::i32 lhs, lpl::core::i32 rhs)
{
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

} // namespace

/**
 * @brief Gate P0: Fixed32 arithmetic and CORDIC give the same Q16.16 words on both targets.
 *
 * @details sqrt(2)/2 is 46341 units of 1/65536 and pi/4 is 51472, both rounded.
 */
LPL_TEST(cordic_and_arithmetic_give_the_same_words)
{
    const lpl::math::Fixed32 quarterPi = lpl::math::Fixed32::pi() / lpl::math::Fixed32::fromInt(4);
    lpl::math::Fixed32 sine;
    lpl::math::Fixed32 cosine;

    lpl::math::Cordic::sincos(quarterPi, sine, cosine);

    const lpl::math::Fixed32 arctangent =
        lpl::math::Cordic::atan2(lpl::math::Fixed32::fromInt(1), lpl::math::Fixed32::fromInt(1));
    const lpl::math::Fixed32 product = lpl::math::Fixed32::fromInt(3) * lpl::math::Fixed32::half();
    const lpl::math::Fixed32 quotient = lpl::math::Fixed32::fromInt(1) / lpl::math::Fixed32::fromInt(3);

    test.check(product.raw() == 0x18000, "three times one half is exactly one and a half");
    test.check(quotient.raw() == 65536 / 3, "one third truncates to the word below it");
    test.check(distance(sine.raw(), cosine.raw()) <= 1,
               "the sine and the cosine of a quarter pi differ by one unit at most");
    test.check(distance(sine.raw(), 46341) <= 4, "and the sine is within four units of sqrt(2)/2");
    test.check(distance(arctangent.raw(), 51472) <= 4, "the arctangent of one over one is within four units of pi/4");

    test.measureHexadecimal("sine_quarter_pi", static_cast<lpl::core::u32>(sine.raw()));
    test.measureHexadecimal("cosine_quarter_pi", static_cast<lpl::core::u32>(cosine.raw()));
    test.measureHexadecimal("arctangent_one_one", static_cast<lpl::core::u32>(arctangent.raw()));
    test.measureHexadecimal("three_times_half", static_cast<lpl::core::u32>(product.raw()));
    test.measureHexadecimal("one_divided_by_three", static_cast<lpl::core::u32>(quotient.raw()));
}
