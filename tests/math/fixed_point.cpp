#include <lpl/math/Cordic.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(fixed_point);

namespace {

[[nodiscard]] lpl::core::i32 distance(lpl::core::i32 lhs, lpl::core::i32 rhs)
{
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

/** @brief The Q16.16 quotient as a double-width division gives it, truncated toward zero. */
[[nodiscard]] lpl::core::i32 wideQuotient(lpl::core::i32 dividend, lpl::core::i32 divisor)
{
    return static_cast<lpl::core::i32>((static_cast<lpl::core::i64>(dividend) << 16) / divisor);
}

/** @brief Dividends on both sides of the one-word limit, 32767 and 32768 units, and at the ends. */
constexpr lpl::core::i32 kEdgeDividends[] = {0,        1,      -1,    32767,  -32767,     32768,           -32768,
                                             32769,    -32769, 65536, -65536, 2147483647, -2147483647 - 1, 1000000,
                                             -1000000, 46341,  -46341};

/** @brief Divisors from the smallest units to the ends of the word, both signs. */
constexpr lpl::core::i32 kEdgeDivisors[] = {1,      -1,    2,      -2,     3,     -3,     7,          65536,
                                            -65536, 65537, 123457, -98765, 32768, -32768, 2147483647, -2147483647 - 1};

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

/**
 * @brief A Fixed32 quotient is the one the double-width division gives, on both sides of the word.
 *
 * @details A dividend shifted by 16 bits that still fits in 32 is divided in 32 bits, any other in
 *          64: the edges of that limit, the ends of the word and a sweep of 20000 pairs from a
 *          linear congruential generator are each compared with the double-width quotient, and
 *          the quotients are folded so the two targets can be compared too.
 */
LPL_TEST(a_quotient_is_the_double_width_one_on_both_sides_of_the_word)
{
    lpl::core::u32 mismatches = 0u;
    lpl::core::u32 narrow = 0u;
    lpl::core::u32 wide = 0u;
    lpl::core::u32 fold = 0x811C9DC5u;
    const auto divideAndCompare = [&](lpl::core::i32 dividend, lpl::core::i32 divisor) {
        const lpl::core::i32 quotient =
            (lpl::math::Fixed32::fromRaw(dividend) / lpl::math::Fixed32::fromRaw(divisor)).raw();
        const lpl::core::i64 shifted = static_cast<lpl::core::i64>(dividend) << 16;
        if (shifted >= -2147483647LL && shifted <= 2147483647LL)
            ++narrow;
        else
            ++wide;
        if (quotient != wideQuotient(dividend, divisor))
            ++mismatches;
        fold = (fold ^ static_cast<lpl::core::u32>(quotient)) * 0x01000193u;
    };

    for (const lpl::core::i32 dividend : kEdgeDividends)
        for (const lpl::core::i32 divisor : kEdgeDivisors)
            divideAndCompare(dividend, divisor);

    lpl::core::u32 state = 0x2545F491u;
    for (lpl::core::u32 pair = 0u; pair < 20000u; ++pair)
    {
        state = state * 1664525u + 1013904223u;
        const auto dividend = static_cast<lpl::core::i32>(state) >> (state & 31u);
        state = state * 1664525u + 1013904223u;
        auto divisor = static_cast<lpl::core::i32>(state) >> (state & 15u);
        if (divisor == 0)
            divisor = 1;
        divideAndCompare(dividend, divisor);
    }

    test.check(mismatches == 0u, "every quotient is the double-width one");
    test.check(narrow > 1000u && wide > 1000u, "and both sides of the one-word limit were divided");

    test.measure("quotients_divided_in_one_word", narrow);
    test.measure("quotients_divided_in_two_words", wide);
    test.measureHexadecimal("quotient_signature", fold);
}
