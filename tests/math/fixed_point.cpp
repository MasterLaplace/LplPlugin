#include <lpl/math/Cordic.hpp>
#include <lpl/math/FixedMath.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(fixed_point);

namespace {

[[nodiscard]] lpl::core::i32 distance(lpl::core::i32 lhs, lpl::core::i32 rhs)
{
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

/**
 * @brief The Q16.16 quotient as a double-width division gives it, truncated toward zero.
 */
[[nodiscard]] lpl::core::i32 wideQuotient(lpl::core::i32 dividend, lpl::core::i32 divisor)
{
    return static_cast<lpl::core::i32>((static_cast<lpl::core::i64>(dividend) << 16) / divisor);
}

/** @brief Dividends on both sides of the one-word limit, 32767 and 32768 units, and at the ends. */
constexpr lpl::core::i32 kEdgeDividends[] = {0,        1,      -1,    32767,  -32767,     32768,           -32768,
                                             32769,    -32769, 65536, -65536, 2147483647, -2147483647 - 1, 1000000,
                                             -1000000, 46341,  -46341};

/** @brief Values on both sides of three raw units, around one unit, and at the ends of the word. */
constexpr lpl::core::i32 kEdgeReciprocals[] = {1,     2,      3,          4,  5,  7,  65535,  65536,
                                               65537, 131072, 2147483647, -1, -2, -3, -65536, -2147483647 - 1};

/** @brief Divisors from the smallest units to the ends of the word, both signs. */
constexpr lpl::core::i32 kEdgeDivisors[] = {1,      -1,    2,      -2,     3,     -3,     7,          65536,
                                            -65536, 65537, 123457, -98765, 32768, -32768, 2147483647, -2147483647 - 1};

constexpr lpl::core::u64 kLastSquare = 0xFFFFFFFE00000001ull; ///< (2^32 - 1)^2, the largest square a u64 holds.

constexpr lpl::core::u64 kWeylStep = 0x9E3779B97F4A7C15ull; ///< Weyl's step, spreading the draws over the range.

constexpr lpl::core::u32 kRootDraws = 4096u; ///< Values the tests of the roots draw.

/**
 * @brief Whether @p root is the floor of the square root of @p value, computed apart from the root under test.
 */
[[nodiscard]] bool isFloorRoot(lpl::core::u64 value, lpl::core::u32 root)
{
    const lpl::core::u64 square = static_cast<lpl::core::u64>(root) * root;
    const lpl::core::u64 next = static_cast<lpl::core::u64>(root) + 1u;
    return square <= value && (root == 0xFFFFFFFFu || next * next > value);
}

} // namespace

/**
 * @brief The root of a 64-bit radicand is the exact floor of its square root, from 0 to 2^64 - 1, the last square
 *        and the value below it included.
 *
 * @details Each draw is checked by squaring, apart from the root: r^2 <= v < (r + 1)^2. The roots are folded so
 *          that both targets compare them.
 */
LPL_TEST(the_root_of_a_64_bit_radicand_is_its_exact_floor_up_to_the_last_value)
{
    test.check(lpl::math::integerSqrt(lpl::core::u64{0}) == 0u && lpl::math::integerSqrt(lpl::core::u64{3}) == 1u &&
                   lpl::math::integerSqrt(lpl::core::u64{4}) == 2u,
               "0, 3 and 4");
    test.check(lpl::math::integerSqrt(lpl::core::u64{1} << 32) == 65536u &&
                   lpl::math::integerSqrt((lpl::core::u64{1} << 32) - 1u) == 65535u,
               "2^32 and the value below it");
    test.check(lpl::math::integerSqrt(lpl::core::u64{1} << 62) == 0x80000000u, "2^62");
    test.check(lpl::math::integerSqrt(kLastSquare) == 0xFFFFFFFFu &&
                   lpl::math::integerSqrt(kLastSquare - 1u) == 0xFFFFFFFEu,
               "(2^32 - 1)^2 and the value below it");
    test.check(lpl::math::integerSqrt(~lpl::core::u64{0}) == 0xFFFFFFFFu, "2^64 - 1");

    bool exact = true;
    lpl::core::u32 fold = 0x811C9DC5u;
    for (lpl::core::u32 draw = 0u; draw < kRootDraws; ++draw)
    {
        const lpl::core::u64 value = static_cast<lpl::core::u64>(draw) * kWeylStep >> (draw % 64u);
        const lpl::core::u32 root = lpl::math::integerSqrt(value);
        exact = exact && isFloorRoot(value, root);
        fold = (fold ^ root) * 0x01000193u;
    }
    test.check(exact, "every draw over the range is the floor of its root");
    test.measureHexadecimal("root_signature", fold);
}

/**
 * @brief The root of a Fixed32 is the floor of the square root of its raw word shifted by 16, a Q32 value, on both
 *        sides of a raw unit and up to the largest word.
 */
LPL_TEST(the_root_of_a_fixed32_is_the_floor_root_of_its_raw_word_in_q32)
{
    bool exact = lpl::math::fixedSqrt(lpl::math::Fixed32::fromRaw(-1)) == lpl::math::Fixed32::zero();
    lpl::core::u32 fold = 0x811C9DC5u;
    for (lpl::core::u32 draw = 0u; draw < kRootDraws; ++draw)
    {
        const lpl::core::i32 raw =
            static_cast<lpl::core::i32>((static_cast<lpl::core::u64>(draw) * kWeylStep >> (draw % 31u + 33u)));
        const lpl::core::i32 root = lpl::math::fixedSqrt(lpl::math::Fixed32::fromRaw(raw)).raw();
        exact = exact && root >= 0 &&
                isFloorRoot(static_cast<lpl::core::u64>(raw) << 16, static_cast<lpl::core::u32>(root));
        fold = (fold ^ static_cast<lpl::core::u32>(root)) * 0x01000193u;
    }
    test.check(exact, "every draw from a raw unit to the largest word is the floor of its root in Q32");
    test.measureHexadecimal("fixed_root_signature", fold);
}

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

/**
 * @brief A reciprocal is the quotient of one by the value, on both sides of three raw units.
 *
 * @details A value of three raw units or more divides 2^32 in one word, any other goes through
 *          operator/: the edges and a sweep of 20000 values from a linear congruential generator
 *          are each compared with one() / value, and the reciprocals are folded so the two targets
 *          can be compared too.
 */
LPL_TEST(a_reciprocal_is_the_quotient_of_one_on_both_sides_of_three_units)
{
    lpl::core::u32 mismatches = 0u;
    lpl::core::u32 inOneWord = 0u;
    lpl::core::u32 byQuotient = 0u;
    lpl::core::u32 fold = 0x811C9DC5u;
    const auto invertAndCompare = [&](lpl::core::i32 raw) {
        const lpl::math::Fixed32 value = lpl::math::Fixed32::fromRaw(raw);
        const lpl::core::i32 reciprocal = value.reciprocal().raw();
        if (raw > 2)
            ++inOneWord;
        else
            ++byQuotient;
        if (reciprocal != (lpl::math::Fixed32::one() / value).raw())
            ++mismatches;
        fold = (fold ^ static_cast<lpl::core::u32>(reciprocal)) * 0x01000193u;
    };

    for (const lpl::core::i32 raw : kEdgeReciprocals)
        invertAndCompare(raw);

    lpl::core::u32 state = 0x9E3779B9u;
    for (lpl::core::u32 draw = 0u; draw < 20000u; ++draw)
    {
        state = state * 1664525u + 1013904223u;
        auto raw = static_cast<lpl::core::i32>(state) >> (state & 31u);
        if (raw == 0)
            raw = 1;
        invertAndCompare(raw);
    }

    test.check(mismatches == 0u, "every reciprocal is the quotient of one by the value");
    test.check(inOneWord > 1000u && byQuotient > 1000u, "and both sides of three raw units were inverted");

    test.measure("reciprocals_in_one_word", inOneWord);
    test.measure("reciprocals_by_quotient", byQuotient);
    test.measureHexadecimal("reciprocal_signature", fold);
}
