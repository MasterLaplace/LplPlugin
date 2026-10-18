#include <lpl/math/FixedPoint.hpp>
#include <lpl/testing/Test.hpp>

#include <limits>

LPL_TEST_SUITE(fixed_point_conversion);

namespace {

constexpr lpl::core::i32 kHighest = std::numeric_limits<lpl::core::i32>::max();
constexpr lpl::core::i32 kLowest = std::numeric_limits<lpl::core::i32>::min();

/**
 * @brief The word a float converts to, read through a volatile so the conversion runs at run time.
 */
[[nodiscard]] lpl::core::i32 wordOfFloat(float value)
{
    volatile float held = value;
    return lpl::math::Fixed32::fromFloat(held).raw();
}

/**
 * @brief The word a double converts to, read through a volatile so the conversion runs at run time.
 */
[[nodiscard]] lpl::core::i32 wordOfDouble(double value)
{
    volatile double held = value;
    return lpl::math::Fixed32::fromDouble(held).raw();
}

} // namespace

/**
 * @brief A float beyond Q16.16 saturates to the nearest end of the word, and NaN converts to zero.
 *
 * @details The conversion of a float the integer type cannot hold is undefined behaviour, and SSE
 * answers it with the lowest word whatever the sign: 1e10 used to become -32768.
 */
LPL_TEST(a_float_beyond_the_word_saturates)
{
    test.check(wordOfFloat(1.0e10f) == kHighest, "1e10 saturates to the highest word");
    test.check(wordOfFloat(-1.0e10f) == kLowest, "-1e10 saturates to the lowest word");
    test.check(wordOfFloat(std::numeric_limits<float>::infinity()) == kHighest, "infinity saturates high");
    test.check(wordOfFloat(-std::numeric_limits<float>::infinity()) == kLowest, "minus infinity saturates low");
    test.check(wordOfFloat(std::numeric_limits<float>::quiet_NaN()) == 0, "NaN converts to zero");
    test.check(wordOfFloat(32768.0f) == kHighest, "32768, one past the highest value, saturates");
    test.check(wordOfFloat(-32768.0f) == kLowest, "-32768 is the lowest word exactly");
    test.check(wordOfFloat(32767.5f) == 32767 * 65536 + 32768, "32767.5 converts exactly");
    test.check(wordOfFloat(-1.25f) == -81920, "-1.25 converts exactly");
}

/**
 * @brief A double beyond Q16.16 saturates the same way.
 */
LPL_TEST(a_double_beyond_the_word_saturates)
{
    test.check(wordOfDouble(1.0e10) == kHighest, "1e10 saturates to the highest word");
    test.check(wordOfDouble(-1.0e10) == kLowest, "-1e10 saturates to the lowest word");
    test.check(wordOfDouble(std::numeric_limits<double>::quiet_NaN()) == 0, "NaN converts to zero");
    test.check(wordOfDouble(32767.9999) == 2147483641, "just below 32768 converts to its truncated word");
    test.check(wordOfDouble(32768.0) == kHighest, "32768 saturates");
    test.check(wordOfDouble(-1.25) == -81920, "-1.25 converts exactly");
}
