/**
 * @file FixedMath.hpp
 * @brief The square roots the passes need, and cannot get from libm.
 *
 * `lpl::pmr::sqrt` maps to the hardware instruction and is documented as being
 * for **non-authoritative** float paths only. Everything in this module is
 * authoritative — it has to fold bit-identically on the Linux oracle and in the
 * i686 kernel — so where an algorithm genuinely needs a square root (the stream
 * power law wants the square root of a drainage area) it has to come from
 * integer arithmetic.
 *
 * Both roots below are exact: they return the largest integer whose square does
 * not exceed the input, computed by the classic restoring shift-and-subtract.
 * No division, no table, no floating point, identical on every target.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_MATH_FIXEDMATH_HPP
#    define LPL_MATH_FIXEDMATH_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::math {

/// 1/sqrt(2), the weight a diagonal step carries relative to a cardinal one.
inline const Fixed32 kInvSqrt2 = Fixed32::fromRaw(46341); // round(65536 / sqrt(2))

/**
 * @brief Integer square root: the largest @c r with r*r <= @p value.
 *
 * Restoring shift-and-subtract, one bit per iteration from the top. Exact for
 * every 32-bit input and free of division.
 *
 * @param value Radicand.
 * @return floor(sqrt(value)).
 */
[[nodiscard]] constexpr core::u32 integerSqrt(core::u32 value) noexcept
{
    core::u32 remainder = value;
    core::u32 root = 0u;
    // Highest even power of four that fits a u32.
    core::u32 bit = core::u32{1} << 30;

    while (bit > remainder)
        bit >>= 2;

    while (bit != 0u)
    {
        if (remainder >= root + bit)
        {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        }
        else
        {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

/**
 * @brief Base-two logarithm of a positive integer, in Fixed32.
 *
 * The integer part is the position of the highest set bit; the fraction is a
 * straight line between that power of two and the next. Piecewise-linear rather
 * than exact, which for its purpose is the right trade: this exists to compress
 * quantities whose useful information is their *order of magnitude*.
 *
 * Flow accumulation is the case in point. Across one map it spans four orders of
 * magnitude — a ridge cell drains one cell, the river mouth drains half the
 * world — so a linear scale puts everything but the trunk at zero, and even a
 * square root leaves a ratio of ninety to one. A logarithm is what the wetness
 * index in the literature actually uses, @f$\ln(a / \tan\beta)@f$, and it is the
 * only compression that gives a usable ramp here.
 *
 * @param value Argument; 0 yields 0 rather than negative infinity.
 * @return log2(value) in Q16.16, up to 32.
 */
[[nodiscard]] constexpr Fixed32 fixedLog2(core::u32 value) noexcept
{
    if (value == 0u)
        return Fixed32::zero();

    core::u32 exponent = 0u;
    core::u32 rest = value;
    while (rest > 1u)
    {
        rest >>= 1;
        ++exponent;
    }

    // Where value sits between 2^exponent and 2^(exponent+1), in Q16.16.
    const core::u32 floorPower = core::u32{1} << exponent;
    const core::u32 fraction =
        exponent == 0u ? 0u : static_cast<core::u32>(((static_cast<core::u64>(value - floorPower)) << 16) >> exponent);

    return Fixed32::fromRaw(static_cast<core::i32>((exponent << 16) + fraction));
}

/// log2(e), the constant that turns a natural exponential into a base-two one.
inline constexpr Fixed32 kLog2E = Fixed32::fromRaw(94548); // round(65536 * 1.4426950408889634)

/**
 * @brief Two raised to a Fixed32 power, in Fixed32.
 *
 * The inverse of @ref fixedLog2, and it exists for the same reason: a module under
 * the determinism contract cannot call libm, and an exponential is not something
 * integer arithmetic gives up for free the way a square root does.
 *
 * Split, then series. @f$2^x = 2^n \cdot 2^f@f$ with @c n the floor and @c f the
 * fraction, so the integer part becomes a shift — exact, whatever its magnitude —
 * and only @f$2^f@f$ on [0, 1) needs approximating. Six Taylor terms of
 * @f$e^{f\ln 2}@f$ evaluated by Horner stay within 6e-5 relative, measured over the
 * whole fractional range: about eight raw units of Q16.16 at the worst point.
 *
 * Most of that is the Horner evaluation and not the truncated series. Each step
 * discards the low sixteen bits of a product, so six steps lose roughly six units
 * before the coefficients contribute anything — which is why adding terms past the
 * sixth does not help, and why the accuracy is stated as measured rather than as
 * whatever the remainder of the Taylor expansion promises.
 *
 * Saturating at both ends rather than wrapping. This is called on softmax scores,
 * where a large negative argument is the ordinary case and must give zero, and a
 * large positive one means an earlier overflow — a wrapped result there would turn
 * the smallest weight in a distribution into the largest.
 *
 * @param value Exponent.
 * @return 2^value, clamped to [0, Fixed32::max()].
 */
[[nodiscard]] constexpr Fixed32 fixedExp2(Fixed32 value) noexcept
{
    // Below this the result is smaller than one raw unit; above it, out of range.
    if (value.raw() <= -(31 << 16))
        return Fixed32::zero();
    if (value.raw() >= (15 << 16))
        return Fixed32::max();

    // Arithmetic shift floors for negatives too, which is what the split needs:
    // -2.25 must give n = -3 and f = 0.75, not n = -2 and f = -0.25.
    const core::i32 integerPart = value.raw() >> 16;
    const core::u32 fraction = static_cast<core::u32>(value.raw()) & 0xFFFFu;

    // Taylor coefficients of e^(f ln2), rounded to Q16.16.
    constexpr core::i64 c1 = 45426; // ln2
    constexpr core::i64 c2 = 15743; // (ln2)^2 / 2
    constexpr core::i64 c3 = 3638;  // (ln2)^3 / 6
    constexpr core::i64 c4 = 630;
    constexpr core::i64 c5 = 87;
    constexpr core::i64 c6 = 10;

    const core::i64 f = static_cast<core::i64>(fraction);
    core::i64 series = c6;
    series = ((series * f) >> 16) + c5;
    series = ((series * f) >> 16) + c4;
    series = ((series * f) >> 16) + c3;
    series = ((series * f) >> 16) + c2;
    series = ((series * f) >> 16) + c1;
    series = ((series * f) >> 16) + Fixed32::kOne;

    if (integerPart >= 0)
    {
        const core::i64 scaled = series << integerPart;
        return scaled >= static_cast<core::i64>(Fixed32::max().raw()) ?
                   Fixed32::max() :
                   Fixed32::fromRaw(static_cast<core::i32>(scaled));
    }
    return Fixed32::fromRaw(static_cast<core::i32>(series >> (-integerPart)));
}

/**
 * @brief The natural exponential of a Fixed32, in Fixed32.
 *
 * @f$e^x = 2^{x \log_2 e}@f$, so there is one approximation in the module and not
 * two. A separate series for base e would be a second thing to keep in agreement
 * with the first.
 *
 * @param value Exponent.
 * @return e^value, clamped like @ref fixedExp2.
 */
[[nodiscard]] constexpr Fixed32 fixedExp(Fixed32 value) noexcept { return fixedExp2(value * kLog2E); }

/**
 * @brief Square root of a non-negative Fixed32, in Fixed32.
 *
 * sqrt(x) in Q16.16 is sqrt(raw << 16) — a 48-bit radicand, so the intermediate
 * is 64-bit. Negative inputs return zero rather than trapping: the passes that
 * call this feed it squared distances and cell counts, where a negative value
 * means an earlier overflow, and propagating a zero is more useful than
 * whatever a wrapped root would be.
 *
 * @param value Radicand; negative yields zero.
 * @return floor(sqrt(value)) as Fixed32.
 */
[[nodiscard]] constexpr Fixed32 fixedSqrt(Fixed32 value) noexcept
{
    if (value.raw() <= 0)
        return Fixed32::zero();

    core::u64 remainder = static_cast<core::u64>(static_cast<core::u32>(value.raw())) << 16;
    core::u64 root = 0u;
    core::u64 bit = core::u64{1} << 46;

    while (bit > remainder)
        bit >>= 2;

    while (bit != 0u)
    {
        if (remainder >= root + bit)
        {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        }
        else
        {
            root >>= 1;
        }
        bit >>= 2;
    }
    return Fixed32::fromRaw(static_cast<core::i32>(root));
}

} // namespace lpl::math

#endif // LPL_MATH_FIXEDMATH_HPP
