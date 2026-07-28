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

#ifndef LPL_PROCGEN_FIXEDMATH_HPP
#    define LPL_PROCGEN_FIXEDMATH_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::procgen {

/// 1/sqrt(2), the weight a diagonal step carries relative to a cardinal one.
inline const math::Fixed32 kInvSqrt2 = math::Fixed32::fromRaw(46341); // round(65536 / sqrt(2))

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
[[nodiscard]] constexpr math::Fixed32 fixedLog2(core::u32 value) noexcept
{
    if (value == 0u)
        return math::Fixed32::zero();

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

    return math::Fixed32::fromRaw(static_cast<core::i32>((exponent << 16) + fraction));
}

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
[[nodiscard]] constexpr math::Fixed32 fixedSqrt(math::Fixed32 value) noexcept
{
    if (value.raw() <= 0)
        return math::Fixed32::zero();

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
    return math::Fixed32::fromRaw(static_cast<core::i32>(root));
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_FIXEDMATH_HPP
