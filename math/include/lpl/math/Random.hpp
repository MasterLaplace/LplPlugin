/**
 * @file Random.hpp
 * @brief The deterministic randomness every pass draws from.
 *
 * Passes that make discrete choices — which WFC tile collapses, where a BSP cut
 * falls, whether a cave cell starts solid — need a random stream. It must be
 * OUR stream: `std::mt19937` is unavailable freestanding, and `rand()` differs
 * between libcs, so either would break the contract that the same seed builds
 * the same world on the host and in the kernel.
 *
 * This is a 32-bit xorshift: three shifts, no multiply-carry, no state beyond
 * one word. It is not cryptographic and does not need to be; it needs to be
 * identical everywhere and cheap, which it is.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_MATH_RANDOM_HPP
#    define LPL_MATH_RANDOM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::math {

/**
 * @class Random
 * @brief Seeded xorshift32 generator.
 */
class Random {
public:
    /**
     * @brief Seeds the stream.
     * @param seed Any value; 0 is remapped because xorshift is stuck at zero.
     */
    explicit constexpr Random(core::u32 seed) noexcept : _state(seed == 0u ? 0x9E3779B9u : seed) {}

    /// @return The next 32-bit word.
    [[nodiscard]] constexpr core::u32 next() noexcept
    {
        _state ^= _state << 13;
        _state ^= _state >> 17;
        _state ^= _state << 5;
        return _state;
    }

    /**
     * @brief Uniform integer in [0, bound).
     *
     * Uses the high bits via a widened multiply rather than a modulo: the low
     * bits of a xorshift are the weakest, and `% bound` would both lean on them
     * and skew the distribution when bound does not divide 2^32.
     *
     * @param bound Exclusive upper bound; 0 yields 0.
     * @return A value in [0, bound).
     */
    [[nodiscard]] constexpr core::u32 below(core::u32 bound) noexcept
    {
        if (bound == 0u)
            return 0u;
        return static_cast<core::u32>((static_cast<core::u64>(next()) * bound) >> 32);
    }

    /**
     * @brief Uniform integer in [low, high].
     * @param low  Inclusive lower bound.
     * @param high Inclusive upper bound; returns @p low when high < low.
     */
    [[nodiscard]] constexpr core::i32 range(core::i32 low, core::i32 high) noexcept
    {
        if (high <= low)
            return low;
        return low + static_cast<core::i32>(below(static_cast<core::u32>(high - low + 1)));
    }

    /**
     * @brief A Fixed32 uniformly in [0, 1).
     *
     * Takes the HIGH sixteen bits. Q16.16 has exactly sixteen fractional bits, so
     * either half of the word would fill the fraction — but a xorshift's low bits
     * are its weakest, which is the same reason @ref below avoids them. Masking
     * the low half here while carefully avoiding it there would have been an
     * inconsistency with visible consequences: @ref chance drives every per-cell
     * decision in the module.
     */
    [[nodiscard]] constexpr Fixed32 unit() noexcept { return Fixed32::fromRaw(static_cast<core::i32>(next() >> 16)); }

    /**
     * @brief A biased coin.
     * @param probability Chance of true, clamped to [0, 1].
     */
    [[nodiscard]] constexpr bool chance(Fixed32 probability) noexcept { return unit() < probability; }

    /// @return The raw internal state (for deriving sub-streams).
    [[nodiscard]] constexpr core::u32 state() const noexcept { return _state; }

private:
    core::u32 _state;
};

/**
 * @brief Derives an independent stream from a seed and a label.
 *
 * Passes must not share one generator: inserting a pass, or changing how many
 * numbers an earlier one draws, would shift every later pass's stream and
 * silently change unrelated parts of the world. Each pass takes its own stream
 * keyed by a distinct salt instead.
 *
 * @param seed Master world seed.
 * @param salt Per-pass constant.
 * @return A generator independent of every other salt.
 */
[[nodiscard]] constexpr Random deriveStream(core::u32 seed, core::u32 salt) noexcept
{
    core::u32 mixed = seed ^ (salt * 0x9E3779B9u);
    mixed ^= mixed >> 16;
    mixed *= 0x7FEB352Du;
    mixed ^= mixed >> 15;
    mixed *= 0x846CA68Bu;
    mixed ^= mixed >> 16;
    return Random{mixed};
}

} // namespace lpl::math

#endif // LPL_MATH_RANDOM_HPP
