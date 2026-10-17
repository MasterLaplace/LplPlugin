/**
 * @file Personality.hpp
 * @brief Six numbers that make two creatures of the same species behave differently.
 *
 * The insight worth keeping from the Rain World study is not the list of traits,
 * it is where they come from: **the creature's identifier IS the seed**. Nothing
 * is stored, nothing is rolled at spawn time, nothing has to be serialised. A
 * creature can be reduced to a node on a graph, forgotten for an hour, and
 * materialised again with the same temperament — because the temperament was
 * never data, it was a function of the id.
 *
 * That is also what makes it kernel-safe and network-safe at no cost: two
 * machines that agree on an entity's id agree on its personality without a byte
 * crossing between them.
 *
 * The traits are not decoration. Each one *modulates a parameter* of something
 * else — a flight threshold, an attack delay, a pathfinding budget — which is the
 * difference between a personality system and a tooltip.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_PERSONALITY_HPP
#    define LPL_AI_PERSONALITY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::ai {

/**
 * @struct PersonalityTraits
 * @brief Six axes in [0, 1], derived from an identifier and never stored.
 */
struct PersonalityTraits {
    math::Fixed32 aggression{};  /**< High: strikes first, pursues longer. Low: flees when hurt. */
    math::Fixed32 bravery{};     /**< High: investigates noises, ignores unreachable threats. */
    math::Fixed32 dominance{};   /**< High: leads packs, shoulders through. Low: yields. */
    math::Fixed32 energy{};      /**< High: moves fast, explores more nodes per tick. */
    math::Fixed32 nervousness{}; /**< High: overreacts to repeated stimuli. */
    math::Fixed32 sympathy{};    /**< High: tolerates, shares, warns before killing. */
};

/**
 * @brief Derives a creature's temperament from its identifier.
 *
 * A separate avalanche per trait rather than six slices of one hash: sliced bits
 * of a single 32-bit word are correlated, so a creature that came out aggressive
 * would tend to come out brave too, and the population would have two
 * personalities instead of a spread.
 *
 * Each axis has its own odd multiplier, then a full avalanche: six independent
 * draws, not six windows onto one. 16 bits of the avalanche become the
 * fractional part directly, so the value lands in [0, 1) with no division.
 *
 * @param id      The creature's identifier.
 * @param species A per-species salt, so the same id in two species differs.
 * @return The six traits, each in [0, 1].
 */
[[nodiscard]] constexpr PersonalityTraits personalityOf(core::u32 id, core::u32 species = 0u) noexcept;

/**
 * @brief How long a creature waits before attacking, as a share of the base delay.
 *
 * The archetypal use: aggression does not add a flag, it scales a number. High
 * aggression drops the delay toward a quarter, low aggression stretches it.
 *
 * @param traits The creature's temperament.
 * @return 1.25 - aggression: a multiplier in roughly [0.25, 1.25].
 */
[[nodiscard]] constexpr math::Fixed32 attackDelayScale(const PersonalityTraits &traits) noexcept
{
    return math::Fixed32::fromRaw(0x14000) - traits.aggression;
}

/**
 * @brief The damage share at which a creature breaks off and runs.
 *
 * Bravery raises it, nervousness lowers it. Two axes on one number, which is what
 * makes a nervous-but-brave creature behave unlike either extreme.
 *
 * @param traits The creature's temperament.
 * @return A threshold in [0, 1].
 */
[[nodiscard]] constexpr math::Fixed32 fleeThreshold(const PersonalityTraits &traits) noexcept;

} // namespace lpl::ai

#    include "Personality.inl"

#endif // LPL_AI_PERSONALITY_HPP
