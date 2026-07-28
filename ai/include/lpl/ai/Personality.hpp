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
    math::Fixed32 aggression{};  ///< High: strikes first, pursues longer. Low: flees when hurt.
    math::Fixed32 bravery{};     ///< High: investigates noises, ignores unreachable threats.
    math::Fixed32 dominance{};   ///< High: leads packs, shoulders through. Low: yields.
    math::Fixed32 energy{};      ///< High: moves fast, explores more nodes per tick.
    math::Fixed32 nervousness{}; ///< High: overreacts to repeated stimuli.
    math::Fixed32 sympathy{};    ///< High: tolerates, shares, warns before killing.
};

/**
 * @brief Derives a creature's temperament from its identifier.
 *
 * A separate avalanche per trait rather than six slices of one hash: sliced bits
 * of a single 32-bit word are correlated, so a creature that came out aggressive
 * would tend to come out brave too, and the population would have two
 * personalities instead of a spread.
 *
 * @param id      The creature's identifier.
 * @param species A per-species salt, so the same id in two species differs.
 * @return The six traits, each in [0, 1].
 */
[[nodiscard]] constexpr PersonalityTraits personalityOf(core::u32 id, core::u32 species = 0u) noexcept
{
    const auto avalanche = [](core::u32 v) -> core::u32 {
        v ^= v >> 16;
        v *= 0x7FEB352Du;
        v ^= v >> 15;
        v *= 0x846CA68Bu;
        v ^= v >> 16;
        return v;
    };

    // Distinct odd multipliers per axis, then a full avalanche each. Six
    // independent draws, not six windows onto one.
    const auto trait = [&](core::u32 salt) -> math::Fixed32 {
        const core::u32 h = avalanche((id ^ (species * 0x9E3779B1u)) * salt + salt);
        // 16 bits of the avalanche become the fractional part directly, so the
        // value lands in [0, 1) with no division.
        return math::Fixed32::fromRaw(static_cast<core::i32>(h & 0xFFFFu));
    };

    PersonalityTraits traits;
    traits.aggression = trait(0x1B873593u);
    traits.bravery = trait(0xCC9E2D51u);
    traits.dominance = trait(0x85EBCA6Bu);
    traits.energy = trait(0xC2B2AE35u);
    traits.nervousness = trait(0x27D4EB2Fu);
    traits.sympathy = trait(0x165667B1u);
    return traits;
}

/**
 * @brief How long a creature waits before attacking, as a share of the base delay.
 *
 * The archetypal use: aggression does not add a flag, it scales a number. High
 * aggression drops the delay toward a quarter, low aggression stretches it.
 *
 * @param traits The creature's temperament.
 * @return A multiplier in roughly [0.25, 1.25].
 */
[[nodiscard]] constexpr math::Fixed32 attackDelayScale(const PersonalityTraits &traits) noexcept
{
    return math::Fixed32::fromRaw(0x14000) - traits.aggression; // 1.25 - aggression
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
[[nodiscard]] constexpr math::Fixed32 fleeThreshold(const PersonalityTraits &traits) noexcept
{
    math::Fixed32 t = math::Fixed32::half() + (traits.bravery - traits.nervousness) * math::Fixed32::half();
    if (t < math::Fixed32::zero())
        t = math::Fixed32::zero();
    if (t > math::Fixed32::one())
        t = math::Fixed32::one();
    return t;
}

} // namespace lpl::ai

#endif // LPL_AI_PERSONALITY_HPP
