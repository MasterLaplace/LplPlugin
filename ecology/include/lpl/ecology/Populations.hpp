/**
 * @file Populations.hpp
 * @brief Predator and prey, bounded so the game survives the mathematics.
 *
 * Lotka and Volterra's equations are the right model and the wrong program. In
 * their classical form,
 *
 * @f[ \frac{dx}{dt} = \alpha x - \beta x y, \qquad
 *     \frac{dy}{dt} = -\gamma y + \delta x y @f]
 *
 * the orbits are neutrally stable: they neither converge nor diverge, which
 * sounds ideal and is a trap. Integrate them numerically — with any explicit
 * scheme, at any step size — and the orbits spiral *outward*, because the
 * discretisation adds energy the continuous system does not have. Amplitude grows
 * until a trough passes below one individual, and a population that reaches zero
 * in a game does not come back. The survey calls it pseudo-extinction and it is
 * the single most common way this model is shipped broken.
 *
 * Two corrections, both from the literature, both non-optional here:
 *
 *  - **Carrying capacity.** Prey grow logistically, @f$\alpha x (1 - x/K)@f$, not
 *    exponentially. Finite grass, finite prey. This alone turns the neutral orbit
 *    into a stable spiral — a system that RETURNS to equilibrium after a shock
 *    rather than remembering it forever.
 *  - **A refuge.** A floor below which predation cannot reach: burrows, cliffs,
 *    the deep water. Ecologically it is intraspecific variation; computationally
 *    it is the guarantee that no species ever reaches zero, so a world cannot be
 *    permanently emptied by one bad hour.
 *
 * Everything here is Fixed32 and integrates at a fixed step, because these
 * populations decide what spawns and what a player finds.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_POPULATIONS_HPP
#    define LPL_ECOLOGY_POPULATIONS_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ecology {

/// Trophic levels a web may hold. Four is the pyramid the energetics allow.
inline constexpr core::u32 kMaxTrophicLevels = 4u;

/**
 * @enum TrophicLevel
 * @brief Where a species sits in the flow of energy.
 */
enum class TrophicLevel : core::u8 {
    Producer = 0, ///< Makes its own energy: plants, mana crystals, algae.
    Primary,      ///< Herbivores. The r-strategists: many, cheap, fast.
    Secondary,    ///< Carnivores. Regulate the herbivores.
    Apex          ///< No natural predator. The K-strategists: few, costly, slow.
};

/**
 * @struct SpeciesParams
 * @brief One population's demography.
 */
struct SpeciesParams {
    TrophicLevel level{TrophicLevel::Primary};

    math::Fixed32 growth{math::Fixed32::fromRaw(0x0CCC)};   ///< @f$\alpha@f$: intrinsic growth per step.
    math::Fixed32 mortality{math::Fixed32::fromRaw(0x0666)};///< @f$\gamma@f$: death rate with no food.
    math::Fixed32 predation{math::Fixed32::fromRaw(0x0199)};///< @f$\beta@f$: how hard it hits its prey.
    math::Fixed32 conversion{math::Fixed32::fromRaw(0x1999)};///< @f$\delta@f$: prey eaten to offspring.

    math::Fixed32 capacity{math::Fixed32::fromInt(1000)};   ///< @f$K@f$: what the habitat supports.

    /**
     * @brief Population that predation cannot reach.
     *
     * The refuge. Not a floor bolted on to stop a crash — a modelled fact: some
     * of the prey are in burrows, and a predator cannot eat them however hungry
     * it is. It is also what makes the system recoverable, which is the property
     * a game needs and the classical model does not have.
     */
    math::Fixed32 refuge{math::Fixed32::fromInt(4)};
};

/**
 * @struct Species
 * @brief A population and what it feeds on.
 */
struct Species {
    SpeciesParams params{};
    math::Fixed32 population{};
    core::u32 preyIndex{0xFFFFFFFFu}; ///< Index of what it eats; kNoPrey for producers.

    static constexpr core::u32 kNoPrey = 0xFFFFFFFFu;
};

/**
 * @struct TrophicWeb
 * @brief The whole food web, stepped together.
 */
struct TrophicWeb {
    lpl::pmr::vector<Species> species;

    /// @brief Adds a species and returns its index.
    core::u32 add(const SpeciesParams &params, math::Fixed32 initial, core::u32 preyIndex);

    /**
     * @brief Advances every population one step.
     *
     * Logistic growth for producers and prey, mass action for predation, and the
     * refuge applied before any predator is fed — so the untouchable fraction
     * really is untouchable rather than merely restored afterwards.
     *
     * @param steps How many steps to run.
     */
    void step(core::u32 steps = 1u);

    /// @return The population of @p index, or zero.
    [[nodiscard]] math::Fixed32 populationOf(core::u32 index) const;

    /**
     * @brief Total population at a trophic level.
     * @param level Level to sum.
     */
    [[nodiscard]] math::Fixed32 levelTotal(TrophicLevel level) const;

    /**
     * @brief Removes a species entirely — a hunt, a cull, an extinction event.
     *
     * The mechanism behind a trophic cascade: everything downstream reacts, and
     * nothing here says how. That is the point.
     *
     * @param index Species to remove.
     */
    void extirpate(core::u32 index);

    /// @brief FNV-1a fold of every population, for determinism checks.
    [[nodiscard]] core::u32 fold() const;
};

/**
 * @brief The share of a level's energy that reaches the next one.
 *
 * The ten-percent rule. It is why a pyramid is a pyramid: an apex predator needs
 * ten times its own biomass in mesopredators beneath it, and a thousand times in
 * plants. Expressed here so a caller can size a habitat from the top down rather
 * than guessing and discovering the top level starves.
 *
 * @param level Level to ask about.
 * @return The transfer efficiency, in [0, 1].
 */
[[nodiscard]] constexpr math::Fixed32 transferEfficiency(TrophicLevel level) noexcept
{
    (void) level;
    return math::Fixed32::fromRaw(0x199A); // 0.10
}

/**
 * @brief Population a level can support given the one below it.
 * @param below   Population of the supporting level.
 * @param level   Level being sized.
 * @return The supportable population.
 */
[[nodiscard]] constexpr math::Fixed32 supportableBy(math::Fixed32 below, TrophicLevel level) noexcept
{
    return below * transferEfficiency(level);
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_POPULATIONS_HPP
