/**
 * @file Genome.hpp
 * @brief Heredity, and the disaster that produces a boss.
 *
 * The personality in `ai/` is derived from an identifier: fixed, free, and the
 * same forever. That is right for temperament and wrong for a *population*, which
 * has to be able to change in response to what is killing it. So a creature also
 * carries a genome — traits that are inherited, recombined, mutated and selected.
 *
 * The consequence is a feedback loop with the player in it. Cull the slow ones
 * because they are easy targets, and selection does the rest: a few hours later
 * the species is fast, and the tactic that built the guild's fortune stops
 * working. Nobody wrote that difficulty curve.
 *
 * **Where bosses come from.** Not a spawn table. Population genetics says drift
 * dominates selection in small populations: when a species is pushed to a
 * handful of survivors, allele frequencies swing at random instead of being
 * averaged back to the mean. Most of the swings are lethal — the mutational
 * meltdown. Occasionally one is not, and a single individual inherits a
 * combination no large population could have reached. It breaks the symmetry of
 * its species, dominates the survivors, and becomes a territorial apex.
 *
 * So the rule is: **amplify mutation when the local population collapses.** The
 * anomaly is then a consequence of the player's own extermination campaign, which
 * is a far better story than a rare spawn — and it costs one conditional.
 *
 * @warning The collapse threshold is a share of the local carrying capacity, not
 *          an absolute head count. An absolute one is a threshold measured
 *          against a distribution that moves with the habitat's size, which is a
 *          mistake this codebase has now made three times in other modules.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_GENOME_HPP
#    define LPL_ECOLOGY_GENOME_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ecology {

/**
 * @struct Genome
 * @brief The heritable traits, all in world units rather than [0, 1].
 *
 * Deliberately not normalised: these are multiplied into speeds and damage, and
 * a normalised gene would need a scale beside it that could drift out of step
 * with the gene itself.
 */
struct Genome {
    math::Fixed32 maxSpeed{math::Fixed32::fromInt(4)};
    math::Fixed32 vision{math::Fixed32::fromInt(8)};
    math::Fixed32 strength{math::Fixed32::fromInt(5)};
    math::Fixed32 absorption{math::Fixed32::one()}; ///< Energy extracted per unit of food.
    math::Fixed32 size{math::Fixed32::one()};       ///< Body scale; drives the island rule.
};

/**
 * @struct Fitness
 * @brief What survival is scored on.
 *
 * Age dominates on purpose: the trait being selected for is *staying alive*, and
 * a score that weighted kills above longevity would breed something that dies
 * gloriously — which is not what evolution optimises.
 */
struct Fitness {
    core::u32 age{0u};
    core::u32 kills{0u};
    math::Fixed32 energyAbsorbed{};
    math::Fixed32 damageTaken{};

    [[nodiscard]] math::Fixed32 score() const noexcept
    {
        return math::Fixed32::fromInt(static_cast<core::i32>(age)) +
               math::Fixed32::fromInt(static_cast<core::i32>(kills)) * math::Fixed32::fromInt(10) +
               energyAbsorbed / math::Fixed32::fromInt(10) - damageTaken / math::Fixed32::fromInt(2);
    }
};

/**
 * @struct HeredityParams
 * @brief Mutation rates, and when they go wrong on purpose.
 */
struct HeredityParams {
    core::u32 mutationChance16{1u};   ///< Per-gene mutation chance, in sixteenths.
    core::f32 mutationAmplitude{0.12f}; ///< Ordinary mutation: +/- this share.

    /**
     * @brief Population share below which drift takes over, in sixteenths of K.
     *
     * Relative, not absolute. A fixed head count means "collapsed" depends on how
     * big the habitat happens to be, so the same parameter would trigger
     * constantly in a small region and never in a large one.
     */
    core::u32 collapseShare16{2u};

    core::u32 meltdownChance16{6u};     ///< Per-gene mutation chance under collapse.
    core::f32 meltdownAmplitude{0.50f}; ///< Chaotic mutation: +/- half.

    /**
     * @brief Standard deviations above the species mean that make an anomaly.
     *
     * The symmetry-breaking test. Two and a half sigma is rare without being
     * impossible — which is what an event should be.
     */
    core::f32 anomalySigma{2.5f};
};

/**
 * @brief Recombines two parents' genomes.
 *
 * Blend crossover: each gene is a weighted mix rather than a coin flip between
 * the two. A coin flip preserves the parents' exact values forever, so a
 * population can only ever hold the alleles it started with; blending lets a
 * lineage reach values neither parent had, which is what makes selection able to
 * move a species rather than only sort it.
 *
 * @param a      First parent.
 * @param b      Second parent.
 * @param stream Random stream state, advanced in place.
 * @return The child's genome, before mutation.
 */
[[nodiscard]] Genome crossover(const Genome &a, const Genome &b, core::u32 &stream);

/**
 * @brief Perturbs a genome.
 * @param genome    Genome to mutate.
 * @param chance16  Per-gene chance, in sixteenths.
 * @param amplitude Share by which a mutated gene may move.
 * @param stream    Random stream state, advanced in place.
 * @return The mutated genome.
 */
[[nodiscard]] Genome mutate(const Genome &genome, core::u32 chance16, core::f32 amplitude, core::u32 &stream);

/**
 * @brief Whether a local population has collapsed far enough for drift to rule.
 * @param local    Head count in the region.
 * @param capacity What the region supports.
 * @param params   Thresholds.
 */
[[nodiscard]] bool inMutationalMeltdown(math::Fixed32 local, math::Fixed32 capacity, const HeredityParams &params);

/**
 * @struct PopulationStats
 * @brief Mean and spread of a trait across a species, for the anomaly test.
 */
struct PopulationStats {
    math::Fixed32 mean{};
    math::Fixed32 deviation{};
    core::u32 count{0u};
};

/**
 * @brief Mean and standard deviation of strength across a sample.
 * @param genomes Sample.
 * @param count   Entries in @p genomes.
 */
[[nodiscard]] PopulationStats strengthStats(const Genome *genomes, core::u32 count);

/**
 * @brief Has this individual broken its species' symmetry?
 *
 * @param genome The individual.
 * @param stats  Its species' distribution.
 * @param params The sigma threshold.
 * @return true when it is an anomaly — a boss.
 */
[[nodiscard]] bool isAnomaly(const Genome &genome, const PopulationStats &stats, const HeredityParams &params);

/**
 * @brief Produces a child, applying meltdown amplification when appropriate.
 *
 * The whole loop in one call: recombine, then mutate at ordinary or chaotic
 * rates depending on whether the local population has collapsed. A caller does
 * not decide which; the population does.
 *
 * @param a        First parent.
 * @param b        Second parent.
 * @param local    Local head count.
 * @param capacity Local carrying capacity.
 * @param params   Heredity parameters.
 * @param stream   Random stream state, advanced in place.
 * @return The child's genome.
 */
[[nodiscard]] Genome breed(const Genome &a, const Genome &b, math::Fixed32 local, math::Fixed32 capacity,
                           const HeredityParams &params, core::u32 &stream);

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_GENOME_HPP
