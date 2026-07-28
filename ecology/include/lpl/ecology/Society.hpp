/**
 * @file Society.hpp
 * @brief Packs, isolation, overflow and invasion — the four consequences.
 *
 * Everything here exists to make a player's action have a second-order effect
 * they did not ask for.
 *
 * **Packs** are not groups, they are institutions with a life cycle: an alpha
 * chosen on merit, budding when the territory runs short, dissolution when the
 * leader dies. Kill the alpha for its trophy and the pack does not simply lose a
 * member — it may shatter into a dozen solitary aggressive animals spreading down
 * the trade roads. The trophy was real. So is the consequence.
 *
 * **The island rule** is where this module meets `procgen`. Foster's observation
 * is that isolation drives small species toward gigantism and large ones toward
 * dwarfism, and the isolation is already computed: a valley ringed by mountains,
 * a cave pocket with one entrance — the connectivity passes know exactly where
 * those are. So the strange fauna in the far valley is not placed there, it is
 * *implied* by the map, and finding it is a reward for having walked that far.
 *
 * **Overflow** is the emergent raid. Nothing decides to attack a village. A
 * region exceeds its carrying capacity, its animals starve, and a starving animal
 * stops respecting the thing that used to keep it away. The rule is one line; the
 * event reads as a story.
 *
 * **Invasion** is the exogenous shock, with the weakness the literature gives it:
 * an introduced population passes through a founder bottleneck, so it is
 * numerically dangerous and *genetically uniform* — vulnerable to anything that
 * works on one of them, if the player acts before it diversifies.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_SOCIETY_HPP
#    define LPL_ECOLOGY_SOCIETY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ecology {

/// Value meaning "no pack".
inline constexpr core::u32 kSolitary = 0xFFFFFFFFu;

/**
 * @struct PackMember
 * @brief One animal, as the social layer sees it.
 */
struct PackMember {
    core::u32 id{0u};          ///< Identity; also the tie-break for every ranking.
    core::u32 lineage{0u};     ///< Family. Kin recognition works on this.
    core::u32 pack{kSolitary}; ///< Current pack.
    math::Fixed32 fitness{};   ///< What the alpha contest is decided on.
    bool alpha{false};
    bool alive{true};
};

/**
 * @struct PackParams
 * @brief When a pack forms, splits, and falls apart.
 */
struct PackParams {
    core::u32 maxSize{8u};              ///< Above this the pack buds.
    core::u32 minSize{2u};              ///< Below this it stops being a pack.
    core::u32 dissolutionChance16{10u}; ///< Chance of shattering when the alpha dies, in sixteenths.
    bool adoptStrays{true};             ///< Whether lone kin may join an existing pack.
};

/**
 * @struct PackEvents
 * @brief What one social step actually did.
 */
struct PackEvents {
    core::u32 formed{0u};
    core::u32 budded{0u};
    core::u32 dissolved{0u};
    core::u32 adopted{0u};
    core::u32 alphaChanges{0u};
    core::u32 scattered{0u}; ///< Members turned solitary by a dissolution.
};

/**
 * @brief Runs one step of the pack life cycle.
 *
 * Alphas are elected by fitness with the identifier as the tie-break. That
 * tie-break is not a detail: fitness values collide constantly in a population of
 * clones, and without it the leader would be whichever member the loop reached
 * first — so two machines would crown different animals and diverge.
 *
 * @param members Population, updated in place.
 * @param count   Entries in @p members.
 * @param params  Thresholds.
 * @param stream  Random stream state, advanced in place.
 * @return What happened.
 */
PackEvents stepPacks(PackMember *members, core::u32 count, const PackParams &params, core::u32 &stream);

/**
 * @brief Kills a member and lets the social consequences follow.
 *
 * Separate from ordinary death because killing the ALPHA is the interesting case,
 * and a caller should not have to know that.
 *
 * @param members Population.
 * @param count   Entries in @p members.
 * @param id      Who dies.
 * @param params  Thresholds.
 * @param stream  Random stream state.
 * @return What happened as a result.
 */
PackEvents killMember(PackMember *members, core::u32 count, core::u32 id, const PackParams &params, core::u32 &stream);

/**
 * @struct IslandParams
 * @brief How hard isolation pushes body size.
 */
struct IslandParams {
    /**
     * @brief Region size below which the island rule applies, as a share of the map.
     *
     * Relative, for the reason every threshold in this codebase is relative: a
     * pocket of forty cells is an island on a small map and a puddle on a large
     * one.
     */
    core::f32 isolationShare{0.05f};

    core::f32 pressure{0.02f};      ///< Share of the gap closed per generation.
    core::f32 giantTarget{2.5f};    ///< Size a small species converges toward when isolated.
    core::f32 dwarfTarget{0.4f};    ///< Size a large species converges toward.
    core::f32 smallThreshold{1.0f}; ///< Below this a species counts as small.
};

/**
 * @brief Applies one generation of insular pressure to a genome.
 *
 * Gradual on purpose — this is meant to be discovered as a curiosity after a long
 * absence, not to be watched happening. A step that converged immediately would
 * turn a biogeographic law into a light switch.
 *
 * @warning The direction is decided by @p ancestralSize, the size the species had
 *          when it arrived — NOT by its current size. Reading the current size
 *          makes the threshold an unstable equilibrium: a small species grows
 *          until it crosses it, is reclassified as large, and is pushed back
 *          down. Measured, a species starting at 0.5 and one starting at 3.0 both
 *          converged on 1.0 and sat there, which is the opposite of the rule —
 *          it says isolation drives species APART. Foster's rule is about what a
 *          lineage was when it was cut off, and the code has to remember that.
 *
 * @param genome        Genome to push.
 * @param ancestralSize Body size of the founding population.
 * @param isolated      Whether the animal's region qualifies as an island.
 * @param params        Pressure and targets.
 * @return The adjusted genome.
 */
[[nodiscard]] Genome applyIslandRule(const Genome &genome, math::Fixed32 ancestralSize, bool isolated,
                                     const IslandParams &params);

/**
 * @brief Marks which regions are isolated enough for the island rule.
 *
 * Reads the connectivity `procgen` already computed rather than recomputing it:
 * the pockets a repair pass had to join, and the regions a partition produced,
 * are the same places that make a species strange.
 *
 * @param regions      Region index per cell.
 * @param regionCount  Number of regions.
 * @param params       Isolation threshold.
 * @param outIsolated  Receives one flag per region.
 * @return Number of isolated regions.
 */
core::u32 markIsolatedRegions(const procgen::Grid<core::u32> &regions, core::u32 regionCount,
                              const IslandParams &params, lpl::pmr::vector<core::u8> &outIsolated);

/**
 * @struct OverflowParams
 * @brief When hunger overrides fear.
 */
struct OverflowParams {
    /**
     * @brief Energy share below which an animal ignores danger.
     *
     * The whole raid mechanic. A well-fed animal weighs a settlement's danger and
     * stays away; a starving one weighs it against starving and walks in. No code
     * anywhere decides to attack a village.
     */
    core::f32 starvingBelow{0.25f};

    core::f32 overcrowdedAbove{0.90f}; ///< Share of carrying capacity that starts the starvation.
};

/**
 * @struct OverflowState
 * @brief Whether a region is about to spill, and how badly.
 */
struct OverflowState {
    bool overcrowded{false};
    bool raiding{false};
    math::Fixed32 pressure{}; ///< How far past capacity, in [0, 1+].
    core::u32 starving{0u};   ///< Animals below the starvation threshold.
};

/**
 * @brief Evaluates a region's overflow pressure.
 * @param population Current head count.
 * @param capacity   What the region supports.
 * @param energies   Per-animal energy share, in [0, 1].
 * @param count      Entries in @p energies.
 * @param params     Thresholds.
 * @return The region's state.
 */
[[nodiscard]] OverflowState evaluateOverflow(math::Fixed32 population, math::Fixed32 capacity,
                                             const math::Fixed32 *energies, core::u32 count,
                                             const OverflowParams &params);

/**
 * @brief Builds a founder population: numerous, and genetically nearly identical.
 *
 * The founder effect, which is the invader's one weakness. A caller can measure
 * the resulting diversity and act on it — a targeted counter that works on one
 * of them works on all of them, until they diversify.
 *
 * @param ancestor Genome the founders descend from.
 * @param count    Founders to produce.
 * @param spread   How much they vary; small by definition.
 * @param stream   Random stream state.
 * @param out      Receives the founders.
 */
void seedInvasion(const Genome &ancestor, core::u32 count, core::f32 spread, core::u32 &stream,
                  lpl::pmr::vector<Genome> &out);

/**
 * @brief Genetic diversity of a population, as the mean spread across genes.
 *
 * The number a player's analysts would report. A native population scores high; a
 * fresh invasion scores near zero.
 *
 * @param genomes Population.
 * @param count   Entries in @p genomes.
 * @return Mean coefficient of variation across the genes.
 */
[[nodiscard]] math::Fixed32 geneticDiversity(const Genome *genomes, core::u32 count);

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_SOCIETY_HPP
