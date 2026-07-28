/**
 * @file Aggregation.hpp
 * @brief Diffusion-limited aggregation: growth that branches on its own.
 *
 * A seed of open space sits at the centre. Particles are released from the
 * periphery and wander until they touch it, at which point they stick. Nothing
 * decides where a branch should go — the shape emerges from the fact that a
 * protruding tip intercepts a wandering particle long before it can reach the
 * sheltered gaps behind it. That screening effect is what produces the
 * dendritic, fractal structure (dimension around 1.71 in 2D) that no noise
 * function reproduces convincingly.
 *
 * It is the right generator for anything that grew rather than being built:
 * lightning-shaped cave networks, river deltas seen from above, crystal
 * formations, corrosion.
 *
 * The cost is real: every particle random-walks until it lands, and many walk a
 * long way. Two bounds keep it finite — a step budget per particle, and a
 * spawn radius that tracks the current extent of the cluster instead of the
 * whole map, so a particle starts near where it can actually stick.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_AGGREGATION_HPP
#    define LPL_PROCGEN_AGGREGATION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Dungeon.hpp>
#    include <lpl/procgen/Grid.hpp>

namespace lpl::procgen {

/**
 * @struct DlaParams
 * @brief Grid, seed and growth budget.
 */
struct DlaParams {
    core::u32 width{64u};          ///< Map width.
    core::u32 depth{64u};          ///< Map depth.
    core::u32 seed{1337u};         ///< Determinism anchor.
    core::u32 particles{600u};     ///< Particles to release.
    core::u32 maxStepsPerParticle{2000u}; ///< Walk budget before a particle is abandoned.
    core::u32 spawnMargin{3u};     ///< Cells beyond the cluster's extent a particle spawns.
    core::u32 thickness{0u};       ///< Radius carved around each stuck particle; 0 keeps the fractal thin.
};

/**
 * @struct DlaReport
 * @brief What the growth achieved.
 */
struct DlaReport {
    core::u32 stuck{0u};      ///< Particles that found the cluster.
    core::u32 abandoned{0u};  ///< Particles that exhausted their budget.
    core::u32 openCells{0u};  ///< Cells the cluster occupies.
    core::u32 extent{0u};     ///< Largest distance from the seed reached.
};

/**
 * @brief Grows a dendritic cave by diffusion-limited aggregation.
 * @param params    Grid, seed and budgets.
 * @param outReport Receives the growth statistics (may be null).
 * @return The map, with the cluster as floor and everything else as rock.
 */
[[nodiscard]] DungeonMap generateDlaCave(const DlaParams &params, DlaReport *outReport = nullptr);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_AGGREGATION_HPP
