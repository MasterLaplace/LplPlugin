/**
 * @file PoissonScatter.hpp
 * @brief Deterministic blue-noise scatter (Poisson-disk) into an ECS world.
 *
 * Places entities so that no two are closer than a minimum radius — the natural
 * distribution for props, trees, rocks, spawn points. Uses the grid-jitter
 * variant of Poisson-disk sampling entirely in Fixed32: one jittered candidate
 * per background-grid cell, accepted only if it clears every neighbour by the
 * squared distance (no @c sqrt, so it stays on the determinism contract and is
 * bit-identical Linux oracle ↔ i686 kernel). Same seed ⇒ same layout.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_POISSONSCATTER_HPP
#    define LPL_PROCGEN_POISSONSCATTER_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::ecs {
class Registry;
}

namespace lpl::procgen {

/**
 * @struct PoissonScatterParams
 * @brief High-level scatter knobs an editor / AI would set (token-cheap).
 */
struct PoissonScatterParams {
    core::u32 seed{4242u};    ///< Layout seed (determinism anchor).
    core::f32 width{16.0f};   ///< Region extent along X (world units).
    core::f32 depth{16.0f};   ///< Region extent along Z (world units).
    core::f32 radius{1.5f};   ///< Minimum distance between two points.
    core::f32 propHalf{0.2f}; ///< Prop half-extent (AABB).
    core::u32 maxPoints{0u};  ///< Hard cap (0 = no cap beyond grid capacity).
};

/**
 * @brief Scatters prop entities (Position + AABB, Fixed32) into @p registry.
 * @param registry Destination world.
 * @param params   Scatter parameters.
 * @return Number of entities created.
 */
core::u32 scatterPoisson(ecs::Registry &registry, const PoissonScatterParams &params);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_POISSONSCATTER_HPP
