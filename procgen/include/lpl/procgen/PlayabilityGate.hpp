/**
 * @file PlayabilityGate.hpp
 * @brief Deterministic reachability check over a generated cost grid (Dijkstra).
 *
 * A procedural world is only worth materialising if it is actually playable — a
 * start that can reach a goal. This gate rasterises the same fBm field the
 * generators use into a 4-connected cost grid (cells above @c wallThreshold are
 * impassable walls) and runs Dijkstra in Fixed32 to decide whether the goal is
 * reachable and at what least-cost. All-integer, no @c sqrt, no heap: the verdict
 * is bit-identical across the Linux oracle and the i686 kernel, so the AI/editor
 * can reject an unplayable seed before ever building the world.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_PLAYABILITYGATE_HPP
#    define LPL_PROCGEN_PLAYABILITYGATE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::procgen {

/**
 * @struct PlayabilityParams
 * @brief Grid + endpoints + wall threshold for the reachability check.
 */
struct PlayabilityParams {
    core::u32 seed{1337u};        ///< Field seed (matches the generator's).
    core::u32 cols{32u};          ///< Grid columns.
    core::u32 rows{32u};          ///< Grid rows.
    core::f32 noiseScale{0.15f};  ///< fBm sampling frequency.
    core::u32 octaves{4u};        ///< fBm octaves.
    core::f32 wallThreshold{0.5f};///< fBm value (in [-1,1]) at/above which a cell is a wall.
    core::u32 startCol{0u};       ///< Start cell column.
    core::u32 startRow{0u};       ///< Start cell row.
    core::u32 goalCol{31u};       ///< Goal cell column.
    core::u32 goalRow{31u};       ///< Goal cell row.
};

/**
 * @struct PlayabilityResult
 * @brief Verdict of a reachability check.
 */
struct PlayabilityResult {
    bool reachable{false};                    ///< Goal reachable from start?
    core::u32 visited{0u};                    ///< Cells expanded (search size).
    math::Fixed32 pathCost{math::Fixed32::zero()}; ///< Least-cost start→goal (0 if unreachable).
};

/**
 * @brief Runs Dijkstra over the fBm-derived cost grid described by @p params.
 * @param params Grid, endpoints, wall threshold.
 * @return Reachability verdict (deterministic in Fixed32).
 */
[[nodiscard]] PlayabilityResult evaluateReachability(const PlayabilityParams &params);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_PLAYABILITYGATE_HPP
