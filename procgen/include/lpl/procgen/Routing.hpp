/**
 * @file Routing.hpp
 * @brief Least-cost paths across terrain: the roads a grammar cannot draw.
 *
 * A grammar gives a road network its texture — the way streets branch, the angle
 * they meet at, how a district reads. What it cannot give is the one thing a road
 * is actually for: getting from somewhere to somewhere else. A rewrite rule has
 * no destination, so a purely grammatical network connects the places it happens
 * to reach and no others.
 *
 * So the arterial layer is routed rather than grown. The cost of crossing a cell
 * is not its distance but what it takes to build there: climbing is expensive,
 * water is nearly prohibitive, and an existing road is nearly free. That last
 * term is what makes successive routes converge into a network instead of
 * accumulating as parallel lines — the second road between two towns prefers the
 * first road's ground, exactly as real ones do.
 *
 * A\* rather than plain Dijkstra, with the admissible heuristic that no cell can
 * cost less than @ref RoutingParams::baseCost: the search then expands a corridor
 * between the endpoints rather than a disc around the start. The tie-break is by
 * cell index, never by insertion order, so the path is a function of the terrain
 * and nothing else.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_ROUTING_HPP
#    define LPL_PROCGEN_ROUTING_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/**
 * @struct RoutingParams
 * @brief What a road pays to cross a cell.
 *
 * Costs are in the same arbitrary unit; only their ratios matter. They are
 * Fixed32 throughout, so a route is authoritative state like everything else
 * here — two targets routing the same terrain take the same road.
 */
struct RoutingParams {
    core::f32 baseCost{1.0f};      ///< Cost of one flat, empty cell.
    core::f32 slopePenalty{6.0f};  ///< Extra cost per unit of height climbed or dropped.
    core::f32 waterPenalty{40.0f}; ///< Extra cost for a cell at or below @ref waterLevel.
    core::f32 waterLevel{0.0f};    ///< Height at or below which a cell counts as water.
    core::f32 reuseDiscount{0.8f}; ///< Share of the base cost waived on an existing road, in [0, 1].
    core::u32 maxExpansions{0u};   ///< Search budget; 0 means the whole grid.
};

/**
 * @struct RoutedPath
 * @brief One route, as the cells it runs through.
 */
struct RoutedPath {
    lpl::pmr::vector<core::u32> cells; ///< Cell indices from start to goal, inclusive.
    core::u32 expanded{0u};            ///< Cells the search settled (its cost).
    math::Fixed32 cost{};              ///< Total cost of the route.
    bool found{false};                 ///< Did the goal turn out to be reachable?
};

/**
 * @brief Routes the cheapest road from one cell to another.
 *
 * @param field    Terrain the road crosses.
 * @param existing Optional 0/1 mask of ground that is already road; cells marked
 *                 here are cheaper, which is what merges routes into a network.
 *                 May be null or empty.
 * @param startX   Start column.
 * @param startZ   Start row.
 * @param goalX    Goal column.
 * @param goalZ    Goal row.
 * @param params   Cost model.
 * @return The path; @c found is false when the goal is unreachable or the search
 *         budget ran out.
 */
[[nodiscard]] RoutedPath routeLeastCost(const Heightfield &field, const Grid<core::u8> *existing, core::u32 startX,
                                        core::u32 startZ, core::u32 goalX, core::u32 goalZ,
                                        const RoutingParams &params);

/**
 * @brief Connects a set of places with roads, cheapest link first.
 *
 * A minimum-spanning-tree shape rather than every pair: N places joined by N-1
 * roads is a network, and joining every pair is a lattice nobody builds. Each
 * accepted route is painted into @p roads before the next is planned, so later
 * routes inherit the discount and the network grows along its own trunk.
 *
 * @param field  Terrain the roads cross.
 * @param places Cell indices to connect (fewer than two is a no-op).
 * @param params Cost model.
 * @param roads  Grid to paint into; cells on a route are set to 1.
 * @return Number of cells painted.
 */
core::u32 connectPlaces(const Heightfield &field, const lpl::pmr::vector<core::u32> &places,
                        const RoutingParams &params, Grid<core::u8> &roads);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_ROUTING_HPP
