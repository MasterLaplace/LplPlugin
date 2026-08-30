/**
 * @file TerrainRoutes.hpp
 * @brief Where a road actually runs, answered from the relief.
 *
 * @warning **The first consumer `procgen::routeLeastCost` has ever had**, and now the first
 * `procgen::routeAcrossWorld` has too. Both were written, documented -- "the roads a grammar cannot
 * draw" -- and reachable by nothing a world runs, which is the orphan this repository keeps
 * finding. A journey is the caller they were waiting for.
 *
 * @warning Correcting this file's own first claim: it said `connectPlaces` had no caller either.
 * That was false when written -- `WorldBuilder::roads` has called it since 2026-07-28, a month
 * earlier. Only the point-to-point router was the orphan, and a grep that reported two of them
 * was a grep that had not been read.
 *
 * **This is the "deterministic fill" half, and only that half.** The corpus says WHICH places
 * were connected -- @ref history::IPlaceResolver::linkedPlaces, evidence no simulation may
 * invent. Nothing records which valley the road took, so a generator is entitled to decide it,
 * and paying a cost per cell is what makes the answer follow a valley instead of cutting
 * straight through a ridge.
 *
 * @warning **`route()` is PURE, and the network is painted ONCE.** The tempting design is to paint each
 * route as it is planned, so later ones inherit `reuseDiscount` and converge on a trunk -- which
 * is how a real network grows, and is what `connectPlaces` does. It is wrong here: the answer
 * would then depend on the order in which bodies happened to ask, so the same question asked
 * twice could give two roads, and a fold would record the tick order rather than the terrain.
 * Instead @ref paveAttested lays down the ATTESTED network first, from the corpus, and every
 * traveller afterwards reads that trunk for free. Facts first, literally: the roads that exist
 * are the ones a source says existed.
 *
 * @warning Planning ALLOCATES (the search's frontier and the path). It happens once when a body picks a
 * destination, never per tick -- which is also why the walk holds a goal instead of re-deciding
 * one every frame.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_TERRAINROUTES_HPP
#    define LPL_ENGINE_SYSTEMS_TERRAINROUTES_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/Place.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Routing.hpp>

namespace lpl::engine::systems {

/**
 * @struct TerrainRouteParams
 * @brief How the routing grid maps onto the world, and what a road pays.
 */
struct TerrainRouteParams {
    /**
     * World units per routing cell.
     *
     * @warning An INTEGER, so the mapping is a shift and a subtraction rather than a Fixed32 division:
     * a division rounds, and a rounded cell index does not lose a fraction of a unit, it puts
     * the endpoint of a road in the wrong cell. Coarser than the world is legitimate -- a
     * continental route does not need metre resolution -- and 1 means one cell per unit.
     */
    core::u32 cellSize{1u};

    /**
     * Fine cells per coarse cell when the grid is too large to search flat, or zero to search it
     * flat.
     *
     * @warning **What makes a planetary road findable at all.** A* keeps cost, parent and a settled
     * flag per cell -- about twelve bytes -- so a global grid at thirty-metre cells is roughly ten
     * TERABYTES of search state, and even a grid it can hold costs expansions proportional to the
     * area between the endpoints rather than the length of the road. Above one, the route is
     * planned on a summary and refined inside the corridor that plan opens: measured on a 256-cell
     * grid, the SAME road for 8 493 expansions instead of 25 399.
     *
     * @warning The summary is DERIVED from the bound field, never supplied. A caller-provided coarse
     * field is a second description of the ground, free to disagree with the first -- and a plan
     * made on a ridge the fine field does not have opens a corridor around nothing.
     */
    core::u32 coarseRatio{0u};

    /**
     * Coarse cells of slack painted either side of the plan.
     *
     * @warning Slack is what lets the fine search improve on the plan instead of merely tracing it.
     * At zero the corridor is the plan's own cells, so refinement can only follow a road drawn on
     * averages; the failure is not a wrong road but a road with a coarse cell's staircase in it.
     */
    core::u32 corridorMargin{1u};

    /// Cost model handed to @ref procgen::routeLeastCost.
    procgen::RoutingParams cost{};
};

/**
 * @class TerrainRoutes
 * @brief @ref history::IRouteResolver answered by least-cost search over a heightfield.
 *
 * @warning World coordinates are the grid CENTRED ON THE ORIGIN, which is the same convention
 * @ref engine::GridTerrain uses for the same field. A second convention would be a road that runs
 * beside the ground it was routed on -- correct in both files and wrong between them.
 */
class TerrainRoutes final : public history::IRouteResolver {
public:
    /**
     * @brief Binds the relief and the gazetteer.
     *
     * Both are held by POINTER and must outlive this: a terrain is large, and a copy taken here
     * would answer for a world that has since been eroded, raised or streamed away.
     *
     * @param field  The relief the roads cross.
     * @param places Where the endpoints are.
     * @param params Grid mapping and cost model.
     */
    void bind(const procgen::Heightfield &field, const history::IPlaceResolver &places,
              const TerrainRouteParams &params);

    /**
     * @brief Lays down the roads a corpus attests.
     *
     * @warning Each route is painted BEFORE the next is planned, so the network grows along its own
     * trunk -- the second road between two regions prefers the first road's ground, exactly as
     * real ones do. That makes the ORDER part of the result, so it is fixed: the caller's place
     * order, then each place's attested links in the order the resolver gives them. Not sorted
     * by cost, and deliberately so -- unlike `connectPlaces`, which SELECTS a spanning subset
     * and must therefore rank, this lays down every attested link. The set is decided by the
     * corpus; ordering only decides which road gets to be the one the others bend toward.
     *
     * A pair is routed ONCE, in the direction (lower identifier -> higher). Routing it twice
     * would give two roads between the same two places, because the second call would see the
     * first one's discount and take it.
     *
     * @param places     Identifiers whose attested links should become roads.
     * @param placeCount How many.
     * @return Cells painted.
     */
    core::u32 paveAttested(const core::u32 *places, core::u32 placeCount);

    /**
     * @brief Plans the way from one place to another.
     *
     * @warning **Waypoints are the road's CORNERS, and they are STRIDED when they do not fit -- never
     * truncated.** Truncation keeps the beginning of the road and drops the end, so a body walks
     * the first third of it and then goes straight across precisely the terrain the route
     * existed to avoid. A strided set of corners is a coarser road with the same shape and the
     * same destination.
     *
     * @param fromPlace Where the body is.
     * @param toPlace   Where it is going.
     * @param out       Receives the waypoints, in order, excluding the start.
     * @param capacity  Room in @p out.
     * @return How many were written; zero means "go straight", which is the honest answer when
     *         either endpoint is off the grid or the search found nothing.
     */
    [[nodiscard]] core::u32 route(core::u32 fromPlace, core::u32 toPlace, history::RouteLeg *out,
                                  core::u32 capacity) const override;

    /**
     * @brief The road mask, for a caller that wants to draw it or hand it to another pass.
     * @return The grid, with @c 0 for no road and @c 1 for a road.
     */
    [[nodiscard]] const procgen::Grid<core::u8> &roads() const noexcept { return _roads; }

    /**
     * @brief Routes planned since @ref bind. Diagnostics, hence `mutable` behind a const @ref route.
     * @return The number of routes planned.
     */
    [[nodiscard]] core::u32 planned() const noexcept { return _planned; }

    /**
     * @brief Routes whose goal the search could not reach, or whose budget ran out.
     * @return The number of unreachable routes.
     */
    [[nodiscard]] core::u32 unreachable() const noexcept { return _unreachable; }

    /**
     * @brief Cells the searches settled, in total.
     *
     * @warning Reported because it is the COST, and a route that is cheap to walk can be expensive to
     * find. A caller that sees this climb has a budget to set (`RoutingParams::maxExpansions`)
     * rather than a mystery to profile.
     * @return The number of cells the searches settled.
     */
    [[nodiscard]] core::u32 expanded() const noexcept { return _expanded; }

    /**
     * @brief Cells painted by @ref paveAttested.
     * @return The number of cells painted.
     */
    [[nodiscard]] core::u32 paved() const noexcept { return _paved; }

    /**
     * @brief Cells the COARSE plans settled, in total; zero when routing flat.
     *
     * @warning Reported beside @ref expanded rather than folded into it, because the two are the
     * halves of the trade the cascade makes: the coarse number is what the cascade costs and the
     * fine number is what it saves. One total would hide a summary so coarse it plans badly behind
     * a fine search that then has to work.
     *
     * @return The number of cells settled at the coarse level.
     */
    [[nodiscard]] core::u32 coarseExpanded() const noexcept { return _coarseExpanded; }

    /**
     * @brief Fine cells the coarse plans opened, in total; zero when routing flat.
     * @return The number of cells opened.
     */
    [[nodiscard]] core::u32 corridorCells() const noexcept { return _corridorCells; }

    /**
     * @brief Routes whose corridor had to be widened before the fine search got through.
     *
     * @warning A coarse cell is an AVERAGE, so a plan may cross a strait that is water at full
     * resolution. Counted because a world where this climbs has a summary too coarse to plan on --
     * which is a tuning fact nothing else reports, and the alternative to reporting it is a router
     * that silently costs twice what it should.
     *
     * @return The number of widened routes.
     */
    [[nodiscard]] core::u32 widened() const noexcept { return _widened; }

    /**
     * @brief The summary the plans are made on, empty when routing flat.
     * @return The coarse field.
     */
    [[nodiscard]] const procgen::Heightfield &coarse() const noexcept { return _coarse; }

    /**
     * @brief Distinct pairs @ref paveAttested laid a road between.
     *
     * @warning Reported because it is what shows the deduplication working: a resolver hands out
     * every link in both directions, so a network of two attested roads arrives as four
     * directed links, and a count of four would mean each road was laid twice -- the second
     * time taking its own discount, which is a parallel road nobody built.
     *
     * @return The number of pairs.
     */
    [[nodiscard]] core::u32 pairs() const noexcept { return _pairs; }

private:
    /**
     * @brief World units to a cell of the routing grid.
     *
     * @warning Floors via an arithmetic shift rather than @c toInt(), which truncates toward zero: with
     * truncation the two cells either side of the origin are half-width, so a body crossing zero
     * skips one.
     *
     * @param x    World X.
     * @param z    World Z.
     * @param outX Receives the column.
     * @param outZ Receives the row.
     * @return false when the position is off the grid.
     */
    [[nodiscard]] bool toCell(math::Fixed32 x, math::Fixed32 z, core::u32 &outX, core::u32 &outZ) const;

    /**
     * @brief The world position of a cell's CENTRE.
     *
     * Its centre rather than its corner: a waypoint on the corner of a cell sits on the boundary
     * between it and three others, and a body aiming at it is aiming at the one place where the
     * ground on either side disagrees.
     *
     * @param cellX Column.
     * @param cellZ Row.
     * @param outX  Receives world X.
     * @param outZ  Receives world Z.
     */
    void toWorld(core::u32 cellX, core::u32 cellZ, math::Fixed32 &outX, math::Fixed32 &outZ) const;

    /**
     * @brief The cell a place stands in.
     *
     * @param place The identifier.
     * @param outX  Receives the column.
     * @param outZ  Receives the row.
     * @return false when the gazetteer does not carry it, does not know where it is, or it falls
     *         off the routing grid.
     */
    [[nodiscard]] bool cellOf(core::u32 place, core::u32 &outX, core::u32 &outZ) const;

    /**
     * @brief One route, flat or cascaded, with the counters kept in one place.
     *
     * @warning Both callers go through here so neither can route by a rule the other does not.
     * @ref paveAttested and @ref route ask the same question of the same terrain, and the day one
     * of them cascaded and the other did not, the network a body reads would be laid on a
     * different search than the one it walks.
     *
     * @param startX Start column.
     * @param startZ Start row.
     * @param goalX  Goal column.
     * @param goalZ  Goal row.
     * @return The path; @c found is false when the goal is unreachable.
     */
    [[nodiscard]] procgen::RoutedPath plan(core::u32 startX, core::u32 startZ, core::u32 goalX, core::u32 goalZ) const;

private:
    const procgen::Heightfield *_field{nullptr};
    const history::IPlaceResolver *_places{nullptr};
    TerrainRouteParams _params{};
    procgen::Heightfield _coarse;
    procgen::Grid<core::u8> _roads;
    core::u32 _paved{0u};
    core::u32 _pairs{0u};
    mutable core::u32 _planned{0u};
    mutable core::u32 _unreachable{0u};
    mutable core::u32 _expanded{0u};
    mutable core::u32 _coarseExpanded{0u};
    mutable core::u32 _corridorCells{0u};
    mutable core::u32 _widened{0u};
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_TERRAINROUTES_HPP
