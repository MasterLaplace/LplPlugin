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
 * A\* rather than plain Dijkstra, with the admissible heuristic that no step can
 * cost less than @ref RoutingParams::baseCost less its share waived by
 * @ref RoutingParams::reuseDiscount: the search then expands a corridor between the
 * endpoints rather than a disc around the start. The tie-break is by cell index,
 * never by insertion order, so the path is a function of the terrain and nothing
 * else.
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
    core::f32 baseCost{1.0f};      /**< Cost of one flat, empty cell. */
    core::f32 slopePenalty{6.0f};  /**< Extra cost per unit of height climbed or dropped. */
    core::f32 waterPenalty{40.0f}; /**< Extra cost for a cell at or below @ref waterLevel. */
    core::f32 waterLevel{0.0f};    /**< Height at or below which a cell counts as water. */
    core::f32 reuseDiscount{0.8f}; /**< Share of the base cost waived on an existing road, in [0, 1]. */
    core::u32 maxExpansions{0u};   /**< Search budget; 0 means the whole grid. */

    /**
     * Columns the world wraps at, or zero for a grid with real edges.
     *
     * @warning **A closed world without this is worse than an open one.** Two places either side of
     * the antimeridian are neighbours; a router that cannot step across it lays a road all the way
     * round the planet instead -- and that road is a perfectly valid, perfectly expensive route, so
     * nothing downstream can tell it was the wrong one.
     *
     * @warning **The heuristic depends on it, and that is the dangerous half.** A* only returns the
     * cheapest road while its heuristic never OVERESTIMATES. Near the seam an unwrapped distance
     * overestimates enormously -- 990 cells where the real gap is 10 -- so the search discards the
     * short way before evaluating it and returns a plausible road instead of the best one, in
     * silence. This file has already been bitten once by exactly that, when the cheapest step was
     * stated as `base` and `reuseDiscount` falsified it.
     *
     * @warning East-west only. A pole crossing, a real shortcut between two high-latitude places on
     * opposite meridians, is @ref wrapPoles' to offer, with the heuristic that stays admissible
     * across it.
     *
     * Must equal the grid's width when set: the grid IS the circumference, which is why a global
     * route is planned at a coarse level. At 30 km a cell the whole earth is 1336 by 667.
     */
    core::u32 wrapColumns{0u};

    /**
     * Whether a road may cross a pole.
     *
     * @warning **A pole crossing is a real shortcut and it is NOT a wrap.** Going north off the top of
     * the grid does not arrive at the bottom -- that is a torus, and it would join the Arctic to the
     * Antarctic. On a sphere you come back at the SAME edge, half a world away in longitude, walking
     * south again. So the step from row zero northward lands on row zero at column x + columns/2.
     *
     * @warning **The heuristic is the dangerous half, and it is why this was left out until now.** A*
     * returns the cheapest road only while its heuristic never overestimates, and offering a
     * shortcut in the graph without teaching the heuristic about it makes the estimate too LARGE for
     * routes that use it -- so the search thrashes and, worse, can settle for a route that is merely
     * plausible. @ref routeLeastCost therefore estimates the minimum of three route classes: the
     * direct one, over the north pole, and over the south. Each is a lower bound on its class, so
     * their minimum is a lower bound on every path, which is exactly what admissibility asks.
     *
     * Requires @ref wrapColumns; a pole makes no sense on a grid that does not close east-west: crossing
     * one turns the longitude by half a world, and half of nothing is nothing.
     */
    bool wrapPoles{false};

    /**
     * Cells the search is allowed into, or null for the whole grid.
     *
     * @warning **What lets a fine search follow a coarse plan without exploring a planet.** A* on a
     * global grid at thirty-metre cells needs about ten TERABYTES of state; at eight kilometres it
     * needs a hundred and sixty megabytes. So a route across a world is planned coarse and refined
     * fine, and refinement is only affordable if the fine search is confined to the corridor the
     * coarse one found. Same cascade `EndlessRiverParams` already uses for trunk rivers: ask the
     * question twice at two scales and let the coarse answer constrain the fine one.
     *
     * @warning **A corridor is a constraint on the SEARCH, never a claim about the ground.** A cell
     * outside it is not impassable, it is merely not looked at -- so a corridor that is wrong makes
     * the search FAIL rather than lie, and `RoutedPath::found` says so. That distinction is the
     * whole reason this is a separate field and not a cost: a very large cost would let the router
     * leave the corridor when it felt like it and return a road nobody planned, which is the
     * failure a corridor exists to make impossible.
     *
     * Must match the grid's dimensions when set; a mismatched one is ignored rather than half
     * applied, for the reason @ref wrapColumns is.
     */
    const Grid<core::u8> *corridor{nullptr};
};

/**
 * @struct RoutedPath
 * @brief One route, as the cells it runs through.
 */
struct RoutedPath {
    lpl::pmr::vector<core::u32> cells; /**< Cell indices from start to goal, inclusive. */
    core::u32 expanded{0u};            /**< Cells the search settled (its cost). */
    math::Fixed32 cost{};              /**< Total cost of the route. */
    bool found{false};                 /**< Did the goal turn out to be reachable? */
};

/**
 * @struct HierarchicalRoute
 * @brief A road planned coarse and walked fine, with what each half cost.
 */
struct HierarchicalRoute {
    RoutedPath fine{};            /**< The road itself, in fine cells. */
    core::u32 coarseExpanded{0u}; /**< Cells the coarse plan settled. */
    core::u32 corridorCells{0u};  /**< Fine cells the coarse plan opened. */
    bool coarseFound{false};      /**< Whether a coarse plan existed at all. */
    /**
     * The corridor had to be widened before the fine search got through.
     *
     * @warning Reported rather than hidden. A coarse cell is an AVERAGE, so a strait that a coarse
     * plan calls land can be water at full resolution -- the classic failure of planning on a
     * summary. Widening once is the cheap recovery; that it happened is worth knowing, because a
     * world where it happens often has a coarse level too coarse to plan on.
     */
    bool widened{false};
};

/**
 * @brief Summarises a heightfield into one cell per block, to plan a route on.
 *
 * @warning **The parameter @ref routeAcrossWorld could not previously be given.** The cascade
 * needs a coarse field and this module offered no way to make one, so the only caller it ever
 * had was its own test -- which built the summary by hand. A function whose argument the module
 * cannot produce is a function nothing will call.
 *
 * @warning **The MEAN, and the alternative loses in a way worth recording.** The obvious rival is
 * the minimum -- "the cheapest ground in the block", which sounds like the optimistic summary a
 * corridor wants. It erases thin barriers: a one-cell ridge inside a four-by-four block leaves
 * twelve cells at valley height, so the minimum reports valley and the coarse plan crosses the
 * wall wherever it likes, opening a corridor that does not contain the pass. The average keeps
 * the wall visible and still reads the gap as cheaper, which is the whole job. Measured in
 * test-terrain-routes.
 *
 * @warning Not a second answer to `harvest::reduceTile`, which reduces raw metres with a void
 * sentinel on the host and never enters ring 0. This reduces authoritative Fixed32 and is linked
 * into the kernel; unifying them would drag a survey's missing-data rule into a router.
 *
 * The block that runs off the edge is averaged over the cells that exist, so the summary is
 * `ceil(width / ratio)` wide -- which is the size @ref routeAcrossWorld indexes it at. Ceiling, so
 * the block that runs off the edge still gets a cell: dropping it would make the summary describe a
 * smaller world than the one being routed, and the goal near the far edge would have no coarse cell
 * to plan toward.
 *
 * The mean floors rather than truncating toward zero, so a block of sea and a block of hill round
 * the same way. Plain integer arithmetic either way, so both targets agree.
 *
 * @param field The full-resolution terrain.
 * @param ratio Fine cells per coarse cell, on each axis. Zero or one returns a copy.
 * @return The summary.
 */
[[nodiscard]] Heightfield reduceHeightfield(const Heightfield &field, core::u32 ratio);

/**
 * @brief Routes across a world too large to search flat.
 *
 * @warning **The measurement that forces this shape.** A* keeps cost, parent and a settled flag per
 * cell -- about twelve bytes. On a global grid at thirty-metre cells that is 890 733 444 400 cells
 * and roughly ten TERABYTES; at 7.7 km it is 163 MB, at 31 km it is 10 MB. A planet cannot be
 * searched flat, so it is searched twice: coarse for the shape, fine for the ground.
 *
 * @warning **The coarse plan is a CORRIDOR, not a commitment.** A coarse cell is an average, so a
 * plan may cross a strait that does not exist at full resolution. The fine search is confined to
 * the corridor but free inside it, and when it cannot get through the corridor is widened once and
 * retried -- after which failure is REPORTED. A refinement that quietly returned the coarse plan
 * instead would hand back a road through water, and it would look like every other road.
 *
 * The coarse plan's own params carry no corridor -- there is nothing above it to be confined by --
 * and its wrap must be the coarse grid's, not the fine one's, or the heuristic folds at a width the
 * grid does not have and stops being admissible. When no coarse plan exists there is no shape to
 * refine: that is reported in `coarseFound`, never papered over with a flat search.
 *
 * @param coarse     The summary field the plan is made on.
 * @param fine       The full-resolution field the road is walked on.
 * @param cellRatio  Fine cells per coarse cell, on each axis. Zero or one makes this a plain route.
 * @param existing   Optional 0/1 mask of ground that is already road, in FINE cells.
 * @param startX     Start column, in fine cells.
 * @param startZ     Start row, in fine cells.
 * @param goalX      Goal column, in fine cells.
 * @param goalZ      Goal row, in fine cells.
 * @param params     Cost model. Its `corridor` is replaced by the one planned here.
 * @param margin     Coarse cells of slack painted either side of the plan.
 * @return The road and what finding it cost.
 */
[[nodiscard]] HierarchicalRoute routeAcrossWorld(const Heightfield &coarse, const Heightfield &fine,
                                                 core::u32 cellRatio, const Grid<core::u8> *existing, core::u32 startX,
                                                 core::u32 startZ, core::u32 goalX, core::u32 goalZ,
                                                 const RoutingParams &params, core::u32 margin = 1u);

/**
 * @brief Routes the cheapest road from one cell to another.
 *
 * The heuristic must never overestimate, or A* stops returning the cheapest road and starts
 * returning a plausible one. Chebyshev distance times the cheapest possible cell is the largest
 * value that still cannot: no route can reach the goal in fewer steps, and no step can cost less
 * than this.
 *
 * @warning The cheapest step is `base` MINUS the reuse discount, and getting that wrong made the two
 * halves of this file contradict each other. The bound used to be `base` on the stated grounds that
 * "no step can cost less than base" -- which `reuseDiscount` is precisely designed to falsify, since
 * entering an existing road costs base * (1 - discount). The heuristic therefore overestimated
 * exactly when a road was available, so the search discarded the merge before evaluating it:
 * measured on two parallel attested roads two cells apart, both were laid in full (82 cells
 * painted, 82 cells expanded) at every discount from 0 to 1. The feature documented as "the whole
 * reason routes merge into a network instead of piling up as parallel lines" did nothing at all,
 * and nothing noticed because the pass had no caller.
 *
 * @warning On a closed world the column distance is the SHORTER way round, and that fold is what
 * decides whether a wrapped route is correct or merely plausible. An unwrapped distance near the
 * seam overestimates by nearly a circumference, so the heuristic stops being admissible exactly
 * where the answer matters and A* discards the short way unevaluated.
 *
 * With @ref RoutingParams::wrapPoles the estimate is the minimum of the three route classes that
 * member describes. Crossing a pole lands half a world away in longitude, so the horizontal work
 * of a pole route is measured from there. North: up to row zero, one step over, then down to the
 * goal; south is the same through the last row. Steps are diagonal, so the bound is the larger of
 * the vertical and horizontal totals, never their sum.
 *
 * Distance: a diagonal is longer than an orthogonal step, and paying the same for both is what
 * makes a router produce staircases. sqrt(2) as 2/sqrt(2), because the module already keeps
 * 1/sqrt(2) exactly and a second constant is a second thing to get wrong.
 *
 * Climbing: what actually decides where a road goes. Absolute, so a descent is as expensive as a
 * climb — a road cut into a hillside pays for the earth it moves either way.
 *
 * An existing road is cheap ground. This is the whole reason routes merge into a network instead
 * of piling up as parallel lines.
 *
 * A cell popped after it was settled is skipped: a cheaper route to it was settled first (lazy
 * deletion).
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
 * Grow one tree: the first place is in, and each round attaches whichever outsider is cheapest to
 * reach from anything already connected. Prim's shape, and the reason it is not Kruskal is that the
 * cost of an edge here CHANGES once a road exists — so edges must be priced against the network as
 * it stands, not once at the start.
 *
 * One search per round, seeded from EVERY connected place at zero cost, and stopped at the first
 * unconnected place it settles. The obvious way to write this is a pair loop — price every
 * (connected, unconnected) pair with its own A*, keep the cheapest — and that is what it used to
 * do. It is O(P^3) least-cost searches over the whole grid, which does not show up on the small
 * maps the tests use and is catastrophic on a real one: a 128x128 world with 36 districts spent
 * ELEVEN SECONDS here, more than the whole rest of generation put together, and it was only
 * visible once the viewer had to rebuild a world on a keypress. Seeding the frontier with the
 * whole connected set answers the same question — what is the cheapest link between the network
 * and anything outside it — in one sweep instead of P^2.
 *
 * The multi-source form is also the more faithful one. Pricing pairs measures the distance between
 * two SITES; pricing from the frontier measures it from the network as it actually stands, roads
 * already painted included, which is what the reuse discount was for in the first place.
 *
 * No heuristic: with many sources and many possible goals there is no single target to estimate
 * toward, and an inadmissible guess would buy speed by returning a road that is merely plausible.
 * Dijkstra is the honest form of this search. Each route is painted back from the place reached to
 * the source it grew from, the only cells at zero cost; when nothing left is reachable the rounds
 * stop, and the network is of what can be joined.
 *
 * @warning The SAME east-west wrap as @ref routeLeastCost. This file has already been bitten by its
 * two halves disagreeing about the cost model; letting only one of them cross the seam would be the
 * same fault in a new place.
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
