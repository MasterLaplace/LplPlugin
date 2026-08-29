/**
 * @file Routing.cpp
 * @brief Implementation of terrain-aware least-cost routing.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Routing.hpp>

#include <lpl/math/FixedMath.hpp>

namespace lpl::procgen {

namespace {

/// Sentinel for "no predecessor" and "not reached".
constexpr core::u32 kNoCell = 0xFFFFFFFFu;

/// A cell waiting its turn, keyed by cost-so-far plus heuristic.
struct SearchEntry {
    core::i32 priority; ///< Raw Q16.16 of f = g + h.
    core::u32 cell;     ///< Cell index.
};

/**
 * @brief Total order on search entries: cheapest first, then lowest index.
 *
 * The index tiebreak is the determinism contract, not a refinement. Equal-cost
 * frontiers are the common case on flat ground, and a heap that resolved them by
 * insertion history would let two targets settle a different cell first and
 * return two different roads of identical cost.
 */
constexpr bool cheaper(const SearchEntry &a, const SearchEntry &b) noexcept
{
    return a.priority != b.priority ? a.priority < b.priority : a.cell < b.cell;
}

/// A binary min-heap over @ref SearchEntry; std::priority_queue is not freestanding.
class SearchHeap {
public:
    void reserve(core::usize capacity) { _items.reserve(capacity); }
    [[nodiscard]] bool empty() const { return _items.empty(); }

    void push(SearchEntry entry)
    {
        _items.push_back(entry);
        core::u32 child = static_cast<core::u32>(_items.size()) - 1u;
        while (child != 0u)
        {
            const core::u32 parent = (child - 1u) / 2u;
            if (!cheaper(_items[child], _items[parent]))
                break;
            const SearchEntry swap = _items[parent];
            _items[parent] = _items[child];
            _items[child] = swap;
            child = parent;
        }
    }

    [[nodiscard]] SearchEntry pop()
    {
        const SearchEntry top = _items[0];
        _items[0] = _items[_items.size() - 1u];
        _items.pop_back();

        const core::u32 count = static_cast<core::u32>(_items.size());
        core::u32 node = 0u;
        for (;;)
        {
            const core::u32 left = node * 2u + 1u;
            const core::u32 right = left + 1u;
            core::u32 best = node;
            if (left < count && cheaper(_items[left], _items[best]))
                best = left;
            if (right < count && cheaper(_items[right], _items[best]))
                best = right;
            if (best == node)
                break;
            const SearchEntry swap = _items[best];
            _items[best] = _items[node];
            _items[node] = swap;
            node = best;
        }
        return top;
    }

private:
    lpl::pmr::vector<SearchEntry> _items;
};

} // namespace

RoutedPath routeLeastCost(const Heightfield &field, const Grid<core::u8> *existing, core::u32 startX, core::u32 startZ,
                          core::u32 goalX, core::u32 goalZ, const RoutingParams &params)
{
    RoutedPath path;
    if (field.empty() || !field.contains(static_cast<core::i32>(startX), static_cast<core::i32>(startZ)) ||
        !field.contains(static_cast<core::i32>(goalX), static_cast<core::i32>(goalZ)))
        return path;

    const core::u32 width = field.width();
    const core::u32 cells = field.cellCount();
    const core::u32 start = field.index(startX, startZ);
    const core::u32 goal = field.index(goalX, goalZ);

    const math::Fixed32 base = math::Fixed32::fromFloat(params.baseCost);
    const math::Fixed32 slopeCost = math::Fixed32::fromFloat(params.slopePenalty);
    const math::Fixed32 waterCost = math::Fixed32::fromFloat(params.waterPenalty);
    const math::Fixed32 waterLevel = math::Fixed32::fromFloat(params.waterLevel);
    const math::Fixed32 discount = math::Fixed32::fromFloat(params.reuseDiscount < 0.0f ? 0.0f :
                                                            params.reuseDiscount > 1.0f ? 1.0f :
                                                                                          params.reuseDiscount);
    const bool hasExisting =
        existing != nullptr && existing->width() == field.width() && existing->depth() == field.depth();
    // A corridor confines the SEARCH and says nothing about the ground: a cell outside it is not
    // impassable, it is simply not looked at, so a corridor that is wrong makes this fail rather
    // than return a road nobody planned.
    const bool hasCorridor = params.corridor != nullptr && params.corridor->width() == field.width() &&
                             params.corridor->depth() == field.depth();

    // The heuristic must never overestimate, or A* stops returning the cheapest
    // road and starts returning a plausible one. Chebyshev distance times the
    // cheapest possible cell is the largest value that still cannot: no route can
    // reach the goal in fewer steps, and no step can cost less than this.
    //
    // @warning The cheapest step is `base` MINUS the reuse discount, and getting that wrong made
    // the two halves of this file contradict each other. The bound used to be `base` on the
    // stated grounds that "no step can cost less than base" -- which `reuseDiscount` is
    // precisely designed to falsify, since entering an existing road costs
    // base * (1 - discount). The heuristic therefore overestimated exactly when a road was
    // available, so the search discarded the merge before evaluating it: measured on two
    // parallel attested roads two cells apart, both were laid in full (82 cells painted,
    // 82 cells expanded) at every discount from 0 to 1. The feature documented as "the whole
    // reason routes merge into a network instead of piling up as parallel lines" did nothing
    // at all, and nothing noticed because the pass had no caller.
    const math::Fixed32 cheapestStep = base - base * discount;
    // @warning On a closed world the column distance is the SHORTER way round, and this line is the
    // one that decides whether a wrapped route is correct or merely plausible. An unwrapped
    // distance near the seam overestimates by nearly a circumference, so the heuristic stops being
    // admissible exactly where the answer matters and A* discards the short way unevaluated.
    const bool wraps = params.wrapColumns > 0u && params.wrapColumns == width;
    // A pole is only meaningful on a grid that already closes east-west: crossing one turns the
    // longitude by half a world, and half of nothing is nothing.
    const bool poles = wraps && params.wrapPoles;
    const core::i32 columns = static_cast<core::i32>(params.wrapColumns);
    const core::i32 rows = static_cast<core::i32>(field.depth());

    const auto foldColumns = [&](core::i32 dx) {
        if (!wraps)
            return dx;
        if (dx > columns / 2)
            return dx - columns;
        if (dx < -(columns / 2))
            return dx + columns;
        return dx;
    };

    const auto heuristic = [&](core::u32 cell) {
        const core::i32 x = static_cast<core::i32>(cell % width);
        const core::i32 z = static_cast<core::i32>(cell / width);
        const core::i32 dx = foldColumns(x - static_cast<core::i32>(goalX));
        const core::i32 dz = z - static_cast<core::i32>(goalZ);
        const core::i32 ax = dx < 0 ? -dx : dx;
        const core::i32 az = dz < 0 ? -dz : dz;
        core::i32 best = ax > az ? ax : az;

        if (poles)
        {
            // @warning Three route classes, and the estimate is their MINIMUM. Any path either stays
            // in the sheet, crosses the north pole, or crosses the south; each expression below is a
            // lower bound on the steps its class needs, so the smallest is a lower bound on every
            // path -- which is exactly what keeps A* returning the cheapest road rather than a
            // plausible one. Offering the shortcut in the graph and NOT here would leave the
            // estimate too large for the routes that use it.
            //
            // Crossing lands half a world away in longitude, so the horizontal work is measured
            // from there.
            const core::i32 acrossX = foldColumns(x + columns / 2 - static_cast<core::i32>(goalX));
            const core::i32 acrossAbs = acrossX < 0 ? -acrossX : acrossX;

            // North: up to row zero, one step over, then down to the goal. Steps are diagonal, so
            // the bound is the larger of the vertical and horizontal totals, never their sum.
            const core::i32 northVertical = z + static_cast<core::i32>(goalZ) + 1;
            const core::i32 north = northVertical > acrossAbs ? northVertical : acrossAbs;
            if (north < best)
                best = north;

            const core::i32 southVertical =
                (rows - 1 - z) + (rows - 1 - static_cast<core::i32>(goalZ)) + 1;
            const core::i32 south = southVertical > acrossAbs ? southVertical : acrossAbs;
            if (south < best)
                best = south;
        }
        return cheapestStep * math::Fixed32::fromInt(best);
    };

    lpl::pmr::vector<math::Fixed32> best(cells, math::Fixed32::max());
    lpl::pmr::vector<core::u32> from(cells, kNoCell);
    lpl::pmr::vector<core::u8> settled(cells, core::u8{0});

    SearchHeap open;
    open.reserve(cells / 4u + 8u);
    best[start] = math::Fixed32::zero();
    open.push(SearchEntry{heuristic(start).raw(), start});

    const core::u32 budget = params.maxExpansions == 0u ? cells * 4u : params.maxExpansions;

    while (!open.empty())
    {
        const SearchEntry current = open.pop();
        if (settled[current.cell] != 0u)
            continue; // a cheaper route to it was settled first (lazy deletion)
        settled[current.cell] = 1u;
        ++path.expanded;

        if (current.cell == goal)
        {
            path.found = true;
            break;
        }
        if (path.expanded >= budget)
            break;

        const core::u32 x = current.cell % width;
        const core::u32 z = current.cell / width;
        const math::Fixed32 here = field[current.cell];

        for (core::u32 n = 0u; n < 8u; ++n)
        {
            core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[n];
            core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[n];
            // @warning A pole is NOT a wrap: stepping north off the top row comes back on the top
            // row, half a world away in longitude, walking south again. Sending it to the bottom
            // row instead would be a torus, joining the Arctic to the Antarctic.
            if (poles && nz < 0)
            {
                nz = 0;
                nx += columns / 2;
            }
            else if (poles && nz >= rows)
            {
                nz = rows - 1;
                nx += columns / 2;
            }
            // The column past the last IS the first: a step across the seam is an ordinary step.
            if (wraps)
            {
                nx %= columns;
                if (nx < 0)
                    nx += columns;
            }
            if (!field.contains(nx, nz))
                continue;
            const core::u32 next = field.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (hasCorridor && (*params.corridor)[next] == 0u)
                continue;
            if (settled[next] != 0u)
                continue;

            // Distance: a diagonal is longer than an orthogonal step, and paying
            // the same for both is what makes a router produce staircases.
            // sqrt(2) as 2/sqrt(2), because the module already keeps 1/sqrt(2)
            // exactly and a second constant is a second thing to get wrong.
            const math::Fixed32 step = n < 4u ? base : base * (math::kInvSqrt2 + math::kInvSqrt2);

            // Climbing: what actually decides where a road goes. Absolute, so a
            // descent is as expensive as a climb — a road cut into a hillside
            // pays for the earth it moves either way.
            const math::Fixed32 rise = (field[next] - here).abs();
            math::Fixed32 cost = step + slopeCost * rise;

            if (field[next] <= waterLevel)
                cost = cost + waterCost;

            // An existing road is cheap ground. This is the whole reason routes
            // merge into a network instead of piling up as parallel lines.
            if (hasExisting && (*existing)[next] != 0u)
                cost = cost - base * discount;
            if (cost.raw() <= 0)
                cost = math::Fixed32::fromRaw(1);

            const math::Fixed32 candidate = best[current.cell] + cost;
            if (candidate < best[next])
            {
                best[next] = candidate;
                from[next] = current.cell;
                open.push(SearchEntry{(candidate + heuristic(next)).raw(), next});
            }
        }
    }

    if (!path.found)
        return path;

    path.cost = best[goal];
    for (core::u32 cell = goal; cell != kNoCell; cell = from[cell])
    {
        path.cells.push_back(cell);
        if (cell == start)
            break;
    }
    // Walked backwards from the goal, so reverse in place: a caller drawing the
    // route should see it run the way it was asked for.
    for (core::usize i = 0u, j = path.cells.size(); i + 1u < j; ++i, --j)
    {
        const core::u32 swap = path.cells[i];
        path.cells[i] = path.cells[j - 1u];
        path.cells[j - 1u] = swap;
    }
    return path;
}

core::u32 connectPlaces(const Heightfield &field, const lpl::pmr::vector<core::u32> &places,
                        const RoutingParams &params, Grid<core::u8> &roads)
{
    if (field.empty() || places.size() < 2u)
        return 0u;
    if (roads.width() != field.width() || roads.depth() != field.depth())
        roads = Grid<core::u8>{field.width(), field.depth(), 0u};

    const core::u32 width = field.width();
    const core::u32 cells = field.cellCount();
    core::u32 painted = 0u;

    const math::Fixed32 base = math::Fixed32::fromFloat(params.baseCost);
    const math::Fixed32 slopeCost = math::Fixed32::fromFloat(params.slopePenalty);
    const math::Fixed32 waterCost = math::Fixed32::fromFloat(params.waterPenalty);
    const math::Fixed32 waterLevel = math::Fixed32::fromFloat(params.waterLevel);
    const math::Fixed32 discount = math::Fixed32::fromFloat(params.reuseDiscount < 0.0f ? 0.0f :
                                                            params.reuseDiscount > 1.0f ? 1.0f :
                                                                                          params.reuseDiscount);

    // Grow one tree: the first place is in, and each round attaches whichever
    // outsider is cheapest to reach from anything already connected. Prim's
    // shape, and the reason it is not Kruskal is that the cost of an edge here
    // CHANGES once a road exists — so edges must be priced against the network as
    // it stands, not once at the start.
    //
    // One search per round, seeded from EVERY connected place at zero cost, and
    // stopped at the first unconnected place it settles. The obvious way to write
    // this is a pair loop — price every (connected, unconnected) pair with its own
    // A*, keep the cheapest — and that is what it used to do. It is O(P^3) least-
    // cost searches over the whole grid, which does not show up on the small maps
    // the tests use and is catastrophic on a real one: a 128x128 world with 36
    // districts spent ELEVEN SECONDS here, more than the whole rest of generation
    // put together, and it was only visible once the viewer had to rebuild a world
    // on a keypress. Seeding the frontier with the whole connected set answers the
    // same question — what is the cheapest link between the network and anything
    // outside it — in one sweep instead of P^2.
    //
    // The multi-source form is also the more faithful one. Pricing pairs measures
    // the distance between two SITES; pricing from the frontier measures it from
    // the network as it actually stands, roads already painted included, which is
    // what the reuse discount was for in the first place.
    lpl::pmr::vector<core::u8> connected(places.size(), core::u8{0});
    connected[0] = 1u;

    // Which place, if any, sits on a cell. Rebuilt never: places do not move.
    lpl::pmr::vector<core::u32> placeAt(cells, kNoCell);
    for (core::u32 i = 0u; i < places.size(); ++i)
        if (places[i] < cells)
            placeAt[places[i]] = i;

    lpl::pmr::vector<math::Fixed32> best(cells, math::Fixed32::max());
    lpl::pmr::vector<core::u32> from(cells, kNoCell);
    lpl::pmr::vector<core::u8> settled(cells, core::u8{0});

    for (core::u32 round = 1u; round < places.size(); ++round)
    {
        for (core::u32 i = 0u; i < cells; ++i)
        {
            best[i] = math::Fixed32::max();
            from[i] = kNoCell;
            settled[i] = 0u;
        }

        SearchHeap open;
        open.reserve(cells / 4u + 8u);
        for (core::u32 i = 0u; i < places.size(); ++i)
            if (connected[i] != 0u && places[i] < cells)
            {
                best[places[i]] = math::Fixed32::zero();
                open.push(SearchEntry{0, places[i]});
            }

        core::u32 reached = kNoCell;
        core::u32 reachedPlace = 0u;
        while (!open.empty())
        {
            const SearchEntry current = open.pop();
            if (settled[current.cell] != 0u)
                continue;
            settled[current.cell] = 1u;

            const core::u32 here = placeAt[current.cell];
            if (here != kNoCell && connected[here] == 0u)
            {
                reached = current.cell;
                reachedPlace = here;
                break;
            }

            const core::u32 x = current.cell % width;
            const core::u32 z = current.cell / width;
            const math::Fixed32 height = field[current.cell];

            for (core::u32 n = 0u; n < 8u; ++n)
            {
                core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[n];
                // @warning The SAME wrap as the router above. This file has already been bitten by its
                // two halves disagreeing about the cost model; letting only one of them cross the
                // seam would be the same fault in a new place.
                if (params.wrapColumns > 0u && params.wrapColumns == field.width())
                {
                    const core::i32 columns = static_cast<core::i32>(params.wrapColumns);
                    nx %= columns;
                    if (nx < 0)
                        nx += columns;
                }
                if (!field.contains(nx, nz))
                    continue;
                const core::u32 next = field.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                if (settled[next] != 0u)
                    continue;

                const math::Fixed32 step = n < 4u ? base : base * (math::kInvSqrt2 + math::kInvSqrt2);
                const math::Fixed32 rise = (field[next] - height).abs();
                math::Fixed32 cost = step + slopeCost * rise;
                if (field[next] <= waterLevel)
                    cost = cost + waterCost;
                if (roads[next] != 0u)
                    cost = cost - base * discount;
                if (cost.raw() <= 0)
                    cost = math::Fixed32::fromRaw(1);

                const math::Fixed32 candidate = best[current.cell] + cost;
                if (candidate < best[next])
                {
                    best[next] = candidate;
                    from[next] = current.cell;
                    // No heuristic: with many sources and many possible goals there
                    // is no single target to estimate toward, and an inadmissible
                    // guess would buy speed by returning a road that is merely
                    // plausible. Dijkstra is the honest form of this search.
                    open.push(SearchEntry{candidate.raw(), next});
                }
            }
        }

        if (reached == kNoCell)
            break; // nothing left is reachable; a network of what can be joined

        for (core::u32 cell = reached; cell != kNoCell; cell = from[cell])
        {
            if (roads[cell] == 0u)
            {
                roads[cell] = 1u;
                ++painted;
            }
            if (best[cell].raw() == 0)
                break; // a source: the route is complete
        }
        connected[reachedPlace] = 1u;
    }
    return painted;
}


namespace {

/**
 * @brief Paints the fine cells a coarse plan opens, with slack either side.
 *
 * @param plan      Coarse cells the route runs through.
 * @param coarse    The summary field, for its width.
 * @param fine      The full-resolution field, for its extent.
 * @param cellRatio Fine cells per coarse cell.
 * @param margin    Coarse cells of slack.
 * @param out       Receives the corridor; already sized to @p fine.
 * @return How many fine cells were opened.
 *
 * @warning A coarse column before the first or past the last is SKIPPED, not wrapped, so on a
 * closed world the corridor is narrower at the seam than anywhere else. Three fixtures were built
 * to make that cost a road and none did: the plan is drawn on the same terrain the fine road
 * crosses, so wherever the road wants the far side of the seam the plan has already gone there --
 * and the plan's own cells wrap. Wrapping the slack too could only help, and is left undone
 * because it would buy expansions everywhere for a case nobody could demonstrate. See
 * test-terrain-routes; if a fixture ever bites, this is the three lines it wants.
 */
[[nodiscard]] core::u32 paintCorridor(const lpl::pmr::vector<core::u32> &plan, const Heightfield &coarse,
                                      const Heightfield &fine, core::u32 cellRatio, core::u32 margin,
                                      Grid<core::u8> &out)
{
    core::u32 opened = 0u;
    const core::i32 slack = static_cast<core::i32>(margin);
    for (const core::u32 cell : plan)
    {
        const core::i32 cx = static_cast<core::i32>(cell % coarse.width());
        const core::i32 cz = static_cast<core::i32>(cell / coarse.width());
        for (core::i32 dz = -slack; dz <= slack; ++dz)
        {
            for (core::i32 dx = -slack; dx <= slack; ++dx)
            {
                const core::i64 baseX = static_cast<core::i64>(cx + dx) * cellRatio;
                const core::i64 baseZ = static_cast<core::i64>(cz + dz) * cellRatio;
                for (core::u32 fz = 0u; fz < cellRatio; ++fz)
                {
                    for (core::u32 fx = 0u; fx < cellRatio; ++fx)
                    {
                        const core::i64 x = baseX + fx;
                        const core::i64 z = baseZ + fz;
                        if (x < 0 || z < 0 || x >= static_cast<core::i64>(fine.width()) ||
                            z >= static_cast<core::i64>(fine.depth()))
                            continue;
                        core::u8 &slot = out.at(static_cast<core::u32>(x), static_cast<core::u32>(z));
                        if (slot == 0u)
                        {
                            slot = 1u;
                            ++opened;
                        }
                    }
                }
            }
        }
    }
    return opened;
}

} // namespace

Heightfield reduceHeightfield(const Heightfield &field, core::u32 ratio)
{
    if (field.empty() || ratio <= 1u)
        return field;

    // Ceiling, so the block that runs off the edge still gets a cell: dropping it would make the
    // summary describe a smaller world than the one being routed, and the goal near the far edge
    // would have no coarse cell to plan toward.
    const core::u32 width = (field.width() + ratio - 1u) / ratio;
    const core::u32 depth = (field.depth() + ratio - 1u) / ratio;

    Heightfield out{width, depth, math::Fixed32::zero()};
    for (core::u32 cz = 0u; cz < depth; ++cz)
    {
        for (core::u32 cx = 0u; cx < width; ++cx)
        {
            core::i64 sum = 0;
            core::u32 count = 0u;
            for (core::u32 fz = cz * ratio; fz < (cz + 1u) * ratio && fz < field.depth(); ++fz)
            {
                for (core::u32 fx = cx * ratio; fx < (cx + 1u) * ratio && fx < field.width(); ++fx)
                {
                    sum += static_cast<core::i64>(field.at(fx, fz).raw());
                    ++count;
                }
            }
            if (count == 0u)
                continue;
            // Floors rather than truncating toward zero, so a block of sea and a block of hill
            // round the same way. Plain integer arithmetic either way, so both targets agree.
            const core::i64 divisor = static_cast<core::i64>(count);
            core::i64 mean = sum / divisor;
            if ((sum % divisor) != 0 && sum < 0)
                --mean;
            out.at(cx, cz) = math::Fixed32::fromRaw(static_cast<core::i32>(mean));
        }
    }
    return out;
}

HierarchicalRoute routeAcrossWorld(const Heightfield &coarse, const Heightfield &fine, core::u32 cellRatio,
                                   const Grid<core::u8> *existing, core::u32 startX, core::u32 startZ,
                                   core::u32 goalX, core::u32 goalZ, const RoutingParams &params,
                                   core::u32 margin)
{
    HierarchicalRoute out{};

    // A ratio of one means the two fields are the same resolution, so there is nothing to cascade
    // and a plain route is both correct and cheaper than pretending otherwise.
    if (cellRatio <= 1u || coarse.width() == 0u || fine.width() == 0u)
    {
        out.fine = routeLeastCost(fine, existing, startX, startZ, goalX, goalZ, params);
        out.coarseFound = out.fine.found;
        return out;
    }

    // The coarse plan. Its own params carry no corridor -- there is nothing above it to be confined
    // by -- and its wrap must be the coarse grid's, not the fine one's, or the heuristic folds at a
    // width the grid does not have and stops being admissible.
    RoutingParams coarseParams = params;
    coarseParams.corridor = nullptr;
    coarseParams.wrapColumns = params.wrapColumns > 0u ? coarse.width() : 0u;

    const RoutedPath plan = routeLeastCost(coarse, nullptr, startX / cellRatio, startZ / cellRatio,
                                           goalX / cellRatio, goalZ / cellRatio, coarseParams);
    out.coarseExpanded = plan.expanded;
    out.coarseFound = plan.found;
    if (!plan.found)
        return out; // No shape to refine. Reported, never papered over with a flat search.

    RoutingParams fineParams = params;
    Grid<core::u8> corridor{fine.width(), fine.depth(), core::u8{0}};
    out.corridorCells = paintCorridor(plan.cells, coarse, fine, cellRatio, margin, corridor);
    fineParams.corridor = &corridor;
    out.fine = routeLeastCost(fine, existing, startX, startZ, goalX, goalZ, fineParams);
    if (out.fine.found)
        return out;

    // @warning A coarse cell is an AVERAGE, so a plan can cross a strait that is water at full
    // resolution. Widening once is the cheap recovery; failing after it is REPORTED, because a
    // refinement that fell back to the coarse plan would return a road through water and it would
    // look like every other road.
    out.widened = true;
    Grid<core::u8> wider{fine.width(), fine.depth(), core::u8{0}};
    out.corridorCells = paintCorridor(plan.cells, coarse, fine, cellRatio, margin + margin + 1u, wider);
    fineParams.corridor = &wider;
    out.fine = routeLeastCost(fine, existing, startX, startZ, goalX, goalZ, fineParams);
    return out;
}

} // namespace lpl::procgen
