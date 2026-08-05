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

    // The heuristic must never overestimate, or A* stops returning the cheapest
    // road and starts returning a plausible one. Chebyshev distance times the
    // cheapest possible cell is the largest value that still cannot: no route can
    // reach the goal in fewer steps, and no step can cost less than base.
    const auto heuristic = [&](core::u32 cell) {
        const core::i32 dx = static_cast<core::i32>(cell % width) - static_cast<core::i32>(goalX);
        const core::i32 dz = static_cast<core::i32>(cell / width) - static_cast<core::i32>(goalZ);
        const core::i32 ax = dx < 0 ? -dx : dx;
        const core::i32 az = dz < 0 ? -dz : dz;
        return base * math::Fixed32::fromInt(ax > az ? ax : az);
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
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[n];
            if (!field.contains(nx, nz))
                continue;
            const core::u32 next = field.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
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
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[n];
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

} // namespace lpl::procgen
