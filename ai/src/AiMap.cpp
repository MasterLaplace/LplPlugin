/**
 * @file AiMap.cpp
 * @brief Implementation of directional, capability-aware pathfinding.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ai/AiMap.hpp>

namespace lpl::ai {

namespace {

/// The index of the direction opposite @p d, in kNeighbor8 order.
///
/// kNeighbor8X/Z is {E, W, S, N, SE, NE, SW, NW}, so opposites pair up as
/// (0,1) (2,3) (4,7) (5,6). Written as a table rather than derived, because
/// deriving it from the offsets at runtime would be three lines that are wrong
/// the day the neighbour order changes and nothing would say so.
constexpr core::u32 kOpposite[8] = {1u, 0u, 3u, 2u, 7u, 6u, 5u, 4u};

/// A node in the open set: (cell, incoming direction) with its cost so far.
struct SearchNode {
    core::u32 cell;
    core::u32 incoming;
    core::u32 estimate; ///< g + h, the ordering key.
};

/// Binary min-heap over @ref SearchNode. Hand-rolled: no std:: freestanding, and
/// the comparison must be exactly this one for the search to be reproducible.
class SearchHeap {
public:
    void push(SearchNode node)
    {
        _items.push_back(node);
        core::u32 child = static_cast<core::u32>(_items.size()) - 1u;
        while (child != 0u)
        {
            const core::u32 parent = (child - 1u) / 2u;
            if (!precedes(_items[child], _items[parent]))
                break;
            const SearchNode swap = _items[parent];
            _items[parent] = _items[child];
            _items[child] = swap;
            child = parent;
        }
    }

    [[nodiscard]] bool empty() const { return _items.empty(); }

    [[nodiscard]] SearchNode pop()
    {
        const SearchNode top = _items[0];
        _items[0] = _items[_items.size() - 1u];
        _items.pop_back();

        const core::u32 count = static_cast<core::u32>(_items.size());
        core::u32 node = 0u;
        for (;;)
        {
            const core::u32 left = node * 2u + 1u;
            const core::u32 right = left + 1u;
            core::u32 smallest = node;
            if (left < count && precedes(_items[left], _items[smallest]))
                smallest = left;
            if (right < count && precedes(_items[right], _items[smallest]))
                smallest = right;
            if (smallest == node)
                break;
            const SearchNode swap = _items[smallest];
            _items[smallest] = _items[node];
            _items[node] = swap;
            node = smallest;
        }
        return top;
    }

private:
    /// Estimate first, then cell, then incoming direction. The two tiebreaks are
    /// what make the expansion order a function of the map alone rather than of
    /// the heap's internal layout.
    static constexpr bool precedes(const SearchNode &a, const SearchNode &b) noexcept
    {
        if (a.estimate != b.estimate)
            return a.estimate < b.estimate;
        if (a.cell != b.cell)
            return a.cell < b.cell;
        return a.incoming < b.incoming;
    }

    lpl::pmr::vector<SearchNode> _items;
};

} // namespace

AiMap::AiMap(core::u32 width, core::u32 depth) : _capability(width, depth, core::u8{0}) {}

void AiMap::setCapability(core::u32 x, core::u32 z, core::u8 mask)
{
    if (x < _capability.width() && z < _capability.depth())
        _capability.at(x, z) = mask;
}

core::u8 AiMap::capability(core::u32 x, core::u32 z) const
{
    if (x >= _capability.width() || z >= _capability.depth())
        return 0u;
    return _capability.at(x, z);
}

core::u32 AiMap::findPath(core::u32 startX, core::u32 startZ, core::u32 goalX, core::u32 goalZ, core::u8 mask,
                          const AiMapParams &params, lpl::pmr::vector<core::u32> &outPath) const
{
    outPath.clear();
    if (_capability.empty() || !passable(startX, startZ, mask) || !passable(goalX, goalZ, mask))
        return kNoPath;

    const core::u32 width = _capability.width();
    const core::u32 depth = _capability.depth();
    const core::u32 cells = width * depth;
    const core::u32 states = cells * (kDirectionCount + 1u);
    const core::u32 start = startZ * width + startX;
    const core::u32 goal = goalZ * width + goalX;

    // One entry per (cell, incoming) pair. The extra slot is the start's "arrived
    // from nowhere" state, which must exist or the first step would be charged a
    // reversal against an arbitrary direction.
    lpl::pmr::vector<core::u32> cost(states, kNoPath);
    lpl::pmr::vector<core::u32> cameFrom(states, kNoPath);

    const auto stateOf = [width](core::u32 cell, core::u32 incoming) { return cell * (kDirectionCount + 1u) + incoming; };
    (void) width;

    const auto heuristic = [&](core::u32 cell) -> core::u32 {
        const core::i32 x = static_cast<core::i32>(cell % width);
        const core::i32 z = static_cast<core::i32>(cell / width);
        const core::i32 dx = x - static_cast<core::i32>(goalX);
        const core::i32 dz = z - static_cast<core::i32>(goalZ);
        const core::u32 adx = static_cast<core::u32>(dx < 0 ? -dx : dx);
        const core::u32 adz = static_cast<core::u32>(dz < 0 ? -dz : dz);
        // Chebyshev times the cheapest step: admissible, because no move costs
        // less than baseCost and a diagonal covers one Chebyshev unit.
        return (adx > adz ? adx : adz) * params.baseCost;
    };

    SearchHeap open;
    cost[stateOf(start, kNoIncoming)] = 0u;
    open.push(SearchNode{start, kNoIncoming, heuristic(start)});

    core::u32 bestGoalState = kNoPath;

    while (!open.empty())
    {
        const SearchNode current = open.pop();
        const core::u32 state = stateOf(current.cell, current.incoming);
        if (cost[state] == kNoPath || current.estimate < cost[state])
            continue; // Lazy deletion: a better route to this state was queued later.

        if (current.cell == goal)
        {
            bestGoalState = state;
            break;
        }

        const core::u32 x = current.cell % width;
        const core::u32 z = current.cell / width;

        for (core::u32 n = 0u; n < kDirectionCount; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor8X[n];
            const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor8Z[n];
            if (nx < 0 || nz < 0 || static_cast<core::u32>(nx) >= width || static_cast<core::u32>(nz) >= depth)
                continue;
            if (!passable(static_cast<core::u32>(nx), static_cast<core::u32>(nz), mask))
                continue;

            const bool diagonal = procgen::kNeighbor8X[n] != 0 && procgen::kNeighbor8Z[n] != 0;
            core::u32 step = diagonal ? params.diagonalCost : params.baseCost;

            if (current.incoming != kNoIncoming)
            {
                // The reversal charge: stepping back the way we came. This is the
                // one line that makes a long body prefer walking on to a junction
                // over folding through itself.
                if (n == kOpposite[current.incoming])
                    step += params.reverseCost;
                else if (n != current.incoming)
                    step += params.turnCost;
            }

            const core::u32 nextCell = static_cast<core::u32>(nz) * width + static_cast<core::u32>(nx);
            const core::u32 nextState = stateOf(nextCell, n);
            const core::u32 candidate = cost[state] + step;
            if (cost[nextState] != kNoPath && candidate >= cost[nextState])
                continue;

            cost[nextState] = candidate;
            cameFrom[nextState] = state;
            open.push(SearchNode{nextCell, n, candidate + heuristic(nextCell)});
        }
    }

    if (bestGoalState == kNoPath)
    {
        // The goal may have been reached through any incoming direction; the loop
        // above stops at the first, but if it never did, check every arrival.
        core::u32 best = kNoPath;
        for (core::u32 d = 0u; d <= kDirectionCount; ++d)
        {
            const core::u32 candidate = stateOf(goal, d);
            if (cost[candidate] != kNoPath && (best == kNoPath || cost[candidate] < cost[best]))
                best = candidate;
        }
        if (best == kNoPath)
            return kNoPath;
        bestGoalState = best;
    }

    const core::u32 total = cost[bestGoalState];
    for (core::u32 state = bestGoalState; state != kNoPath; state = cameFrom[state])
        outPath.push_back(state / (kDirectionCount + 1u));

    // Reverse in place: the walk above went goal to start.
    for (core::u32 i = 0u, j = static_cast<core::u32>(outPath.size()); i + 1u < j; ++i, --j)
    {
        const core::u32 swap = outPath[i];
        outPath[i] = outPath[j - 1u];
        outPath[j - 1u] = swap;
    }
    return total;
}

core::u32 countSelfIntersections(const core::u32 *path, core::u32 count, core::u32 bodyLength)
{
    if (path == nullptr || count == 0u || bodyLength == 0u)
        return 0u;

    core::u32 hits = 0u;
    for (core::u32 head = 1u; head < count; ++head)
    {
        const core::u32 first = head > bodyLength ? head - bodyLength : 0u;
        for (core::u32 segment = first; segment + 1u <= head; ++segment)
        {
            if (segment == head)
                continue;
            if (path[segment] == path[head])
            {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

} // namespace lpl::ai
