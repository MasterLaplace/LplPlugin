#include <lpl/procgen/QualityGate.hpp>

#include <lpl/std/vector.hpp>

namespace lpl::procgen {

namespace {

/**
 * @brief Forwards to @ref isWalkable.
 *
 * The predicate lives in Dungeon.hpp: what counts as walkable decides what
 * "connected" means, and two answers to that question is one too many.
 */
constexpr bool walkable(DungeonCell cell) noexcept { return lpl::procgen::isWalkable(cell); }

/**
 * @brief Counts a cell's walkable 4-neighbours.
 */
core::u32 walkableNeighbours(const DungeonMap &map, core::u32 x, core::u32 z)
{
    core::u32 count = 0u;
    for (core::u32 n = 0u; n < 4u; ++n)
    {
        const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
        const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
        if (map.contains(nx, nz) && walkable(map.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz))))
            ++count;
    }
    return count;
}

/**
 * @brief Farthest reachable cell from a source, and its distance.
 *
 * Strictly greater keeps the first of several equally distant cells, in scan order, so
 * the choice is reproducible.
 */
bool farthestFrom(const DungeonMap &map, core::u32 startX, core::u32 startZ, core::u32 &outX, core::u32 &outZ,
                  core::u32 &outDistance)
{
    const DistanceMap distances = computeDistanceMap(map, startX, startZ);
    bool found = false;
    core::u32 best = 0u;

    for (core::u32 i = 0u; i < distances.cellCount(); ++i)
    {
        if (distances[i] == kUnreachable)
            continue;
        if (!found || distances[i] > best)
        {
            best = distances[i];
            outX = i % map.width();
            outZ = i / map.width();
            found = true;
        }
    }
    outDistance = best;
    return found;
}

} // namespace

DistanceMap computeDistanceMap(const DungeonMap &map, core::u32 startX, core::u32 startZ)
{
    DistanceMap distances{map.width(), map.depth(), kUnreachable};
    if (map.empty() || !map.contains(static_cast<core::i32>(startX), static_cast<core::i32>(startZ)))
        return distances;
    if (!walkable(map.at(startX, startZ)))
        return distances;

    lpl::pmr::vector<core::u32> queue;
    queue.reserve(map.cellCount());

    const core::u32 start = map.index(startX, startZ);
    distances[start] = 0u;
    queue.push_back(start);

    for (core::u32 head = 0u; head < queue.size(); ++head)
    {
        const core::u32 cell = queue[head];
        const core::u32 x = cell % map.width();
        const core::u32 z = cell / map.width();
        const core::u32 next = distances[cell] + 1u;

        for (core::u32 n = 0u; n < 4u; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
            if (!map.contains(nx, nz))
                continue;
            const core::u32 index = map.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (!walkable(map[index]) || distances[index] != kUnreachable)
                continue;
            distances[index] = next;
            queue.push_back(index);
        }
    }
    return distances;
}

DistanceMap computeDistanceMapFrom(const DungeonMap &map, const Grid<core::u8> &sources)
{
    DistanceMap distances{map.width(), map.depth(), kUnreachable};
    if (map.empty() || sources.width() != map.width() || sources.depth() != map.depth())
        return distances;

    lpl::pmr::vector<core::u32> queue;
    queue.reserve(map.cellCount());

    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        if (sources[i] != 0u && walkable(map[i]))
        {
            distances[i] = 0u;
            queue.push_back(i);
        }

    for (core::u32 head = 0u; head < queue.size(); ++head)
    {
        const core::u32 cell = queue[head];
        const core::u32 x = cell % map.width();
        const core::u32 z = cell / map.width();
        const core::u32 next = distances[cell] + 1u;

        for (core::u32 n = 0u; n < 4u; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
            if (!map.contains(nx, nz))
                continue;
            const core::u32 index = map.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (!walkable(map[index]) || distances[index] != kUnreachable)
                continue;
            distances[index] = next;
            queue.push_back(index);
        }
    }
    return distances;
}

HotPathAnalysis analyseHotPath(const DungeonMap &map, core::u32 startX, core::u32 startZ, core::u32 goalX,
                               core::u32 goalZ, core::u32 detourLimit)
{
    HotPathAnalysis analysis;
    if (map.empty() || !map.contains(static_cast<core::i32>(goalX), static_cast<core::i32>(goalZ)))
        return analysis;

    const DistanceMap fromStart = computeDistanceMap(map, startX, startZ);
    const core::u32 goal = map.index(goalX, goalZ);
    if (fromStart[goal] == kUnreachable)
        return analysis;

    analysis.onPath = Grid<core::u8>{map.width(), map.depth(), 0u};
    core::u32 cell = goal;
    for (;;)
    {
        analysis.onPath[cell] = 1u;
        ++analysis.pathCells;
        if (fromStart[cell] == 0u)
            break;

        const core::u32 x = cell % map.width();
        const core::u32 z = cell / map.width();
        core::u32 previous = cell;
        for (core::u32 n = 0u; n < 4u; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
            if (!map.contains(nx, nz))
                continue;
            const core::u32 index = map.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (fromStart[index] != kUnreachable && fromStart[index] + 1u == fromStart[cell])
            {
                previous = index;
                break;
            }
        }
        if (previous == cell)
            break;
        cell = previous;
    }

    analysis.detour = computeDistanceMapFrom(map, analysis.onPath);
    for (core::u32 i = 0u; i < analysis.detour.cellCount(); ++i)
    {
        const core::u32 depth = analysis.detour[i];
        if (depth == kUnreachable)
            continue;
        if (depth > detourLimit)
            ++analysis.excessiveCells;
        if (depth > analysis.deepestDetour)
        {
            analysis.deepestDetour = depth;
            analysis.farthestCell = i;
        }
    }

    analysis.valid = true;
    return analysis;
}

core::u32 placeAlongHotPath(const DungeonMap &map, const HotPathAnalysis &analysis, core::u32 startX, core::u32 startZ,
                            const PlacementParams &params, Placement *out, core::u32 capacity)
{
    if (out == nullptr || capacity == 0u || !analysis.valid || map.empty() || analysis.onPath.empty())
        return 0u;

    const DistanceMap fromStart = computeDistanceMap(map, startX, startZ);
    core::u32 written = 0u;

    core::u32 spine = 0u;
    for (core::u32 i = 0u; i < analysis.onPath.cellCount(); ++i)
        if (analysis.onPath[i] != 0u && fromStart[i] != kUnreachable && fromStart[i] > spine)
            spine = fromStart[i];

    const core::u32 encounterStart = written;
    for (core::u32 k = 1u; k <= params.encounters && written < capacity; ++k)
    {
        const core::u32 wanted = (spine * k) / (params.encounters + 1u);
        bool found = false;
        core::u32 bestCell = 0u;
        core::u32 bestError = 0u;
        for (core::u32 i = 0u; i < analysis.onPath.cellCount(); ++i)
        {
            if (analysis.onPath[i] == 0u || fromStart[i] == kUnreachable)
                continue;
            const core::u32 progress = fromStart[i];
            const core::u32 error = progress > wanted ? progress - wanted : wanted - progress;
            if (!found || error < bestError)
            {
                found = true;
                bestCell = i;
                bestError = error;
            }
        }
        if (!found)
            break;

        bool tooClose = false;
        for (core::u32 e = encounterStart; e < written; ++e)
        {
            const core::u32 other = out[e].progress;
            const core::u32 gap =
                other > fromStart[bestCell] ? other - fromStart[bestCell] : fromStart[bestCell] - other;
            if (gap < params.minSpacing)
                tooClose = true;
        }
        if (tooClose)
            continue;

        out[written].x = bestCell % map.width();
        out[written].z = bestCell / map.width();
        out[written].role = PlacementRole::Encounter;
        out[written].detour = 0u;
        out[written].progress = fromStart[bestCell];
        ++written;
    }

    const core::u32 rewardStart = written;
    for (core::u32 k = 0u; k < params.rewards && written < capacity; ++k)
    {
        bool found = false;
        core::u32 bestCell = 0u;
        core::u32 bestDepth = 0u;
        for (core::u32 i = 0u; i < analysis.detour.cellCount(); ++i)
        {
            const core::u32 depth = analysis.detour[i];
            if (depth == kUnreachable || depth < params.rewardMinDetour)
                continue;
            const core::u32 x = i % map.width();
            const core::u32 z = i / map.width();
            if (!walkable(map.at(x, z)) || walkableNeighbours(map, x, z) != 1u)
                continue;

            bool taken = false;
            for (core::u32 r = rewardStart; r < written; ++r)
            {
                if (out[r].x == x && out[r].z == z)
                {
                    taken = true;
                    break;
                }
                const core::u32 dx = out[r].x > x ? out[r].x - x : x - out[r].x;
                const core::u32 dz = out[r].z > z ? out[r].z - z : z - out[r].z;
                if (dx + dz < params.minSpacing)
                    taken = true;
            }
            if (taken)
                continue;

            if (!found || depth > bestDepth)
            {
                found = true;
                bestCell = i;
                bestDepth = depth;
            }
        }
        if (!found)
            break;

        out[written].x = bestCell % map.width();
        out[written].z = bestCell / map.width();
        out[written].role = PlacementRole::Reward;
        out[written].detour = bestDepth;
        out[written].progress = fromStart[bestCell] == kUnreachable ? 0u : fromStart[bestCell];
        ++written;
    }

    return written;
}

DistanceMap computeFleeMap(const DungeonMap &map, const DistanceMap &danger, core::u32 safeDistance)
{
    DistanceMap flee{map.width(), map.depth(), kUnreachable};
    if (map.empty() || danger.width() != map.width() || danger.depth() != map.depth())
        return flee;

    core::u32 threshold = safeDistance;
    if (threshold == 0u)
    {
        for (core::u32 i = 0u; i < danger.cellCount(); ++i)
            if (danger[i] != kUnreachable && danger[i] > threshold)
                threshold = danger[i];
    }

    Grid<core::u8> havens{map.width(), map.depth(), 0u};
    bool anyHaven = false;
    for (core::u32 i = 0u; i < danger.cellCount(); ++i)
        if (danger[i] != kUnreachable && danger[i] >= threshold)
        {
            havens[i] = 1u;
            anyHaven = true;
        }
    if (!anyHaven)
        return flee;

    return computeDistanceMapFrom(map, havens);
}

LevelQuality evaluateLevel(const DungeonMap &map, core::u32 startX, core::u32 startZ, core::u32 goalX, core::u32 goalZ)
{
    LevelQuality quality;
    if (map.empty())
        return quality;

    for (core::u32 z = 0u; z < map.depth(); ++z)
    {
        for (core::u32 x = 0u; x < map.width(); ++x)
        {
            if (!walkable(map.at(x, z)))
                continue;
            ++quality.walkableCells;

            const core::u32 neighbours = walkableNeighbours(map, x, z);
            if (neighbours == 1u)
                ++quality.deadEnds;
            else if (neighbours >= 3u)
                ++quality.junctions;
        }
    }

    const DistanceMap distances = computeDistanceMap(map, startX, startZ);
    for (core::u32 i = 0u; i < distances.cellCount(); ++i)
    {
        if (distances[i] == kUnreachable)
            continue;
        ++quality.reachableCells;
        if (distances[i] > quality.longestDistance)
            quality.longestDistance = distances[i];
    }

    if (map.contains(static_cast<core::i32>(goalX), static_cast<core::i32>(goalZ)))
    {
        const core::u32 goal = distances[map.index(goalX, goalZ)];
        quality.goalReachable = goal != kUnreachable;
        quality.pathLength = quality.goalReachable ? goal : 0u;
    }

    quality.fullyConnected = quality.walkableCells != 0u && quality.reachableCells == quality.walkableCells;
    return quality;
}

bool passesGate(const LevelQuality &quality, const GateCriteria &criteria)
{
    if (criteria.requireGoalReachable && !quality.goalReachable)
        return false;
    if (criteria.requireFullyConnected && !quality.fullyConnected)
        return false;
    if (quality.pathLength < criteria.minPathLength)
        return false;
    if (quality.walkableCells < criteria.minWalkableCells)
        return false;

    if (quality.walkableCells != 0u)
    {
        const core::u32 ratio = (quality.deadEnds * 100u) / quality.walkableCells;
        if (ratio > criteria.maxDeadEndRatio)
            return false;
    }
    return true;
}

bool findFarthestPair(const DungeonMap &map, core::u32 &outStartX, core::u32 &outStartZ, core::u32 &outGoalX,
                      core::u32 &outGoalZ)
{
    if (map.empty())
        return false;

    core::i32 seed = -1;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        if (walkable(map[i]))
        {
            seed = static_cast<core::i32>(i);
            break;
        }
    if (seed < 0)
        return false;

    const core::u32 seedX = static_cast<core::u32>(seed) % map.width();
    const core::u32 seedZ = static_cast<core::u32>(seed) / map.width();

    core::u32 firstX = seedX;
    core::u32 firstZ = seedZ;
    core::u32 distance = 0u;
    if (!farthestFrom(map, seedX, seedZ, firstX, firstZ, distance))
        return false;

    core::u32 secondX = firstX;
    core::u32 secondZ = firstZ;
    if (!farthestFrom(map, firstX, firstZ, secondX, secondZ, distance))
        return false;

    outStartX = firstX;
    outStartZ = firstZ;
    outGoalX = secondX;
    outGoalZ = secondZ;
    return true;
}

DesireMap combineDesires(const DungeonMap &map, const DesireTerm *terms, core::u32 count)
{
    DesireMap desires{map.width(), map.depth(), 0};
    if (map.empty() || terms == nullptr || count == 0u)
        return desires;

    for (core::u32 t = 0u; t < count; ++t)
    {
        const DistanceMap *source = terms[t].map;
        if (source == nullptr || source->width() != map.width() || source->depth() != map.depth())
            continue;

        for (core::u32 i = 0u; i < desires.cellCount(); ++i)
        {
            const core::u32 distance = (*source)[i];
            if (distance == kUnreachable)
                continue;
            desires[i] += (static_cast<core::i32>(distance) * terms[t].weight) / 16;
        }
    }
    return desires;
}

bool descendDesire(const DungeonMap &map, const DesireMap &desires, core::u32 x, core::u32 z, core::u32 &outX,
                   core::u32 &outZ)
{
    if (map.empty() || desires.width() != map.width() || desires.depth() != map.depth())
        return false;
    if (!map.contains(static_cast<core::i32>(x), static_cast<core::i32>(z)))
        return false;

    core::i32 best = desires.at(x, z);
    bool found = false;
    for (core::u32 n = 0u; n < 4u; ++n)
    {
        const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
        const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
        if (!map.contains(nx, nz))
            continue;
        if (map.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) == DungeonCell::Wall)
            continue;
        const core::i32 value = desires.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
        if (value < best)
        {
            best = value;
            outX = static_cast<core::u32>(nx);
            outZ = static_cast<core::u32>(nz);
            found = true;
        }
    }
    return found;
}

} // namespace lpl::procgen
