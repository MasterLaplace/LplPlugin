/**
 * @file Dungeon.cpp
 * @brief Implementation of the dungeon and cave generators.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Dungeon.hpp>

#include <lpl/procgen/Random.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

constexpr core::u32 kInvalidRegion = 0xFFFFFFFFu;

/// Carves an axis-aligned rectangle of floor, clipped to the map.
void carveRect(DungeonMap &map, core::u32 x, core::u32 z, core::u32 width, core::u32 depth)
{
    for (core::u32 cz = z; cz < z + depth; ++cz)
        for (core::u32 cx = x; cx < x + width; ++cx)
            if (map.contains(static_cast<core::i32>(cx), static_cast<core::i32>(cz)))
                map.at(cx, cz) = DungeonCell::Floor;
}

/// Carves an L-shaped corridor: along X first, then along Z.
void carveCorridor(DungeonMap &map, core::u32 fromX, core::u32 fromZ, core::u32 toX, core::u32 toZ,
                   core::u32 thickness)
{
    const core::u32 width = thickness == 0u ? 1u : thickness;

    const core::u32 x0 = fromX < toX ? fromX : toX;
    const core::u32 x1 = fromX < toX ? toX : fromX;
    for (core::u32 x = x0; x <= x1; ++x)
        carveRect(map, x, fromZ, 1u, width);

    const core::u32 z0 = fromZ < toZ ? fromZ : toZ;
    const core::u32 z1 = fromZ < toZ ? toZ : fromZ;
    for (core::u32 z = z0; z <= z1; ++z)
        carveRect(map, toX, z, width, 1u);
}

/**
 * @brief Recursively splits a region, carving a room in every leaf.
 *
 * Returns the representative point of the sub-tree (a room centre), which the
 * caller uses to join its two children. That is what makes connectivity
 * structural: every split immediately links the two halves it created, so the
 * whole tree ends up joined without any global pathfinding.
 */
struct BspNode {
    core::u32 x, z, width, depth;
};

void splitAndCarve(DungeonMap &map, const BspNode &node, core::u32 level, const BspDungeonParams &params,
                   Random &random, lpl::pmr::vector<Room> *outRooms, core::u32 &outCenterX, core::u32 &outCenterZ)
{
    const bool tooSmallToSplit = node.width < params.minLeafSize * 2u && node.depth < params.minLeafSize * 2u;

    if (level >= params.maxDepth || tooSmallToSplit)
    {
        // Leaf: carve a room somewhere inside, leaving the padding as rock.
        const core::u32 padding = params.roomPadding;
        if (node.width <= padding * 2u + 2u || node.depth <= padding * 2u + 2u)
        {
            outCenterX = node.x + node.width / 2u;
            outCenterZ = node.z + node.depth / 2u;
            return;
        }

        const core::u32 maxWidth = node.width - padding * 2u;
        const core::u32 maxDepth = node.depth - padding * 2u;
        const core::u32 roomWidth = static_cast<core::u32>(random.range(2, static_cast<core::i32>(maxWidth)));
        const core::u32 roomDepth = static_cast<core::u32>(random.range(2, static_cast<core::i32>(maxDepth)));
        const core::u32 roomX =
            node.x + padding + static_cast<core::u32>(random.below(maxWidth - roomWidth + 1u));
        const core::u32 roomZ =
            node.z + padding + static_cast<core::u32>(random.below(maxDepth - roomDepth + 1u));

        carveRect(map, roomX, roomZ, roomWidth, roomDepth);
        const Room room{roomX, roomZ, roomWidth, roomDepth};
        if (outRooms != nullptr)
            outRooms->push_back(room);

        outCenterX = room.centerX();
        outCenterZ = room.centerZ();
        return;
    }

    // Split along the longer axis, so nodes stay roughly square and rooms do
    // not degenerate into slivers.
    const bool splitVertically = node.width > node.depth;
    core::u32 leftCenterX = 0u, leftCenterZ = 0u, rightCenterX = 0u, rightCenterZ = 0u;

    if (splitVertically)
    {
        const core::u32 low = params.minLeafSize;
        const core::u32 high = node.width - params.minLeafSize;
        const core::u32 cut = high > low ? low + random.below(high - low) : node.width / 2u;
        splitAndCarve(map, BspNode{node.x, node.z, cut, node.depth}, level + 1u, params, random, outRooms,
                      leftCenterX, leftCenterZ);
        splitAndCarve(map, BspNode{node.x + cut, node.z, node.width - cut, node.depth}, level + 1u, params, random,
                      outRooms, rightCenterX, rightCenterZ);
    }
    else
    {
        const core::u32 low = params.minLeafSize;
        const core::u32 high = node.depth - params.minLeafSize;
        const core::u32 cut = high > low ? low + random.below(high - low) : node.depth / 2u;
        splitAndCarve(map, BspNode{node.x, node.z, node.width, cut}, level + 1u, params, random, outRooms,
                      leftCenterX, leftCenterZ);
        splitAndCarve(map, BspNode{node.x, node.z + cut, node.width, node.depth - cut}, level + 1u, params, random,
                      outRooms, rightCenterX, rightCenterZ);
    }

    carveCorridor(map, leftCenterX, leftCenterZ, rightCenterX, rightCenterZ, params.corridorWidth);
    outCenterX = leftCenterX;
    outCenterZ = leftCenterZ;
}

/// Exchanges two maps without touching their cells.
void swapMaps(DungeonMap &a, DungeonMap &b)
{
    DungeonMap temporary = static_cast<DungeonMap &&>(a);
    a = static_cast<DungeonMap &&>(b);
    b = static_cast<DungeonMap &&>(temporary);
}

/// Counts rock cells in the 8-neighbourhood; out-of-bounds counts as rock.
core::u32 rockNeighbours(const DungeonMap &map, core::u32 x, core::u32 z)
{
    core::u32 count = 0u;
    for (core::u32 n = 0u; n < 8u; ++n)
    {
        const core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[n];
        const core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[n];
        // Treating the outside as solid keeps caves away from the border,
        // which is what stops them opening onto nothing.
        if (!map.contains(nx, nz) || map.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) == DungeonCell::Wall)
            ++count;
    }
    return count;
}

/**
 * @brief Labels every walkable cell with its connected-region index.
 * @return Number of regions found; @p outSizes receives each region's cell count.
 */
core::u32 labelRegions(const DungeonMap &map, Grid<core::u32> &outLabels, lpl::pmr::vector<core::u32> &outSizes)
{
    outLabels = Grid<core::u32>{map.width(), map.depth(), kInvalidRegion};
    outSizes.clear();

    lpl::pmr::vector<core::u32> queue;
    queue.reserve(map.cellCount());

    core::u32 regionCount = 0u;
    for (core::u32 start = 0u; start < map.cellCount(); ++start)
    {
        if (!isWalkable(map[start]) || outLabels[start] != kInvalidRegion)
            continue;

        // Breadth-first from this seed. An explicit queue, not recursion: a
        // large cavern would overflow any reasonable stack.
        core::u32 size = 0u;
        queue.clear();
        queue.push_back(start);
        outLabels[start] = regionCount;

        for (core::u32 head = 0u; head < queue.size(); ++head)
        {
            const core::u32 cell = queue[head];
            ++size;
            const core::u32 x = cell % map.width();
            const core::u32 z = cell / map.width();

            for (core::u32 n = 0u; n < 4u; ++n)
            {
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                if (!map.contains(nx, nz))
                    continue;
                const core::u32 neighbour =
                    outLabels.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                if (!isWalkable(map[neighbour]) || outLabels[neighbour] != kInvalidRegion)
                    continue;
                outLabels[neighbour] = regionCount;
                queue.push_back(neighbour);
            }
        }
        outSizes.push_back(size);
        ++regionCount;
    }
    return regionCount;
}

/**
 * @brief Digs the shortest possible tunnel from every region to the largest one.
 *
 * The alternative, and what this replaces, is to pick one representative cell per
 * region and chain them with L-shaped corridors. That guarantees connectivity and
 * nothing else: the representatives are whichever cell happens to come first in
 * scan order, so two adjacent caverns can be joined by a corridor that crosses the
 * entire map and drives straight through three rooms on the way.
 *
 * A single breadth-first sweep from the largest region, allowed to travel through
 * rock, gives every cell its distance to that region and the neighbour it was
 * reached from. The nearest cell of each other region is then the mouth of the
 * shortest tunnel, and walking the predecessors back carves exactly it. One pass
 * over the map, and the corridors go where a person would have put them.
 *
 * @param map      Level to tunnel through, modified in place.
 * @param labels   Region label per cell from @ref labelRegions.
 * @param sizes    Cell count per region.
 * @param regions  Number of regions.
 * @return Number of tunnels dug.
 */
core::u32 tunnelToMainRegion(DungeonMap &map, const Grid<core::u32> &labels,
                             const lpl::pmr::vector<core::u32> &sizes, core::u32 regions)
{
    // Attach to the biggest cavern rather than to region 0: region 0 is merely the
    // first found in scan order and may be a three-cell alcove, which would make
    // every other region tunnel toward the least significant part of the level.
    core::u32 main = 0u;
    for (core::u32 r = 1u; r < regions; ++r)
        if (sizes[r] > sizes[main])
            main = r;

    constexpr core::u32 kUnvisited = 0xFFFFFFFFu;
    Grid<core::u32> distance{map.width(), map.depth(), kUnvisited};
    Grid<core::u32> cameFrom{map.width(), map.depth(), kUnvisited};

    lpl::pmr::vector<core::u32> queue;
    queue.reserve(map.cellCount());
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        if (labels[i] == main)
        {
            distance[i] = 0u;
            queue.push_back(i);
        }

    for (core::u32 head = 0u; head < queue.size(); ++head)
    {
        const core::u32 cell = queue[head];
        const core::u32 x = cell % map.width();
        const core::u32 z = cell / map.width();
        const core::u32 next = distance[cell] + 1u;

        for (core::u32 n = 0u; n < 4u; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
            if (!map.contains(nx, nz))
                continue;
            const core::u32 index = map.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (distance[index] != kUnvisited)
                continue;
            distance[index] = next;
            cameFrom[index] = cell;
            queue.push_back(index);
        }
    }

    // Nearest cell of each region to the main one; ties go to the lowest index,
    // which the ascending scan below picks by using a strict comparison.
    lpl::pmr::vector<core::u32> mouth(regions, kUnvisited);
    lpl::pmr::vector<core::u32> mouthDistance(regions, kUnvisited);
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
    {
        const core::u32 label = labels[i];
        if (label == kInvalidRegion || label == main || distance[i] == kUnvisited)
            continue;
        if (distance[i] < mouthDistance[label])
        {
            mouthDistance[label] = distance[i];
            mouth[label] = i;
        }
    }

    core::u32 dug = 0u;
    for (core::u32 r = 0u; r < regions; ++r)
    {
        if (r == main || mouth[r] == kUnvisited)
            continue;
        for (core::u32 cell = mouth[r]; cell != kUnvisited && distance[cell] != 0u; cell = cameFrom[cell])
            map[cell] = DungeonCell::Floor;
        ++dug;
    }
    return dug;
}

} // namespace

DungeonMap generateBspDungeon(const BspDungeonParams &params, lpl::pmr::vector<Room> *outRooms)
{
    if (params.width == 0u || params.depth == 0u)
        return DungeonMap{};

    DungeonMap map{params.width, params.depth, DungeonCell::Wall};
    Random random = deriveStream(params.seed, 0xB59u);

    core::u32 centerX = 0u;
    core::u32 centerZ = 0u;
    splitAndCarve(map, BspNode{0u, 0u, params.width, params.depth}, 0u, params, random, outRooms, centerX, centerZ);
    return map;
}

DungeonMap generateCellularCave(const CaveParams &params)
{
    if (params.width == 0u || params.depth == 0u)
        return DungeonMap{};

    DungeonMap map{params.width, params.depth, DungeonCell::Wall};
    Random random = deriveStream(params.seed, 0xCA7Eu);
    const math::Fixed32 fill = math::Fixed32::fromFloat(params.fillProbability);

    // Seed with noise, keeping a solid border.
    for (core::u32 z = 0u; z < params.depth; ++z)
    {
        for (core::u32 x = 0u; x < params.width; ++x)
        {
            const bool border = x == 0u || z == 0u || x + 1u == params.width || z + 1u == params.depth;
            map.at(x, z) = (border || random.chance(fill)) ? DungeonCell::Wall : DungeonCell::Floor;
        }
    }

    // The 4-5 rule, applied synchronously: the next state depends only on the
    // frozen previous one, so a scratch buffer is not an optimisation, it is
    // what makes the automaton an automaton.
    DungeonMap scratch = map;
    for (core::u32 step = 0u; step < params.steps; ++step)
    {
        for (core::u32 z = 0u; z < params.depth; ++z)
        {
            for (core::u32 x = 0u; x < params.width; ++x)
            {
                const core::u32 neighbours = rockNeighbours(map, x, z);
                if (map.at(x, z) == DungeonCell::Wall)
                    scratch.at(x, z) = neighbours >= params.survivalLimit ? DungeonCell::Wall : DungeonCell::Floor;
                else
                    scratch.at(x, z) = neighbours >= params.birthLimit ? DungeonCell::Wall : DungeonCell::Floor;
            }
        }
        // Swap rather than copy: the automaton needs two buffers, not two copies,
        // and a step is O(cells) of real work either way.
        swapMaps(map, scratch);
    }

    // The automaton is blind to topology and routinely seals off pockets.
    (void) connectRegions(map, params.minRegionSize);
    return map;
}

DungeonMap generateDrunkardWalk(const DrunkardParams &params)
{
    if (params.width == 0u || params.depth == 0u)
        return DungeonMap{};

    DungeonMap map{params.width, params.depth, DungeonCell::Wall};
    Random random = deriveStream(params.seed, 0xD24Bu);

    const core::u32 margin = params.margin;
    if (params.width <= margin * 2u + 1u || params.depth <= margin * 2u + 1u)
        return map;

    const core::u32 target =
        static_cast<core::u32>((math::Fixed32::fromFloat(params.targetFill) *
                                math::Fixed32::fromInt(static_cast<core::i32>(map.cellCount())))
                                   .toInt());
    core::u32 carved = 0u;

    for (core::u32 digger = 0u; digger < params.diggers && carved < target; ++digger)
    {
        // Every digger starts mid-map so the galleries meet near the centre
        // rather than fragmenting into disjoint burrows at the edges.
        core::u32 x = params.width / 2u;
        core::u32 z = params.depth / 2u;

        for (core::u32 step = 0u; step < params.stepsPerDigger && carved < target; ++step)
        {
            if (map.at(x, z) == DungeonCell::Wall)
            {
                map.at(x, z) = DungeonCell::Floor;
                ++carved;
            }

            const core::u32 direction = random.below(4u);
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[direction];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[direction];

            // Confinement: a step that would leave the margin is refused rather
            // than clamped, so the walk bounces inward instead of hugging the
            // border (which is where an unconstrained walk spends its time).
            if (nx < static_cast<core::i32>(margin) || nz < static_cast<core::i32>(margin) ||
                nx >= static_cast<core::i32>(params.width - margin) ||
                nz >= static_cast<core::i32>(params.depth - margin))
                continue;

            x = static_cast<core::u32>(nx);
            z = static_cast<core::u32>(nz);
        }
    }

    (void) connectRegions(map, 1u);
    return map;
}

DungeonReport connectRegions(DungeonMap &map, core::u32 minRegionSize)
{
    DungeonReport report;
    if (map.empty())
        return report;

    Grid<core::u32> labels;
    lpl::pmr::vector<core::u32> sizes;
    report.regions = labelRegions(map, labels, sizes);

    if (report.regions == 0u)
        return report;

    // ── Fill the pockets not worth keeping ──────────────────────────────────
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
    {
        const core::u32 label = labels[i];
        if (label == kInvalidRegion)
            continue;
        if (sizes[label] < minRegionSize)
        {
            map[i] = DungeonCell::Wall;
            ++report.pocketsFilled;
        }
    }

    // ── Link what survived ──────────────────────────────────────────────────
    report.regions = labelRegions(map, labels, sizes);
    if (report.regions > 1u)
        report.linksDug = tunnelToMainRegion(map, labels, sizes, report.regions);

    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        if (isWalkable(map[i]))
            ++report.floorCells;

    report.connected = isFullyConnected(map);
    return report;
}

bool isFullyConnected(const DungeonMap &map)
{
    Grid<core::u32> labels;
    lpl::pmr::vector<core::u32> sizes;
    const core::u32 regions = labelRegions(map, labels, sizes);
    return regions <= 1u;
}

void erodeEdges(DungeonMap &map, core::u32 seed, core::f32 strength)
{
    if (map.empty())
        return;

    Random random = deriveStream(seed, 0xE20Du);
    const math::Fixed32 chance = math::Fixed32::fromFloat(strength);
    const DungeonMap source = map;

    for (core::u32 z = 1u; z + 1u < map.depth(); ++z)
    {
        for (core::u32 x = 1u; x + 1u < map.width(); ++x)
        {
            const core::u32 neighbours = rockNeighbours(source, x, z);
            // Only cells on a boundary are eligible: eroding the middle of a
            // room would punch holes, and eroding deep rock would do nothing
            // visible while risking a new pocket.
            const bool onBoundary = neighbours != 0u && neighbours != 8u;
            if (!onBoundary || !random.chance(chance))
                continue;
            map.at(x, z) = source.at(x, z) == DungeonCell::Wall ? DungeonCell::Floor : DungeonCell::Wall;
        }
    }

    // Erosion can seal a corridor or open a pocket; re-establish the guarantee.
    (void) connectRegions(map, 1u);
}

core::u32 forceConnectivity(DungeonMap &map, core::u32 seed)
{
    if (map.empty())
        return 0u;

    Random random = deriveStream(seed, 0xC0FFEEu);
    core::u32 broken = 0u;

    for (;;)
    {
        Grid<core::u32> labels;
        lpl::pmr::vector<core::u32> sizes;
        const core::u32 regions = labelRegions(map, labels, sizes);
        if (regions <= 1u)
            break;

        // Largest component wins the title of "the level"; everything else is an
        // orphan to be let in. Ties go to the lower label so the choice does not
        // depend on the labelling's traversal order.
        core::u32 mainLabel = 0u;
        for (core::u32 r = 1u; r < sizes.size(); ++r)
            if (sizes[r] > sizes[mainLabel])
                mainLabel = r;

        // The thinnest wall between the main component and anything else: a wall
        // cell whose opposite neighbours belong to two different components is a
        // one-cell breach. Preferring those is what makes the repair read as a
        // doorway instead of a gouge.
        core::u32 bestCell = 0u;
        core::u32 bestThickness = 0xFFFFFFFFu;
        core::u32 candidates = 0u;

        for (core::u32 z = 1u; z + 1u < map.depth(); ++z)
        {
            for (core::u32 x = 1u; x + 1u < map.width(); ++x)
            {
                if (isWalkable(map.at(x, z)))
                    continue;

                // Horizontal and vertical breaches of thickness 1.
                for (core::u32 axis = 0u; axis < 2u; ++axis)
                {
                    const core::i32 dx = axis == 0u ? 1 : 0;
                    const core::i32 dz = axis == 0u ? 0 : 1;
                    const core::u32 ax = static_cast<core::u32>(static_cast<core::i32>(x) - dx);
                    const core::u32 az = static_cast<core::u32>(static_cast<core::i32>(z) - dz);
                    const core::u32 bx = static_cast<core::u32>(static_cast<core::i32>(x) + dx);
                    const core::u32 bz = static_cast<core::u32>(static_cast<core::i32>(z) + dz);
                    if (!isWalkable(map.at(ax, az)) || !isWalkable(map.at(bx, bz)))
                        continue;

                    const core::u32 la = labels.at(ax, az);
                    const core::u32 lb = labels.at(bx, bz);
                    if (la == kInvalidRegion || lb == kInvalidRegion || la == lb)
                        continue;
                    if (la != mainLabel && lb != mainLabel)
                        continue;

                    const core::u32 thickness = 1u;
                    // Reservoir choice among equally thin walls, so a level with
                    // a dozen candidate doorways does not always use the
                    // top-left one.
                    ++candidates;
                    if (thickness < bestThickness || (thickness == bestThickness && random.below(candidates) == 0u))
                    {
                        bestThickness = thickness;
                        bestCell = map.index(x, z);
                    }
                }
            }
        }

        if (bestThickness == 0xFFFFFFFFu)
        {
            // No one-cell breach exists. Fall back on the tunneller, which digs a
            // corridor rather than a doorway — uglier, but it always succeeds,
            // and an ugly connection beats an unreachable room.
            const core::u32 dug = tunnelToMainRegion(map, labels, sizes, regions);
            if (dug == 0u)
                break; // Nothing left that can be joined; stop rather than spin.
            broken += dug;
            continue;
        }

        map[bestCell] = DungeonCell::Floor;
        ++broken;
    }
    return broken;
}

core::u32 mergeRoomsAsymmetric(DungeonMap &map, core::u32 seed, core::f32 strength)
{
    if (map.empty())
        return 0u;

    Random random = deriveStream(seed, 0x11E56u);
    const math::Fixed32 chance = math::Fixed32::fromFloat(strength);
    const DungeonMap source = map;
    core::u32 opened = 0u;

    for (core::u32 z = 1u; z + 1u < map.depth(); ++z)
    {
        for (core::u32 x = 1u; x + 1u < map.width(); ++x)
        {
            if (isWalkable(source.at(x, z)))
                continue;

            // A party wall: open on both sides along one axis. Dissolving it
            // merges two rooms; dissolving anything else would punch a hole in
            // the outer shell.
            const bool horizontal = isWalkable(source.at(x - 1u, z)) && isWalkable(source.at(x + 1u, z));
            const bool vertical = isWalkable(source.at(x, z - 1u)) && isWalkable(source.at(x, z + 1u));
            if (!horizontal && !vertical)
                continue;
            if (!random.chance(chance))
                continue;

            map.at(x, z) = DungeonCell::Floor;
            ++opened;
        }
    }
    return opened;
}

core::u32 misalignPillars(DungeonMap &map, core::u32 seed, core::f32 density)
{
    if (map.empty())
        return 0u;

    Random random = deriveStream(seed, 0x9111Au);
    const math::Fixed32 chance = math::Fixed32::fromFloat(density);
    core::u32 placed = 0u;

    for (core::u32 z = 2u; z + 2u < map.depth(); ++z)
    {
        for (core::u32 x = 2u; x + 2u < map.width(); ++x)
        {
            if (!isWalkable(map.at(x, z)) || !random.chance(chance))
                continue;

            // Never in a spot whose four neighbours are not all open: a pillar in
            // a corridor is a plug, and a plug is a softlock waiting for the
            // connectivity pass to notice.
            bool surrounded = true;
            for (core::u32 n = 0u; n < 4u; ++n)
            {
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                if (!isWalkable(map.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz))))
                    surrounded = false;
            }
            if (!surrounded)
                continue;

            map.at(x, z) = DungeonCell::Wall;
            ++placed;
        }
    }
    return placed;
}

core::u32 foldDungeon(const DungeonMap &map)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        hash = (hash ^ static_cast<core::u32>(map[i])) * kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
