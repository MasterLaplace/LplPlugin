/**
 * @file CaveSystem.cpp
 * @brief Implementation of the stacked cave system.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/CaveSystem.hpp>

#include <lpl/procgen/Random.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// A cell of the stack, as one index. Layer-major so a layer stays contiguous.
struct StackIndex {
    core::u32 layer;
    core::u32 cell;
};

[[nodiscard]] core::u32 flatten(const CaveSystem &system, core::u32 layer, core::u32 cell)
{
    return layer * system.layer[0].cellCount() + cell;
}

[[nodiscard]] StackIndex unflatten(const CaveSystem &system, core::u32 flat)
{
    const core::u32 perLayer = system.layer[0].cellCount();
    return StackIndex{flat / perLayer, flat % perLayer};
}

/// Is there a shaft joining these two layers at this cell?
[[nodiscard]] bool hasShaft(const CaveSystem &system, core::u32 upper, core::u32 lower, core::u32 cell)
{
    const core::u32 width = system.layer[0].width();
    const core::u32 x = cell % width;
    const core::u32 z = cell / width;
    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const CaveShaft &shaft = system.shafts[i];
        if (shaft.x == x && shaft.z == z && shaft.upperLayer == upper && shaft.lowerLayer == lower)
            return true;
    }
    return false;
}

/**
 * @brief Flood-fills the stack from every entrance, counting what it can reach.
 *
 * Movement is 4-connected within a layer and vertical only where a shaft exists —
 * which is exactly what a body can do, and what makes the count mean "reachable"
 * rather than "present".
 */
core::u32 floodFromEntrances(const CaveSystem &system, lpl::pmr::vector<core::u8> &visited)
{
    const core::u32 perLayer = system.layer[0].cellCount();
    visited.clear();
    visited.resize(static_cast<core::usize>(perLayer) * system.layerCount, core::u8{0});

    lpl::pmr::vector<core::u32> queue;
    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const CaveShaft &shaft = system.shafts[i];
        if (!shaft.surface)
            continue;
        const core::u32 cell = system.layer[0].index(shaft.x, shaft.z);
        const core::u32 flat = flatten(system, shaft.upperLayer, cell);
        if (visited[flat] == 0u && isWalkable(system.layer[shaft.upperLayer][cell]))
        {
            visited[flat] = 1u;
            queue.push_back(flat);
        }
    }

    core::u32 reached = 0u;
    core::u32 head = 0u;
    while (head < queue.size())
    {
        const core::u32 flat = queue[head++];
        ++reached;
        const StackIndex here = unflatten(system, flat);
        const DungeonMap &plan = system.layer[here.layer];
        const core::u32 x = here.cell % plan.width();
        const core::u32 z = here.cell / plan.width();

        for (core::u32 n = 0u; n < 4u; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
            if (!plan.contains(nx, nz))
                continue;
            const core::u32 cell = plan.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (!isWalkable(plan[cell]))
                continue;
            const core::u32 next = flatten(system, here.layer, cell);
            if (visited[next] != 0u)
                continue;
            visited[next] = 1u;
            queue.push_back(next);
        }

        // Vertical moves, only where a shaft was cut.
        if (here.layer + 1u < system.layerCount && hasShaft(system, here.layer, here.layer + 1u, here.cell) &&
            isWalkable(system.layer[here.layer + 1u][here.cell]))
        {
            const core::u32 next = flatten(system, here.layer + 1u, here.cell);
            if (visited[next] == 0u)
            {
                visited[next] = 1u;
                queue.push_back(next);
            }
        }
        if (here.layer > 0u && hasShaft(system, here.layer - 1u, here.layer, here.cell) &&
            isWalkable(system.layer[here.layer - 1u][here.cell]))
        {
            const core::u32 next = flatten(system, here.layer - 1u, here.cell);
            if (visited[next] == 0u)
            {
                visited[next] = 1u;
                queue.push_back(next);
            }
        }
    }
    return reached;
}

void recountReachability(CaveSystem &system)
{
    system.hollowCells = 0u;
    for (core::u32 l = 0u; l < system.layerCount; ++l)
        for (core::u32 i = 0u; i < system.layer[l].cellCount(); ++i)
            if (isWalkable(system.layer[l][i]))
                ++system.hollowCells;

    lpl::pmr::vector<core::u8> visited;
    system.reachableCells = floodFromEntrances(system, visited);
}

} // namespace

CaveSystem generateCaveSystem(const CaveSystemParams &params, const Heightfield &surface, const BiomeMap *biomes)
{
    CaveSystem system;
    if (params.width == 0u || params.depth == 0u || params.layers == 0u)
        return system;

    system.layerCount = params.layers > kMaxCaveLayers ? kMaxCaveLayers : params.layers;

    // ── Layers: the same generator, told to open out with depth ─────────────
    for (core::u32 l = 0u; l < system.layerCount; ++l)
    {
        const math::Fixed32 t = system.layerCount <= 1u ?
                                    math::Fixed32::zero() :
                                    math::Fixed32::fromInt(static_cast<core::i32>(l)) /
                                        math::Fixed32::fromInt(static_cast<core::i32>(system.layerCount - 1u));
        const math::Fixed32 top = math::Fixed32::fromFloat(params.topFill);
        const math::Fixed32 deep = math::Fixed32::fromFloat(params.deepFill);

        CaveParams cave;
        cave.width = params.width;
        cave.depth = params.depth;
        // Each layer gets its own stream, derived from its index. Reusing one
        // stream across layers would make layer 2 a continuation of layer 1's
        // sequence, so changing the layer count would change every layer.
        cave.seed = params.seed ^ (0x1A7E5u * (l + 1u));
        cave.fillProbability = (top + (deep - top) * t).toFloat();
        cave.steps = params.automatonSteps;
        cave.minRegionSize = params.minChamberSize;
        system.layer[l] = generateCellularCave(cave);
    }

    Random random{params.seed ^ 0x5AF75u};
    const DungeonMap &top = system.layer[0];

    // ── Shafts between layers ───────────────────────────────────────────────
    //
    // A shaft is valid only where BOTH layers are hollow at the same (x, z).
    // That is the whole reason this cannot be done inside a layer generator: it
    // is the one fact a single plan does not contain. Candidates are gathered
    // first and drawn from, rather than rejection-sampled, so a stack whose
    // layers barely overlap still gets its shafts instead of silently getting
    // none.
    for (core::u32 upper = 0u; upper + 1u < system.layerCount; ++upper)
    {
        lpl::pmr::vector<core::u32> candidates;
        for (core::u32 i = 0u; i < top.cellCount(); ++i)
            if (isWalkable(system.layer[upper][i]) && isWalkable(system.layer[upper + 1u][i]))
                candidates.push_back(i);
        if (candidates.empty())
            continue;

        for (core::u32 s = 0u; s < params.shaftsPerPair; ++s)
        {
            const core::u32 cell = candidates[random.below(static_cast<core::u32>(candidates.size()))];
            if (hasShaft(system, upper, upper + 1u, cell))
                continue;
            CaveShaft shaft;
            shaft.x = cell % top.width();
            shaft.z = cell / top.width();
            shaft.upperLayer = upper;
            shaft.lowerLayer = upper + 1u;
            system.shafts.push_back(shaft);
        }
    }

    // ── Entrances: shafts that pierce the surface ───────────────────────────
    //
    // Without these the system is a sealed void. An entrance needs hollow ground
    // beneath it AND ground above it a body could stand on — steep rock and the
    // sea floor are both places a cave mouth does not belong.
    if (!surface.empty() && surface.width() == params.width && surface.depth() == params.depth)
    {
        const math::Fixed32 maxSlope = math::Fixed32::fromFloat(params.entranceMaxSlope);
        lpl::pmr::vector<core::u32> candidates;
        for (core::u32 z = 0u; z < params.depth; ++z)
        {
            for (core::u32 x = 0u; x < params.width; ++x)
            {
                const core::u32 cell = top.index(x, z);
                if (!isWalkable(system.layer[0][cell]))
                    continue;
                if (slopeAt(surface, x, z) > maxSlope)
                    continue;
                if (biomes != nullptr && biomes->width() == params.width && biomes->depth() == params.depth &&
                    (isWater(biomes->at(x, z)) || biomes->at(x, z) == BiomeId::Snow))
                    continue;
                candidates.push_back(cell);
            }
        }

        for (core::u32 e = 0u; e < params.entrances && !candidates.empty(); ++e)
        {
            const core::u32 cell = candidates[random.below(static_cast<core::u32>(candidates.size()))];
            CaveShaft shaft;
            shaft.x = cell % top.width();
            shaft.z = cell / top.width();
            shaft.upperLayer = 0u;
            shaft.lowerLayer = 0u;
            shaft.surface = true;
            system.shafts.push_back(shaft);
            ++system.entranceCount;
        }
    }

    recountReachability(system);
    system.repairedCells = repairCaveReachability(system, params.seed ^ 0x3E9A18u);
    return system;
}

core::u32 repairCaveReachability(CaveSystem &system, core::u32 seed)
{
    if (system.layerCount == 0u || system.layer[0].empty())
        return 0u;
    if (system.entranceCount == 0u)
        return 0u; // Nothing to be reachable FROM; the caller wanted a sealed system.

    Random random{seed};
    core::u32 opened = 0u;
    const core::u32 perLayer = system.layer[0].cellCount();
    const core::u32 width = system.layer[0].width();

    // Bounded: each round joins at least one component, and there are at most as
    // many components as cells. The bound is what stops a pathological stack from
    // spinning here forever — which matters because this runs at world build.
    for (core::u32 round = 0u; round < perLayer; ++round)
    {
        lpl::pmr::vector<core::u8> visited;
        const core::u32 reached = floodFromEntrances(system, visited);
        if (reached == system.hollowCells)
            break;

        // Find an unreached hollow cell adjacent — horizontally or vertically —
        // to a reached one, and open the way. Vertically means cutting a shaft,
        // which is the move a flat repair cannot make and the reason this is not
        // just forceConnectivity run per layer.
        bool joined = false;
        core::u32 bestFlat = 0u;
        core::u32 bestNeighbour = 0u;
        bool bestVertical = false;
        core::u32 candidates = 0u;

        for (core::u32 l = 0u; l < system.layerCount && !joined; ++l)
        {
            const DungeonMap &plan = system.layer[l];
            for (core::u32 cell = 0u; cell < perLayer; ++cell)
            {
                const core::u32 flat = flatten(system, l, cell);
                if (visited[flat] != 0u || !isWalkable(plan[cell]))
                    continue;

                const core::u32 x = cell % width;
                const core::u32 z = cell / width;

                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                    if (!plan.contains(nx, nz))
                        continue;
                    const core::u32 other = plan.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                    if (visited[flatten(system, l, other)] == 0u)
                        continue;
                    ++candidates;
                    if (random.below(candidates) == 0u)
                    {
                        bestFlat = flat;
                        bestNeighbour = other;
                        bestVertical = false;
                    }
                }

                for (core::u32 dir = 0u; dir < 2u; ++dir)
                {
                    const core::u32 other = dir == 0u ? l - 1u : l + 1u;
                    if ((dir == 0u && l == 0u) || (dir == 1u && l + 1u >= system.layerCount))
                        continue;
                    if (visited[flatten(system, other, cell)] == 0u)
                        continue;
                    ++candidates;
                    if (random.below(candidates) == 0u)
                    {
                        bestFlat = flat;
                        bestNeighbour = other;
                        bestVertical = true;
                    }
                }
            }
        }

        if (candidates == 0u)
        {
            // No unreached cell touches a reached one on any axis. Nothing short
            // of tunnelling would join them, and a tunnel through solid rock
            // between two unrelated chambers reads worse than a sealed pocket —
            // so fill the orphans in instead, and say how many.
            for (core::u32 l = 0u; l < system.layerCount; ++l)
                for (core::u32 cell = 0u; cell < perLayer; ++cell)
                    if (visited[flatten(system, l, cell)] == 0u && isWalkable(system.layer[l][cell]))
                    {
                        system.layer[l][cell] = DungeonCell::Wall;
                        ++opened;
                    }
            recountReachability(system);
            break;
        }

        const StackIndex target = unflatten(system, bestFlat);
        if (bestVertical)
        {
            CaveShaft shaft;
            shaft.x = target.cell % width;
            shaft.z = target.cell / width;
            shaft.upperLayer = target.layer < bestNeighbour ? target.layer : bestNeighbour;
            shaft.lowerLayer = target.layer < bestNeighbour ? bestNeighbour : target.layer;
            system.shafts.push_back(shaft);
        }
        else
        {
            // The two cells are already both hollow and adjacent — being
            // unreached means the flood had not got here yet on a previous round,
            // so simply re-running the flood is the whole repair.
            (void) bestNeighbour;
        }
        ++opened;
        joined = true;
        (void) joined;

        recountReachability(system);
    }

    recountReachability(system);
    return opened;
}

VoxelVolume caveVolume(const CaveSystem &system, const CaveSystemParams &params, core::u8 rock)
{
    VoxelVolume volume;
    if (system.layerCount == 0u || system.layer[0].empty())
        return volume;

    const core::u32 perLayerLevels = params.levelsPerLayer == 0u ? 1u : params.levelsPerLayer;
    volume.width = system.layer[0].width();
    volume.depth = system.layer[0].depth();
    volume.levels = system.layerCount * perLayerLevels;
    volume.cells.resize(static_cast<core::usize>(volume.width) * volume.depth * volume.levels, core::u8{0});

    for (core::u32 l = 0u; l < system.layerCount; ++l)
    {
        // Layer 0 at the TOP of the volume: the array is indexed by depth, the
        // volume by height, and quietly conflating the two puts the deepest
        // cavern in the sky.
        const core::u32 base = (system.layerCount - 1u - l) * perLayerLevels;
        for (core::u32 z = 0u; z < volume.depth; ++z)
            for (core::u32 x = 0u; x < volume.width; ++x)
            {
                if (isWalkable(system.layer[l].at(x, z)))
                    continue;
                for (core::u32 y = 0u; y < perLayerLevels; ++y)
                    volume.at(x, base + y, z) = rock;
            }
    }

    // Shafts punch through the rock between the layers they join, otherwise the
    // link exists in the graph and not in the geometry — and a player would walk
    // into a ceiling the pathfinder swears is a hole.
    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const CaveShaft &shaft = system.shafts[i];
        const core::u32 upperBase = (system.layerCount - 1u - shaft.upperLayer) * perLayerLevels;
        const core::u32 lowerBase = (system.layerCount - 1u - shaft.lowerLayer) * perLayerLevels;
        const core::u32 from = lowerBase < upperBase ? lowerBase : upperBase;
        const core::u32 to = upperBase + perLayerLevels;
        for (core::u32 y = from; y < to && y < volume.levels; ++y)
            volume.at(shaft.x, y, shaft.z) = 0u;
    }
    return volume;
}

core::u32 foldCaveSystem(const CaveSystem &system)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 l = 0u; l < system.layerCount; ++l)
        for (core::u32 i = 0u; i < system.layer[l].cellCount(); ++i)
            hash = (hash ^ static_cast<core::u32>(system.layer[l][i])) * kFnv1aPrime;
    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const CaveShaft &shaft = system.shafts[i];
        hash = (hash ^ shaft.x) * kFnv1aPrime;
        hash = (hash ^ shaft.z) * kFnv1aPrime;
        hash = (hash ^ shaft.upperLayer) * kFnv1aPrime;
        hash = (hash ^ shaft.lowerLayer) * kFnv1aPrime;
        hash = (hash ^ (shaft.surface ? 1u : 0u)) * kFnv1aPrime;
    }
    return hash;
}

} // namespace lpl::procgen
