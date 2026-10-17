#include <lpl/procgen/CaveSystem.hpp>

#include <lpl/math/Random.hpp>
#include <lpl/procgen/Aggregation.hpp>
#include <lpl/procgen/QualityGate.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/** A cell of the stack, as one index. Layer-major so a layer stays contiguous. */
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

/**
 * @brief Is there a shaft joining these two layers at this cell?
 */
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
 * @brief Flood-fills the stack from every entrance, in steps.
 *
 * Movement is 4-connected within a layer and vertical only where a shaft exists —
 * which is exactly what a body can do, and what makes the count mean "reachable"
 * rather than "present".
 *
 * It carries DISTANCES rather than a visited flag, because the two questions the
 * stack has to answer — "can a body get everywhere?" and "how far in is the far end?"
 * — are the same traversal. Two traversals would be two chances to disagree about
 * what a shaft is.
 *
 * @param outDistance Steps from the nearest entrance; @ref kUnreachable where none.
 * @return Cells actually reached.
 */
core::u32 floodFromEntrances(const CaveSystem &system, lpl::pmr::vector<core::u32> &outDistance)
{
    const core::u32 perLayer = system.layer[0].cellCount();
    outDistance.clear();
    outDistance.resize(static_cast<core::usize>(perLayer) * system.layerCount, kUnreachable);

    lpl::pmr::vector<core::u32> queue;
    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const CaveShaft &shaft = system.shafts[i];
        if (!shaft.surface)
            continue;
        const core::u32 cell = system.layer[0].index(shaft.x, shaft.z);
        const core::u32 flat = flatten(system, shaft.upperLayer, cell);
        if (outDistance[flat] == kUnreachable && isWalkable(system.layer[shaft.upperLayer][cell]))
        {
            outDistance[flat] = 0u;
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
            if (outDistance[next] != kUnreachable)
                continue;
            outDistance[next] = outDistance[flat] + 1u;
            queue.push_back(next);
        }

        if (here.layer + 1u < system.layerCount && hasShaft(system, here.layer, here.layer + 1u, here.cell) &&
            isWalkable(system.layer[here.layer + 1u][here.cell]))
        {
            const core::u32 next = flatten(system, here.layer + 1u, here.cell);
            if (outDistance[next] == kUnreachable)
            {
                outDistance[next] = outDistance[flat] + 1u;
                queue.push_back(next);
            }
        }
        if (here.layer > 0u && hasShaft(system, here.layer - 1u, here.layer, here.cell) &&
            isWalkable(system.layer[here.layer - 1u][here.cell]))
        {
            const core::u32 next = flatten(system, here.layer - 1u, here.cell);
            if (outDistance[next] == kUnreachable)
            {
                outDistance[next] = outDistance[flat] + 1u;
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

    lpl::pmr::vector<core::u32> distance;
    system.reachableCells = floodFromEntrances(system, distance);
}

/**
 * @brief Which generator a layer gets, and why it is a function of the parameters alone.
 *
 * A stack whose layers all came out of one automaton is one cave drawn three times.
 * Choosing the generator per layer is what makes a descent read as going somewhere: a
 * natural mouth over a warren of rooms is a place with a history, and both halves
 * already exist in this module.
 *
 * The choice is over (parameters, layer index) and nothing else, so the same stack comes
 * back the same way on any target and in any order — the property the whole module rests
 * on.
 *
 * Without @ref CaveSystemParams::mixLayerKinds, neither of the two non-fillings may reach
 * a generator. Clamping rather than switching on them is deliberate: a stack inside a
 * stack has no meaning, and an unresolved Auto is a caller that forgot
 * @ref chooseCaveKind — both are better as the default cave than as an unhandled case.
 *
 * With it, the shallowest floor is always the natural one: it is the floor a mouth opens
 * into, and a corridor of dressed rooms starting at a hole in a hillside reads as a
 * mistake rather than as a ruin. Below that, the same evidence @ref chooseCaveKind reads,
 * applied a floor at a time so a descent can pass through more than one thing. What is
 * dug sits under what is lived on and over what dissolved long before either.
 *
 * @param params Generator selection.
 * @param layer  Which layer, shallowest first.
 * @return The kind to fill it with; never @c Layered.
 */
[[nodiscard]] CaveKind kindForLayer(const CaveSystemParams &params, core::u32 layer)
{
    if (!params.mixLayerKinds)
    {
        if (params.layerKind == CaveKind::Layered || params.layerKind == CaveKind::Auto)
            return CaveKind::Cellular;
        return params.layerKind;
    }

    if (layer == 0u)
        return CaveKind::Cellular;

    const bool deepest = layer + 1u >= params.layers;
    if (params.settled && !deepest)
        return CaveKind::Bsp;
    if (params.wetness >= kKarstWetness)
        return CaveKind::Dla;
    return params.settled ? CaveKind::Bsp : CaveKind::Cellular;
}

/**
 * @brief Fills one layer with the generator its kind names.
 *
 * The three cases are three of the four generators a document can already spell,
 * called with the budget this stack was given rather than with a second set of
 * parameters — a layer of a warren is not a bounded dungeon and has no business
 * carrying one.
 *
 * Each layer gets its own stream, derived from its index. Reusing one stream across
 * layers would make layer 2 a continuation of layer 1's sequence, so changing the layer
 * count would change every layer.
 *
 * The aggregation's particle budget is scaled to the floor's AREA. The default is tuned
 * for the 64-cell map the struct declares, and a particle budget left at that on a
 * forty-cell warren packs the fractal solid — a branching cave stops branching and
 * becomes a room with a ragged edge.
 *
 * @param params Stack parameters.
 * @param layer  Layer index, shallowest first.
 * @param fill   Fill probability for this depth.
 * @return The plan.
 */
[[nodiscard]] DungeonMap generateLayer(const CaveSystemParams &params, core::u32 layer, core::f32 fill)
{
    const core::u32 seed = params.seed ^ (0x1A7E5u * (layer + 1u));

    switch (kindForLayer(params, layer))
    {
    case CaveKind::Bsp: {
        BspDungeonParams rooms;
        rooms.width = params.width;
        rooms.depth = params.depth;
        rooms.seed = seed;
        return generateBspDungeon(rooms);
    }
    case CaveKind::Dla: {
        DlaParams aggregation;
        aggregation.width = params.width;
        aggregation.depth = params.depth;
        aggregation.seed = seed;
        aggregation.particles = (params.width * params.depth) / 5u;
        return generateDlaCave(aggregation);
    }
    case CaveKind::Cellular:
    case CaveKind::Layered:
    default: break;
    }

    CaveParams cave;
    cave.width = params.width;
    cave.depth = params.depth;
    cave.seed = seed;
    cave.fillProbability = fill;
    cave.steps = params.automatonSteps;
    cave.minRegionSize = params.minChamberSize;
    return generateCellularCave(cave);
}

} // namespace

void recountCaveReachability(CaveSystem &system) { recountReachability(system); }

CaveKind chooseCaveKind(const CaveContext &context)
{
    if (context.layerCount > 1u && context.settled && context.wetness >= kKarstWetness)
        return CaveKind::Layered;
    if (context.settled)
        return CaveKind::Bsp;
    if (context.wetness >= kKarstWetness)
        return CaveKind::Dla;
    return CaveKind::Cellular;
}

CaveSystem generateCaveSystem(const CaveSystemParams &params, const Heightfield &surface, const BiomeMap *biomes)
{
    CaveSystem system;
    if (params.width == 0u || params.depth == 0u || params.layers == 0u)
        return system;

    system.layerCount = params.layers > kMaxCaveLayers ? kMaxCaveLayers : params.layers;

    for (core::u32 l = 0u; l < system.layerCount; ++l)
    {
        const math::Fixed32 t = system.layerCount <= 1u ?
                                    math::Fixed32::zero() :
                                    math::Fixed32::fromInt(static_cast<core::i32>(l)) /
                                        math::Fixed32::fromInt(static_cast<core::i32>(system.layerCount - 1u));
        const math::Fixed32 top = math::Fixed32::fromFloat(params.topFill);
        const math::Fixed32 deep = math::Fixed32::fromFloat(params.deepFill);

        system.layer[l] = generateLayer(params, l, (top + (deep - top) * t).toFloat());
    }

    math::Random random{params.seed ^ 0x5AF75u};
    const DungeonMap &top = system.layer[0];

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

LevelQuality evaluateCaveSystem(const CaveSystem &system)
{
    LevelQuality quality{};
    if (system.layerCount == 0u || system.layer[0].empty())
        return quality;

    const core::u32 width = system.layer[0].width();
    const core::u32 perLayer = system.layer[0].cellCount();

    lpl::pmr::vector<core::u32> distance;
    quality.reachableCells = floodFromEntrances(system, distance);

    for (core::u32 l = 0u; l < system.layerCount; ++l)
        for (core::u32 cell = 0u; cell < perLayer; ++cell)
            if (isWalkable(system.layer[l][cell]))
                ++quality.walkableCells;

    quality.fullyConnected = quality.walkableCells != 0u && quality.reachableCells == quality.walkableCells;

    const core::u32 deepest = system.layerCount - 1u;
    for (core::u32 l = 0u; l < system.layerCount; ++l)
    {
        const DungeonMap &plan = system.layer[l];
        for (core::u32 cell = 0u; cell < perLayer; ++cell)
        {
            if (!isWalkable(plan[cell]))
                continue;

            const core::u32 x = cell % width;
            const core::u32 z = cell / width;
            core::u32 ways = 0u;
            for (core::u32 n = 0u; n < 4u; ++n)
            {
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                if (plan.contains(nx, nz) &&
                    isWalkable(plan[plan.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz))]))
                    ++ways;
            }
            if (l + 1u < system.layerCount && hasShaft(system, l, l + 1u, cell) &&
                isWalkable(system.layer[l + 1u][cell]))
                ++ways;
            if (l > 0u && hasShaft(system, l - 1u, l, cell) && isWalkable(system.layer[l - 1u][cell]))
                ++ways;

            quality.deadEnds += ways == 1u ? 1u : 0u;
            quality.junctions += ways >= 3u ? 1u : 0u;

            const core::u32 steps = distance[flatten(system, l, cell)];
            if (steps == kUnreachable)
                continue;
            if (steps > quality.longestDistance)
                quality.longestDistance = steps;
            if (l == deepest)
            {
                quality.goalReachable = true;
                if (steps > quality.pathLength)
                    quality.pathLength = steps;
            }
        }
    }

    return quality;
}

core::u32 repairCaveReachability(CaveSystem &system, core::u32 seed)
{
    if (system.layerCount == 0u || system.layer[0].empty())
        return 0u;
    if (system.entranceCount == 0u)
        return 0u;

    math::Random random{seed};
    core::u32 opened = 0u;
    const core::u32 perLayer = system.layer[0].cellCount();
    const core::u32 width = system.layer[0].width();

    for (core::u32 round = 0u; round < perLayer; ++round)
    {
        lpl::pmr::vector<core::u32> visited;
        const core::u32 reached = floodFromEntrances(system, visited);
        if (reached == system.hollowCells)
            break;

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
                if (visited[flat] != kUnreachable || !isWalkable(plan[cell]))
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
                    if (visited[flatten(system, l, other)] == kUnreachable)
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
                    if (visited[flatten(system, other, cell)] == kUnreachable)
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
            for (core::u32 l = 0u; l < system.layerCount; ++l)
                for (core::u32 cell = 0u; cell < perLayer; ++cell)
                    if (visited[flatten(system, l, cell)] == kUnreachable && isWalkable(system.layer[l][cell]))
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
