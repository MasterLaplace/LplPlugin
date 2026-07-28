/**
 * @file Liminal.cpp
 * @brief Implementation of the layered liminal-space pipeline.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Liminal.hpp>

#include <lpl/procgen/Random.hpp>
#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// How each zone shapes its partition. This table IS the difference between the
/// four zones — everything downstream reads it rather than testing the zone.
struct ZoneShape {
    core::u32 minLeafSize;   ///< Smallest room the partition allows.
    core::u32 corridorWidth; ///< Corridor thickness.
    core::f32 erodeScale;    ///< Multiplier on the erosion strength.
    core::f32 mergeScale;    ///< Multiplier on the wall-dissolution strength.
};

constexpr ZoneShape kZoneShapes[static_cast<core::u32>(LiminalZone::Count)] = {
    {5u,  1u, 1.30f, 0.40f}, // Corridor: small rooms, thin links, heavily chewed.
    {8u,  1u, 0.70f, 1.00f}, // Office: the uncanny baseline — regular, lightly merged.
    {18u, 3u, 0.50f, 1.60f}, // Hall: few big rooms, walls mostly dissolved.
    {13u, 2u, 0.35f, 1.20f}, // Pool: wide, shallow partitions, barely eroded.
};

/// The zone that covers most of the map, which decides the partition's shape.
LiminalZone dominantZone(const LiminalZoneMap &zones)
{
    core::u32 counts[static_cast<core::u32>(LiminalZone::Count)] = {};
    for (core::u32 i = 0u; i < zones.cellCount(); ++i)
    {
        const core::u32 index = static_cast<core::u32>(zones[i]);
        if (index < static_cast<core::u32>(LiminalZone::Count))
            ++counts[index];
    }
    core::u32 best = 0u;
    for (core::u32 i = 1u; i < static_cast<core::u32>(LiminalZone::Count); ++i)
        if (counts[i] > counts[best])
            best = i;
    return static_cast<LiminalZone>(best);
}

/**
 * @brief Runs the pipeline on an already-zoned map.
 *
 * Shared by the bounded and the chunked entry points, because they must produce
 * the same space from the same inputs — two spellings of the pipeline would drift
 * and a sector would stop matching the map it belongs to.
 */
LiminalSpace buildFromZones(const LiminalParams &params, LiminalZoneMap zones, core::u32 seed)
{
    LiminalSpace space;
    space.zones = zones;

    const LiminalZone zone = dominantZone(space.zones);
    const ZoneShape &shape = kZoneShapes[static_cast<core::u32>(zone)];

    // ── 1. Partition: rigid order, laid down first ──────────────────────────
    BspDungeonParams bsp;
    bsp.width = params.width;
    bsp.depth = params.depth;
    bsp.seed = seed;
    bsp.maxDepth = params.bspDepth;
    bsp.minLeafSize = shape.minLeafSize;
    bsp.corridorWidth = shape.corridorWidth;
    space.map = generateBspDungeon(bsp, nullptr);

    // ── 2. Erode: a process that knows nothing about the partition ──────────
    erodeEdges(space.map, seed ^ 0x1E20Du, params.erosionStrength * shape.erodeScale);
    space.wallsDissolved = mergeRoomsAsymmetric(space.map, seed ^ 0x2E20Du, params.mergeStrength * shape.mergeScale);
    space.pillars = misalignPillars(space.map, seed ^ 0x3E20Du, params.pillarDensity);

    // ── 3. Repair: the guarantee, after everything that could break it ──────
    //
    // Last, not first. Every erosion step above can seal a corridor or strand a
    // room, so a repair run any earlier would be repairing a map that no longer
    // exists by the time a player walks it.
    space.wallsBroken = forceConnectivity(space.map, seed ^ 0x4E20Du);
    space.connected = isFullyConnected(space.map);

    for (core::u32 i = 0u; i < space.map.cellCount(); ++i)
        if (isWalkable(space.map[i]))
            ++space.openCells;

    // ── 4. Read the topology back, and place things by what it says ─────────
    core::u32 startX = 0u;
    core::u32 startZ = 0u;
    core::u32 goalX = 0u;
    core::u32 goalZ = 0u;
    if (!findFarthestPair(space.map, startX, startZ, goalX, goalZ))
        return space;

    const HotPathAnalysis path = analyseHotPath(space.map, startX, startZ, goalX, goalZ, 0u);
    if (!path.valid)
        return space;

    // Events go ON the route, because that is where a player will be. Spacing
    // them evenly along it rather than at random keeps the pacing readable —
    // three anomalies in the same corridor and then nothing is not tension.
    lpl::pmr::vector<core::u32> route;
    for (core::u32 i = 0u; i < path.onPath.cellCount(); ++i)
        if (path.onPath[i] != 0u)
            route.push_back(i);

    if (params.hotPathEvents > 0u && !route.empty())
    {
        const core::u32 stride = static_cast<core::u32>(route.size()) / params.hotPathEvents;
        for (core::u32 e = 0u; e < params.hotPathEvents; ++e)
        {
            const core::u32 index = stride > 0u ? e * stride : e;
            if (index < route.size())
                space.eventSites.push_back(route[index]);
        }
    }

    // Rewards go as FAR from the route as the level allows. The survey makes the
    // same point twice over: the cell that maximises the detour distance is both
    // the generator's worst structural offence and the perfect place to hide
    // something, which turns a topological defect into a reason to explore it.
    if (params.secretSites > 0u)
    {
        lpl::pmr::vector<core::u32> byDepth;
        for (core::u32 i = 0u; i < space.map.cellCount(); ++i)
            if (isWalkable(space.map[i]) && path.detour[i] != kUnreachable && path.detour[i] > 0u)
                byDepth.push_back(i);

        // Selection of the deepest few, by repeated scan rather than a sort: the
        // count is a handful and a sort would need a comparator whose ties are
        // resolved somehow — this way the tie rule is "lower index", visibly.
        for (core::u32 s = 0u; s < params.secretSites; ++s)
        {
            core::u32 best = 0xFFFFFFFFu;
            core::u32 bestDepth = 0u;
            for (core::u32 i = 0u; i < byDepth.size(); ++i)
            {
                const core::u32 cell = byDepth[i];
                if (cell == 0xFFFFFFFFu)
                    continue;
                if (path.detour[cell] > bestDepth)
                {
                    bestDepth = path.detour[cell];
                    best = i;
                }
            }
            if (best == 0xFFFFFFFFu)
                break;
            space.secretSites.push_back(byDepth[best]);
            byDepth[best] = 0xFFFFFFFFu;
        }
    }

    return space;
}

} // namespace

LiminalZoneMap zoneMap(const LiminalParams &params)
{
    LiminalZoneMap zones{params.width, params.depth, LiminalZone::Office};
    if (zones.empty())
        return zones;

    const core::u32 longAxis = params.width > params.depth ? params.width : params.depth;
    const math::Fixed32 frequency =
        math::Fixed32::fromFloat(params.zoneBelts) / math::Fixed32::fromInt(static_cast<core::i32>(longAxis));

    for (core::u32 z = 0u; z < params.depth; ++z)
    {
        for (core::u32 x = 0u; x < params.width; ++x)
        {
            const math::Fixed32 n = ValueNoise2D::fbm(math::Fixed32::fromInt(static_cast<core::i32>(x)) * frequency,
                                                      math::Fixed32::fromInt(static_cast<core::i32>(z)) * frequency,
                                                      params.zoneOctaves, params.zoneSeed);
            // [-1, 1] into one of four bands, by quarters. Even bands rather than
            // tuned ones: a zone that covers a tenth of every map is a zone a
            // player will never knowingly visit.
            const math::Fixed32 unit = (n + math::Fixed32::one()) * math::Fixed32::half();
            core::i32 band = (unit * math::Fixed32::fromInt(static_cast<core::i32>(LiminalZone::Count))).toInt();
            if (band < 0)
                band = 0;
            if (band >= static_cast<core::i32>(LiminalZone::Count))
                band = static_cast<core::i32>(LiminalZone::Count) - 1;
            zones.at(x, z) = static_cast<LiminalZone>(band);
        }
    }
    return zones;
}

LiminalSpace generateLiminal(const LiminalParams &params)
{
    if (params.width == 0u || params.depth == 0u)
        return LiminalSpace{};
    return buildFromZones(params, zoneMap(params), params.seed);
}

LiminalSpace generateLiminalChunk(const LiminalParams &params, ChunkCoord coord)
{
    if (params.width == 0u)
        return LiminalSpace{};

    // The chunk's own seed, derived from the world seed and the coordinates. No
    // running state: a sector is a pure function of where it is, which is what
    // lets two players in opposite corners agree without a message.
    ChunkParams chunkParams;
    chunkParams.size = params.width;
    chunkParams.worldSeed = params.seed;
    const core::u32 seed = chunkSeed(chunkParams, coord);

    // The zoning field is sampled at ABSOLUTE world coordinates, so a zone that
    // straddles two sectors is one zone rather than two that happen to touch.
    LiminalParams local = params;
    local.depth = params.width;
    LiminalZoneMap zones{params.width, params.width, LiminalZone::Office};

    const math::Fixed32 frequency =
        math::Fixed32::fromFloat(params.zoneBelts) / math::Fixed32::fromInt(static_cast<core::i32>(params.width) * 8);
    const core::i32 originX = coord.x * static_cast<core::i32>(params.width);
    const core::i32 originZ = coord.z * static_cast<core::i32>(params.width);

    for (core::u32 z = 0u; z < params.width; ++z)
    {
        for (core::u32 x = 0u; x < params.width; ++x)
        {
            const core::i32 worldX = originX + static_cast<core::i32>(x);
            const core::i32 worldZ = originZ + static_cast<core::i32>(z);
            const math::Fixed32 n =
                ValueNoise2D::fbm(math::Fixed32::fromInt(worldX) * frequency,
                                  math::Fixed32::fromInt(worldZ) * frequency, params.zoneOctaves, params.zoneSeed);
            const math::Fixed32 unit = (n + math::Fixed32::one()) * math::Fixed32::half();
            core::i32 band = (unit * math::Fixed32::fromInt(static_cast<core::i32>(LiminalZone::Count))).toInt();
            if (band < 0)
                band = 0;
            if (band >= static_cast<core::i32>(LiminalZone::Count))
                band = static_cast<core::i32>(LiminalZone::Count) - 1;
            zones.at(x, z) = static_cast<LiminalZone>(band);
        }
    }

    return buildFromZones(local, zones, seed);
}

TileGrid presetFromMask(const Grid<core::u8> &mask, core::u8 pinnedTile)
{
    TileGrid preset{mask.width(), mask.depth(), kNoTile};
    for (core::u32 i = 0u; i < mask.cellCount(); ++i)
        if (mask[i] != 0u)
            preset[i] = pinnedTile;
    return preset;
}

const char *liminalZoneName(LiminalZone zone) noexcept
{
    switch (zone)
    {
    case LiminalZone::Corridor: return "corridor";
    case LiminalZone::Office: return "office";
    case LiminalZone::Hall: return "hall";
    case LiminalZone::Pool: return "pool";
    case LiminalZone::Count: break;
    }
    return "unknown";
}

core::u32 foldLiminal(const LiminalSpace &space)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < space.map.cellCount(); ++i)
        hash = (hash ^ static_cast<core::u32>(space.map[i])) * kFnv1aPrime;
    for (core::u32 i = 0u; i < space.zones.cellCount(); ++i)
        hash = (hash ^ static_cast<core::u32>(space.zones[i])) * kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
