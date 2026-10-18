#include <lpl/procgen/Liminal.hpp>

#include <cstdio>

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    if (!condition)
    {
        ++gFailures;
        std::printf("  FAIL %s\n", what);
    }
}

/**
 * @brief Every liminal sector is fully connected, at every seed.
 *
 * How often the repair actually fires is the interesting number: never would mean the erosion is too timid to be
 * doing anything, always would mean the partition is being destroyed rather than chewed.
 */
void testConnectivityIsGuaranteed()
{
    std::printf("every liminal sector is fully connected, at every seed\n");

    lpl::core::u32 totalBroken = 0u;
    lpl::core::u32 sectorsNeedingRepair = 0u;

    for (lpl::core::u32 seed = 0u; seed < 24u; ++seed)
    {
        lpl::procgen::LiminalParams params;
        params.width = 64u;
        params.depth = 64u;
        params.seed = 900u + seed * 131u;

        const lpl::procgen::LiminalSpace space = lpl::procgen::generateLiminal(params);
        check(!space.map.empty(), "a sector is generated");
        check(space.openCells > 0u, "a sector has open space");
        if (!space.connected)
            std::printf("    seed %u came out disconnected\n", params.seed);
        check(space.connected, "the sector is fully connected");

        totalBroken += space.wallsBroken;
        if (space.wallsBroken > 0u)
            ++sectorsNeedingRepair;
    }

    std::printf("    %u of 24 sectors needed repair, %u walls broken in total\n", sectorsNeedingRepair, totalBroken);
    check(sectorsNeedingRepair > 0u, "the erosion is strong enough to actually strand rooms");
}

/**
 * @brief The four zones exist and shape different spaces.
 *
 * Same seed twice is the same space. This is the whole determinism claim.
 */
void testZonesDiffer()
{
    std::printf("the four zones exist and shape different spaces\n");

    lpl::procgen::LiminalParams params;
    params.width = 96u;
    params.depth = 96u;
    params.seed = 5150u;

    const lpl::procgen::LiminalZoneMap zones = lpl::procgen::zoneMap(params);
    lpl::core::u32 counts[static_cast<lpl::core::u32>(lpl::procgen::LiminalZone::Count)] = {};
    for (lpl::core::u32 i = 0u; i < zones.cellCount(); ++i)
        ++counts[static_cast<lpl::core::u32>(zones[i])];

    lpl::core::u32 present = 0u;
    std::printf("    ");
    for (lpl::core::u32 i = 0u; i < static_cast<lpl::core::u32>(lpl::procgen::LiminalZone::Count); ++i)
    {
        std::printf("%s=%u ", lpl::procgen::liminalZoneName(static_cast<lpl::procgen::LiminalZone>(i)), counts[i]);
        if (counts[i] != 0u)
            ++present;
    }
    std::printf("\n");
    check(present >= 2u, "a map holds more than one kind of space");

    const lpl::procgen::LiminalSpace a = lpl::procgen::generateLiminal(params);
    const lpl::procgen::LiminalSpace b = lpl::procgen::generateLiminal(params);
    check(lpl::procgen::foldLiminal(a) == lpl::procgen::foldLiminal(b), "a sector is reproducible from its seed");
}

/**
 * @brief Events sit on the route, secrets sit as far from it as possible.
 *
 * A secret placed one step off the main corridor is not a secret. It has to be genuinely out of the way, which is
 * what the detour distance measures.
 */
void testEventAndSecretPlacement()
{
    std::printf("events sit on the route, secrets sit as far from it as possible\n");

    lpl::procgen::LiminalParams params;
    params.width = 80u;
    params.depth = 80u;
    params.seed = 31337u;
    params.hotPathEvents = 6u;
    params.secretSites = 3u;

    const lpl::procgen::LiminalSpace space = lpl::procgen::generateLiminal(params);
    check(!space.eventSites.empty(), "events were placed");
    check(!space.secretSites.empty(), "secrets were placed");

    lpl::core::u32 startX = 0u;
    lpl::core::u32 startZ = 0u;
    lpl::core::u32 goalX = 0u;
    lpl::core::u32 goalZ = 0u;
    check(lpl::procgen::findFarthestPair(space.map, startX, startZ, goalX, goalZ), "the sector has two far ends");
    const lpl::procgen::HotPathAnalysis path =
        lpl::procgen::analyseHotPath(space.map, startX, startZ, goalX, goalZ, 0u);
    check(path.valid, "the critical path is traceable");

    bool eventsOnPath = true;
    for (lpl::core::u32 i = 0u; i < space.eventSites.size(); ++i)
        if (path.onPath[space.eventSites[i]] == 0u)
            eventsOnPath = false;
    check(eventsOnPath, "every event site lies on the critical path");

    lpl::core::u32 shallowestSecret = 0xFFFFFFFFu;
    for (lpl::core::u32 i = 0u; i < space.secretSites.size(); ++i)
    {
        const lpl::core::u32 detour = path.detour[space.secretSites[i]];
        if (detour != lpl::procgen::kUnreachable && detour < shallowestSecret)
            shallowestSecret = detour;
    }
    std::printf("    deepest detour %u, shallowest secret at %u\n", path.deepestDetour, shallowestSecret);

    check(shallowestSecret > 1u, "secrets are genuinely off the route");
}

/**
 * @brief An unbounded liminal space is generated by sector.
 *
 * A sector must depend only on where it is, so generating a 3x3 block in two different orders has to give the same
 * nine sectors. And every sector, generated in isolation, still carries the connectivity guarantee.
 */
void testChunksAreIndependentAndReproducible()
{
    std::printf("an unbounded liminal space is generated by sector\n");

    lpl::procgen::LiminalParams params;
    params.width = 48u;
    params.depth = 48u;
    params.seed = 77u;

    lpl::core::u32 forward[9] = {};
    lpl::core::u32 backward[9] = {};
    lpl::core::u32 index = 0u;
    for (lpl::core::i32 z = -1; z <= 1; ++z)
        for (lpl::core::i32 x = -1; x <= 1; ++x)
            forward[index++] =
                lpl::procgen::foldLiminal(lpl::procgen::generateLiminalChunk(params, lpl::procgen::ChunkCoord{x, z}));

    index = 9u;
    for (lpl::core::i32 z = 1; z >= -1; --z)
        for (lpl::core::i32 x = 1; x >= -1; --x)
            backward[--index] =
                lpl::procgen::foldLiminal(lpl::procgen::generateLiminalChunk(params, lpl::procgen::ChunkCoord{x, z}));

    bool identical = true;
    lpl::core::u32 distinct = 0u;
    for (lpl::core::u32 i = 0u; i < 9u; ++i)
    {
        if (forward[i] != backward[i])
            identical = false;
        bool seen = false;
        for (lpl::core::u32 j = 0u; j < i; ++j)
            if (forward[j] == forward[i])
                seen = true;
        if (!seen)
            ++distinct;
    }

    std::printf("    %u distinct sectors out of 9\n", distinct);
    check(identical, "a sector does not depend on the order sectors were generated in");
    check(distinct == 9u, "neighbouring sectors are different spaces");

    bool allConnected = true;
    for (lpl::core::i32 z = -1; z <= 1; ++z)
        for (lpl::core::i32 x = -1; x <= 1; ++x)
            if (!lpl::procgen::generateLiminalChunk(params, lpl::procgen::ChunkCoord{x, z}).connected)
                allConnected = false;
    check(allConnected, "every sector of the block is connected");
}

void testMaskFirstPreset()
{
    std::printf("the eroded shape becomes a hard constraint on the dressing\n");

    lpl::procgen::Grid<lpl::core::u8> mask{8u, 8u, lpl::core::u8{0}};
    for (lpl::core::u32 i = 0u; i < mask.cellCount(); i += 3u)
        mask[i] = 1u;

    const lpl::procgen::TileGrid preset = lpl::procgen::presetFromMask(mask, 2u);
    lpl::core::u32 pinned = 0u;
    lpl::core::u32 free = 0u;
    for (lpl::core::u32 i = 0u; i < preset.cellCount(); ++i)
    {
        if (preset[i] == lpl::procgen::kNoTile)
            ++free;
        else if (preset[i] == 2u && mask[i] != 0u)
            ++pinned;
    }
    check(pinned + free == preset.cellCount(), "every cell is either pinned to the mask or left free");
    check(pinned > 0u && free > 0u, "the preset constrains some cells and not others");
}

} // namespace

/**
 * @brief Probes for the layered liminal pipeline.
 *
 * The claim that matters is the anti-softlock guarantee: after the erosion
 * stages have chewed the partitions, EVERY open cell must still be reachable
 * from every other. A generator that only reports "not connected" has moved the
 * problem to its caller; this one repairs, so the test is that it always does.
 */
int main()
{
    std::printf("== procgen liminal ==\n");
    testConnectivityIsGuaranteed();
    testZonesDiffer();
    testEventAndSecretPlacement();
    testChunksAreIndependentAndReproducible();
    testMaskFirstPreset();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
