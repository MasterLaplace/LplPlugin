#include <lpl/procgen/CaveSystem.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

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

lpl::procgen::Heightfield makeSurface(lpl::core::u32 size, lpl::core::u32 seed)
{
    lpl::procgen::NoiseParams noise;
    noise.seed = seed;
    noise.frequency = 4.0f / static_cast<lpl::core::f32>(size);
    noise.amplitude = 12.0f;
    noise.octaves = 4u;
    lpl::procgen::Heightfield field = lpl::procgen::generateNoiseHeightfield(size, size, noise);
    lpl::procgen::normalizeHeights(field, lpl::math::Fixed32::fromFloat(-6.0f), lpl::math::Fixed32::fromFloat(14.0f));
    return field;
}

void testEntrancesAndReachability()
{
    std::printf("a cave has a way in, and everything inside is reachable through it\n");

    lpl::core::u32 sealedSystems = 0u;
    lpl::core::u32 unreachableSystems = 0u;

    for (lpl::core::u32 seed = 0u; seed < 16u; ++seed)
    {
        lpl::procgen::CaveSystemParams params;
        params.width = 64u;
        params.depth = 64u;
        params.seed = 4000u + seed * 977u;
        params.layers = 3u;
        params.entrances = 2u;

        const lpl::procgen::Heightfield surface = makeSurface(64u, params.seed);
        const lpl::procgen::CaveSystem system = lpl::procgen::generateCaveSystem(params, surface, nullptr);

        check(system.layerCount == 3u, "the requested layers were generated");
        if (system.entranceCount == 0u)
            ++sealedSystems;
        if (!system.fullyReachable())
        {
            ++unreachableSystems;
            std::printf("    seed %u: %u of %u hollow cells reachable\n", params.seed, system.reachableCells,
                        system.hollowCells);
        }
    }

    std::printf("    %u of 16 systems sealed, %u with unreachable volume\n", sealedSystems, unreachableSystems);
    check(sealedSystems == 0u, "every system has at least one entrance");
    check(unreachableSystems == 0u, "every hollow cell is reachable from an entrance");
}

/**
 * @brief A shaft only exists where both layers are hollow.
 *
 * The invariant the flat version could not express: a shaft joins two hollow cells. One end in rock is a shaft into
 * nowhere.
 */
void testShaftsAreValid()
{
    std::printf("a shaft only exists where both layers are hollow\n");

    lpl::procgen::CaveSystemParams params;
    params.width = 64u;
    params.depth = 64u;
    params.seed = 8080u;
    params.layers = 4u;

    const lpl::procgen::Heightfield surface = makeSurface(64u, params.seed);
    const lpl::procgen::CaveSystem system = lpl::procgen::generateCaveSystem(params, surface, nullptr);

    check(!system.shafts.empty(), "shafts were cut");

    lpl::core::u32 intoRock = 0u;
    lpl::core::u32 surfaceShafts = 0u;
    for (lpl::core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const lpl::procgen::CaveShaft &shaft = system.shafts[i];
        if (shaft.surface)
        {
            ++surfaceShafts;
            if (!lpl::procgen::isWalkable(system.layer[shaft.upperLayer].at(shaft.x, shaft.z)))
                ++intoRock;
            continue;
        }
        if (!lpl::procgen::isWalkable(system.layer[shaft.upperLayer].at(shaft.x, shaft.z)) ||
            !lpl::procgen::isWalkable(system.layer[shaft.lowerLayer].at(shaft.x, shaft.z)))
            ++intoRock;
    }

    std::printf("    %u shafts, %u of them entrances, %u ending in rock\n",
                static_cast<lpl::core::u32>(system.shafts.size()), surfaceShafts, intoRock);
    check(intoRock == 0u, "no shaft ends in solid rock");
    check(surfaceShafts > 0u, "some shafts reach the surface");
}

/**
 * @brief The system opens out as it descends.
 *
 * A higher fill probability means more rock, so the shallow layer must be the tighter one. If this ever inverts, the
 * depth ramp is backwards.
 *
 * The layers are distinct, not one plan repeated: a stack of identical floors is a multi-storey car park.
 */
void testLayersDifferAndDeepen()
{
    std::printf("the system opens out as it descends\n");

    lpl::procgen::CaveSystemParams params;
    params.width = 80u;
    params.depth = 80u;
    params.seed = 606u;
    params.layers = 4u;
    params.topFill = 0.50f;
    params.deepFill = 0.34f;

    const lpl::procgen::Heightfield surface = makeSurface(80u, params.seed);
    const lpl::procgen::CaveSystem system = lpl::procgen::generateCaveSystem(params, surface, nullptr);

    lpl::core::u32 hollow[lpl::procgen::kMaxCaveLayers] = {};
    for (lpl::core::u32 l = 0u; l < system.layerCount; ++l)
    {
        for (lpl::core::u32 i = 0u; i < system.layer[l].cellCount(); ++i)
            if (lpl::procgen::isWalkable(system.layer[l][i]))
                ++hollow[l];
        std::printf("    layer %u: %u hollow cells\n", l, hollow[l]);
    }

    check(hollow[system.layerCount - 1u] > hollow[0], "the deepest layer is more open than the shallowest");

    bool distinct = true;
    for (lpl::core::u32 l = 1u; l < system.layerCount; ++l)
        if (lpl::procgen::foldDungeon(system.layer[l]) == lpl::procgen::foldDungeon(system.layer[0]))
            distinct = false;
    check(distinct, "every layer is a different plan");
}

void testDeterminism()
{
    std::printf("a system is reproducible from its seed\n");

    lpl::procgen::CaveSystemParams params;
    params.width = 48u;
    params.depth = 48u;
    params.seed = 12345u;
    const lpl::procgen::Heightfield surface = makeSurface(48u, params.seed);

    const lpl::procgen::CaveSystem a = lpl::procgen::generateCaveSystem(params, surface, nullptr);
    const lpl::procgen::CaveSystem b = lpl::procgen::generateCaveSystem(params, surface, nullptr);
    check(lpl::procgen::foldCaveSystem(a) == lpl::procgen::foldCaveSystem(b), "two runs give the same system");

    params.seed = 12346u;
    const lpl::procgen::CaveSystem c = lpl::procgen::generateCaveSystem(params, surface, nullptr);
    check(lpl::procgen::foldCaveSystem(a) != lpl::procgen::foldCaveSystem(c),
          "a different seed gives a different system");
}

/**
 * @brief The builder digs a cave system that agrees with its world.
 *
 * An entrance must not open under the sea or on a glacier: the biome map is consulted precisely so a cave mouth lands
 * somewhere a body could stand.
 */
void testBuilderIntegration()
{
    std::printf("the builder digs a cave system that agrees with its world\n");

    lpl::procgen::WorldBuilder builder{2718u};
    lpl::procgen::CaveSystemParams params;
    params.layers = 3u;
    params.entrances = 3u;

    builder.terrain(72u, 72u).normalize(-6.0f, 14.0f).rivers().biomes().caveSystem(params);
    const lpl::procgen::BuiltWorldStats stats = builder.bakeGrids();

    std::printf("    %u layers, %u entrances, %u/%u hollow reachable, %u underground voxels\n", stats.caveLayers,
                stats.caveEntrances, stats.caveReachable, stats.caveHollow, stats.undergroundVoxels);
    check(stats.caveLayers == 3u, "the builder dug the requested layers");
    check(stats.caveEntrances > 0u, "the builder's system has an entrance");
    check(stats.caveReachable == stats.caveHollow, "the builder's system is fully reachable");
    check(stats.undergroundVoxels > 0u, "the system has volume");

    const lpl::procgen::BiomeMap &biomes = builder.biomeMap();
    bool allSensible = true;
    for (lpl::core::u32 i = 0u; i < builder.caves().shafts.size(); ++i)
    {
        const lpl::procgen::CaveShaft &shaft = builder.caves().shafts[i];
        if (!shaft.surface)
            continue;
        const lpl::procgen::BiomeId biome = biomes.at(shaft.x, shaft.z);
        if (lpl::procgen::isWater(biome) || biome == lpl::procgen::BiomeId::Snow)
            allSensible = false;
    }
    check(allSensible, "no entrance opens under water or in ice");
}

} // namespace

/**
 * @brief The gate judges the stack, and judges it for the right reason.
 *
 * The measurement it replaces did not exist: `checkPlayability` on a layered recipe
 * was asked of the FLAT map, which this generator leaves empty, so it reported zero
 * open cells and rejected a navigable world. A document had to switch the gate off to
 * use the generator — and a check you switch off to use a feature has stopped meaning
 * anything.
 *
 * Both directions are asserted, because only one of them is evidence. That a healthy
 * system passes proves nothing on its own: an evaluator that returns "fine" for
 * everything passes it too. So a system is BROKEN on purpose — every shaft to the
 * deepest layer filled in — and the gate has to notice. That leaves a system that
 * still LOOKS generous — the same hollow cells, the same corridors — and whose bottom
 * floor no body can ever reach.
 *
 * The last check is the failure that started all this: judging the flat map instead.
 * It is empty for a layered system, so it reports a world with no cells at all — which
 * is why the old verdict was "fails" rather than "passes", and why nothing looked wrong.
 */
void testTheGateJudgesTheStack()
{
    std::printf("the playability gate judges a stack in three dimensions\n");

    lpl::procgen::CaveSystemParams params;
    params.width = 64u;
    params.depth = 64u;
    params.seed = 20260804u;
    params.layers = 3u;
    params.entrances = 2u;

    const lpl::procgen::Heightfield surface = makeSurface(64u, params.seed);
    const lpl::procgen::CaveSystem system = lpl::procgen::generateCaveSystem(params, surface, nullptr);

    const lpl::procgen::LevelQuality quality = lpl::procgen::evaluateCaveSystem(system);
    std::printf("    walkable=%u reachable=%u deepest=%u longest=%u deadEnds=%u junctions=%u\n", quality.walkableCells,
                quality.reachableCells, quality.pathLength, quality.longestDistance, quality.deadEnds,
                quality.junctions);

    check(quality.walkableCells == system.hollowCells, "it counts the same hollow cells the generator did");
    check(quality.reachableCells == system.reachableCells, "and reaches the same ones");
    check(quality.fullyConnected, "a repaired system is fully connected");
    check(quality.goalReachable, "and its deepest layer can be reached from the surface");
    check(quality.pathLength > 0u, "the way down is a walk, not a step");
    check(quality.longestDistance >= quality.pathLength, "the far end is at least as far as the bottom");
    check(lpl::procgen::passesGate(quality, lpl::procgen::GateCriteria{}), "so it passes the default gate");

    lpl::procgen::CaveSystem sealed = system;
    const lpl::core::u32 deepest = sealed.layerCount - 1u;
    for (lpl::core::u32 i = 0u; i < sealed.shafts.size(); ++i)
    {
        const lpl::procgen::CaveShaft &shaft = sealed.shafts[i];
        if (shaft.lowerLayer != deepest || shaft.upperLayer == shaft.lowerLayer)
            continue;
        sealed.layer[deepest].at(shaft.x, shaft.z) = lpl::procgen::DungeonCell::Wall;
    }

    const lpl::procgen::LevelQuality sealedQuality = lpl::procgen::evaluateCaveSystem(sealed);
    std::printf("    sealed: reachable=%u of %u, goal=%s\n", sealedQuality.reachableCells, sealedQuality.walkableCells,
                sealedQuality.goalReachable ? "yes" : "no");
    check(!sealedQuality.goalReachable, "sealing every shaft to the bottom makes the goal unreachable");
    check(!sealedQuality.fullyConnected, "and leaves a whole floor cut off");
    check(!lpl::procgen::passesGate(sealedQuality, lpl::procgen::GateCriteria{}), "and the gate refuses it");

    check(lpl::procgen::evaluateCaveSystem(lpl::procgen::CaveSystem{}).walkableCells == 0u,
          "an empty stack measures as empty rather than as a healthy one");
}

/**
 * @brief Probes for the layered cave system.
 *
 * The claim that separates a cave from a buried dungeon plan: it has a way in,
 * and everything inside can be reached through it. Both halves are measured,
 * because both were absent before — the old underground was one flat layer with
 * no connection to the world above it at all.
 */
int main()
{
    std::printf("== procgen cave systems ==\n");
    testEntrancesAndReachability();
    testShaftsAreValid();
    testLayersDifferAndDeepen();
    testDeterminism();
    testBuilderIntegration();
    testTheGateJudgesTheStack();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
