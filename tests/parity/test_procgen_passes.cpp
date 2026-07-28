/**
 * @file test_procgen_passes.cpp
 * @brief Every procedural pass: is it deterministic, and does it do its job?
 *
 * Two questions per pass, because either alone is worthless. Determinism
 * without correctness gives a world that is reliably wrong; correctness without
 * determinism gives a world the client and the server disagree about.
 *
 * So each pass gets a reproducibility check (same seed, same fold), a
 * sensitivity check (different seed, different fold), and at least one
 * INVARIANT — a property that would have to hold for the pass to have done
 * anything meaningful. Erosion must reduce steepness. Rivers must run downhill.
 * WFC must not violate the rules it was given. A cave must be walkable end to
 * end. Those are what catch an implementation that folds beautifully and
 * produces nonsense.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/ecs/Registry.hpp>
#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/procgen/Erosion.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/Hydrology.hpp>
#include <lpl/procgen/Random.hpp>
#include <lpl/procgen/WaveFunctionCollapse.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool condition, const char *label)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++g_failures;
}

/// Average steepness over a field — the number erosion should reduce.
double averageSlope(const lpl::procgen::Heightfield &field)
{
    if (field.empty())
        return 0.0;
    double total = 0.0;
    for (lpl::core::u32 z = 0u; z < field.depth(); ++z)
        for (lpl::core::u32 x = 0u; x < field.width(); ++x)
            total += static_cast<double>(lpl::procgen::slopeAt(field, x, z).toFloat());
    return total / static_cast<double>(field.cellCount());
}

lpl::procgen::Heightfield makeTestTerrain(lpl::core::u32 seed)
{
    lpl::procgen::NoiseParams noise;
    noise.seed = seed;
    noise.frequency = 0.08f;
    noise.amplitude = 20.0f;
    noise.octaves = 5u;
    return lpl::procgen::generateNoiseHeightfield(48u, 48u, noise);
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== procedural passes: determinism and invariants ==\n\n");

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("-- random stream --\n");
    {
        procgen::Random a{1337u};
        procgen::Random b{1337u};
        bool identical = true;
        for (core::u32 i = 0u; i < 1000u; ++i)
            identical = identical && a.next() == b.next();
        check(identical, "the same seed replays the same stream");

        procgen::Random c{1338u};
        procgen::Random d{1337u};
        check(c.next() != d.next(), "a different seed diverges");

        // A stream that leans on its low bits would show here.
        procgen::Random spread{99u};
        core::u32 buckets[4] = {0u, 0u, 0u, 0u};
        for (core::u32 i = 0u; i < 4000u; ++i)
            ++buckets[spread.below(4u)];
        bool balanced = true;
        for (core::u32 i = 0u; i < 4u; ++i)
            balanced = balanced && buckets[i] > 700u && buckets[i] < 1300u;
        check(balanced, "below() is roughly uniform");

        check(procgen::deriveStream(1337u, 1u).state() != procgen::deriveStream(1337u, 2u).state(),
              "different salts give independent streams");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- heightfield --\n");
    {
        const procgen::Heightfield a = makeTestTerrain(1337u);
        const procgen::Heightfield b = makeTestTerrain(1337u);
        const procgen::Heightfield c = makeTestTerrain(2024u);
        check(procgen::foldHeightfield(a) == procgen::foldHeightfield(b), "same seed reproduces the terrain");
        check(procgen::foldHeightfield(a) != procgen::foldHeightfield(c), "a different seed changes the terrain");

        math::Fixed32 low{}, high{};
        check(procgen::heightRange(a, low, high) && low < high, "the terrain has relief");

        procgen::Heightfield normalized = a;
        procgen::normalizeHeights(normalized, math::Fixed32::zero(), math::Fixed32::fromInt(10));
        math::Fixed32 nlow{}, nhigh{};
        (void) procgen::heightRange(normalized, nlow, nhigh);
        check(nlow == math::Fixed32::zero() && nhigh == math::Fixed32::fromInt(10),
              "normalisation hits the requested range exactly");

        procgen::Heightfield smoothed = a;
        procgen::smoothHeights(smoothed, 3u);
        check(averageSlope(smoothed) < averageSlope(a), "smoothing reduces steepness");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- erosion --\n");
    {
        const procgen::Heightfield original = makeTestTerrain(1337u);

        procgen::Heightfield thermal = original;
        procgen::ThermalErosionParams tp;
        tp.iterations = 12u;
        tp.talus = 0.5f;
        const math::Fixed32 moved = procgen::thermalErode(thermal, tp);
        check(moved > math::Fixed32::zero(), "thermal erosion moves material");
        check(averageSlope(thermal) < averageSlope(original), "thermal erosion reduces steepness");

        procgen::Heightfield thermalTwin = original;
        procgen::thermalErode(thermalTwin, tp);
        check(procgen::foldHeightfield(thermal) == procgen::foldHeightfield(thermalTwin),
              "thermal erosion is reproducible");

        // The scan-order artefact this pass is built to avoid: eroding in place
        // would make the result depend on direction, so a mirrored input would
        // not give a mirrored output. Cheap proxy: the fold must not depend on
        // how many iterations are split across calls.
        procgen::Heightfield split = original;
        procgen::ThermalErosionParams half = tp;
        half.iterations = 6u;
        procgen::thermalErode(split, half);
        procgen::thermalErode(split, half);
        check(procgen::foldHeightfield(split) == procgen::foldHeightfield(thermal),
              "12 iterations equal 6 then 6 (no per-call state)");

        procgen::Heightfield hydraulic = original;
        procgen::HydraulicErosionParams hp;
        hp.iterations = 10u;
        const math::Fixed32 displaced = procgen::hydraulicErode(hydraulic, hp);
        check(displaced > math::Fixed32::zero(), "hydraulic erosion displaces material");

        procgen::Heightfield hydraulicTwin = original;
        procgen::hydraulicErode(hydraulicTwin, hp);
        check(procgen::foldHeightfield(hydraulic) == procgen::foldHeightfield(hydraulicTwin),
              "hydraulic erosion is reproducible");
        check(procgen::foldHeightfield(hydraulic) != procgen::foldHeightfield(original),
              "hydraulic erosion actually changed the terrain");

        // THE invariant for this pass, and the one that was missing. A slope-blind
        // implementation folds perfectly and reproduces perfectly while doing the
        // wrong thing: measured on this very terrain it moved marginally MORE
        // material on flat cells than on steep ones, so it lowered the ground
        // uniformly and carved nothing. Transport capacity goes as the slope, so
        // steep ground has to change more than flat ground.
        double steepChange = 0.0;
        double flatChange = 0.0;
        core::u32 steepCells = 0u;
        core::u32 flatCells = 0u;
        const math::Fixed32 slopeCut = math::Fixed32::fromFloat(0.6f);
        for (core::u32 z = 0u; z < original.depth(); ++z)
            for (core::u32 x = 0u; x < original.width(); ++x)
            {
                const double delta = static_cast<double>((hydraulic.at(x, z) - original.at(x, z)).abs().toFloat());
                if (procgen::slopeAt(original, x, z) > slopeCut)
                {
                    ++steepCells;
                    steepChange += delta;
                }
                else
                {
                    ++flatCells;
                    flatChange += delta;
                }
            }
        const double steepMean = steepCells ? steepChange / steepCells : 0.0;
        const double flatMean = flatCells ? flatChange / flatCells : 0.0;
        check(steepMean > flatMean * 1.15,
              "hydraulic erosion attacks steep ground harder than flat (capacity follows slope)");
        std::printf("    steep mean |delta|=%.4f over %u cells, flat mean=%.4f over %u cells\n", steepMean, steepCells,
                    flatMean, flatCells);
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- hydrology --\n");
    {
        const procgen::Heightfield terrain = makeTestTerrain(1337u);
        const procgen::DrainageNetwork network = procgen::computeDrainage(terrain);

        check(network.maxAccumulation > 1u, "drainage accumulates flow");
        check(network.maxAccumulation <= terrain.cellCount(), "no cell drains more than the whole map (conservation)");

        // The invariant that makes it hydrology and not decoration. Judged on the
        // FILLED surface, which is the one routing was computed over: a cell inside
        // a filled basin drains to a neighbour lower than itself on that surface,
        // and need not be lower on the raw terrain.
        bool alwaysDownhill = true;
        for (core::u32 z = 0u; z < terrain.depth() && alwaysDownhill; ++z)
        {
            for (core::u32 x = 0u; x < terrain.width(); ++x)
            {
                const core::u8 direction = network.direction.at(x, z);
                if (direction == procgen::kNoFlow)
                    continue;
                const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor8X[direction];
                const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor8Z[direction];
                if (network.filled.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) >=
                    network.filled.at(x, z))
                {
                    alwaysDownhill = false;
                    break;
                }
            }
        }
        check(alwaysDownhill, "every cell drains strictly downhill");

        // Depression filling is what makes the whole layer mean anything: without
        // it a noise terrain routes into several hundred separate basins and the
        // largest carries a couple of percent of the map whatever its size.
        core::u32 interiorSinks = 0u;
        for (core::u32 z = 1u; z + 1u < terrain.depth(); ++z)
            for (core::u32 x = 1u; x + 1u < terrain.width(); ++x)
                if (network.direction.at(x, z) == procgen::kNoFlow)
                    ++interiorSinks;
        check(interiorSinks == 0u, "depression filling leaves no interior sink");
        check(network.maxAccumulation > terrain.cellCount() / 8u,
              "the trunk drains a real share of the map, not one basin");

        // The filled surface must never sit below the terrain: filling raises
        // ground into lakes, it does not excavate.
        bool fillOnlyRaises = true;
        for (core::u32 i = 0u; i < terrain.cellCount(); ++i)
            fillOnlyRaises = fillOnlyRaises && network.filled[i] >= terrain[i];
        check(fillOnlyRaises, "filling only ever raises ground");

        const procgen::DrainageNetwork twin = procgen::computeDrainage(terrain);
        check(twin.maxAccumulation == network.maxAccumulation, "drainage is reproducible");
        check(procgen::foldHeightfield(twin.filled) == procgen::foldHeightfield(network.filled),
              "the filled surface is reproducible");

        procgen::Heightfield carved = terrain;
        const core::u32 riverCells = procgen::carveRivers(carved, network, procgen::RiverParams{});
        check(riverCells > 0u, "rivers are carved");
        check(procgen::foldHeightfield(carved) != procgen::foldHeightfield(terrain), "carving lowers the bed");

        // ── The river network keeps its density at every map size ────────────
        //
        // The third time this module has been bitten by a threshold judged
        // against a distribution that moves: a share of the LARGEST flow made 34%
        // of a 24x24 world river and 3.7% of a 128x128 one, from the same recipe.
        // A quantile of the accumulation distribution is scale-free, and that is
        // what this pins — a drainage network, not a water table.
        {
            const procgen::RiverParams asked; // default density
            bool densityHolds = true;
            core::u32 shares[4] = {0u, 0u, 0u, 0u};
            const core::u32 sizes[4] = {24u, 48u, 64u, 128u};
            for (core::u32 s = 0u; s < 4u; ++s)
            {
                procgen::NoiseParams noise;
                noise.seed = 1337u;
                noise.frequency = 0.15f;
                noise.octaves = 4u;
                procgen::Heightfield field = procgen::generateNoiseHeightfield(sizes[s], sizes[s], noise);
                procgen::normalizeHeights(field, math::Fixed32::fromFloat(-8.0f), math::Fixed32::fromFloat(16.0f));
                const procgen::DrainageNetwork drainage = procgen::computeDrainage(field);
                const core::u32 marked = procgen::carveRivers(field, drainage, asked);
                shares[s] = (marked * 1000u) / (sizes[s] * sizes[s]); // per mille
                // Asked for 4%: accept 1.5% to 7%. A band, not a point, because
                // ties are kept whole and a tiny map has coarse granularity.
                densityHolds = densityHolds && shares[s] >= 15u && shares[s] <= 70u;
            }
            check(densityHolds, "river density is scale-free: the same recipe wets the same share at any map size");
            std::printf("    river share per mille: 24^2=%u 48^2=%u 64^2=%u 128^2=%u\n", shares[0], shares[1],
                        shares[2], shares[3]);
        }

        const procgen::Heightfield moisture = procgen::computeMoisture(terrain, network, procgen::MoistureParams{});
        bool inRange = true;
        for (core::u32 i = 0u; i < moisture.cellCount(); ++i)
            inRange = inRange && moisture[i] >= math::Fixed32::zero() && moisture[i] <= math::Fixed32::one();
        check(inRange, "moisture stays within [0, 1]");

        // The whole point of normalising against the land: the thresholds a biome
        // classifier applies have to be reachable. A ceiling of 0.6 on land made
        // forest, rainforest and marsh impossible however they were configured.
        math::Fixed32 wettest = math::Fixed32::zero();
        math::Fixed32 driest = math::Fixed32::one();
        for (core::u32 i = 0u; i < moisture.cellCount(); ++i)
        {
            if (moisture[i] > wettest)
                wettest = moisture[i];
            if (moisture[i] < driest)
                driest = moisture[i];
        }
        check(wettest > math::Fixed32::fromFloat(0.9f) && driest < math::Fixed32::fromFloat(0.1f),
              "moisture spans its full range, so thresholds are reachable");

        // Rain shadow: a windward ridge has to dry the land behind it. Compared
        // against the same world with the shadow switched off, so only the shadow
        // term differs.
        procgen::MoistureParams shadowed;
        shadowed.rainShadow = 0.8f;
        procgen::MoistureParams unshadowed = shadowed;
        unshadowed.rainShadow = 0.0f;
        check(procgen::foldHeightfield(procgen::computeMoisture(terrain, network, shadowed)) !=
                  procgen::foldHeightfield(procgen::computeMoisture(terrain, network, unshadowed)),
              "the rain shadow changes the climate");

        // Wind direction has to matter, or the shadow is not orographic.
        procgen::MoistureParams eastward = shadowed;
        eastward.windDirection = 0u;
        procgen::MoistureParams westward = shadowed;
        westward.windDirection = 1u;
        check(procgen::foldHeightfield(procgen::computeMoisture(terrain, network, eastward)) !=
                  procgen::foldHeightfield(procgen::computeMoisture(terrain, network, westward)),
              "reversing the wind moves the dry side");

        // Climate is the lowest-frequency thing in a world, so its scale is
        // expressed relative to the map: doubling the map must not double the
        // number of wet and dry belts a player crosses.
        const procgen::Heightfield small = procgen::generateNoiseHeightfield(48u, 48u, procgen::NoiseParams{});
        const procgen::Heightfield large = procgen::generateNoiseHeightfield(96u, 96u, procgen::NoiseParams{});
        const auto beltCrossings = [](const procgen::Heightfield &field) {
            const procgen::DrainageNetwork net = procgen::computeDrainage(field);
            const procgen::Heightfield wet = procgen::computeMoisture(field, net, procgen::MoistureParams{});
            core::u32 crossings = 0u;
            const core::u32 row = field.depth() / 2u;
            bool above = wet.at(0u, row) > math::Fixed32::half();
            for (core::u32 x = 1u; x < field.width(); ++x)
            {
                const bool now = wet.at(x, row) > math::Fixed32::half();
                if (now != above)
                    ++crossings;
                above = now;
            }
            return crossings;
        };
        const core::u32 smallBelts = beltCrossings(small);
        const core::u32 largeBelts = beltCrossings(large);
        check(largeBelts <= smallBelts + 3u, "climate belts do not multiply with map size");
        std::printf("    belts: 48x48 -> %u crossings, 96x96 -> %u crossings\n", smallBelts, largeBelts);
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- biomes --\n");
    {
        const procgen::BiomeParams params;

        // The thresholds are absolute world units, so the terrain has to be put in
        // the range they describe. Reading the range off the params rather than
        // hardcoding one is the point: a test that assumed sea level was zero would
        // pass or fail on the default rather than on the classifier.
        procgen::Heightfield terrain = makeTestTerrain(1337u);
        procgen::normalizeHeights(terrain, math::Fixed32::fromFloat(params.seaLevel - 4.0f),
                                  math::Fixed32::fromFloat(params.snowHeight + 2.0f));

        procgen::MoistureParams climate;
        climate.seaLevel = params.seaLevel;
        const procgen::DrainageNetwork network = procgen::computeDrainage(terrain);
        const procgen::Heightfield moisture = procgen::computeMoisture(terrain, network, climate);

        procgen::ClimateParams axes;
        axes.seaLevel = params.seaLevel;
        const procgen::ClimateField climateField = procgen::computeClimate(terrain, moisture, network, axes);

        const procgen::BiomeMap map = procgen::classifyBiomes(terrain, climateField, params);
        check(!map.empty(), "biomes are classified");

        core::u32 counts[static_cast<core::u32>(procgen::BiomeId::Count)] = {};
        procgen::countBiomes(map, counts);

        core::u32 distinct = 0u;
        for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
            if (counts[i] != 0u)
                ++distinct;
        check(distinct >= 3u, "a varied terrain yields several biomes");

        // Altitude must dominate: nothing below sea level may be dry land.
        bool oceanConsistent = true;
        for (core::u32 i = 0u; i < map.cellCount(); ++i)
            if (terrain[i] <= math::Fixed32::fromFloat(params.seaLevel) && map[i] != procgen::BiomeId::Ocean)
                oceanConsistent = false;
        check(oceanConsistent, "everything at or below sea level is ocean");

        // The lapse rate is what bands a mountain. Without it a summit is classified
        // like the valley it rises from, so the same world with the rate at zero must
        // come out differently — and warmer up high.
        procgen::ClimateParams flatAxes = axes;
        flatAxes.lapseRate = 0.0f;
        const procgen::ClimateField flatField = procgen::computeClimate(terrain, moisture, network, flatAxes);
        check(procgen::foldBiomeMap(procgen::classifyBiomes(terrain, flatField, params)) != procgen::foldBiomeMap(map),
              "the lapse rate changes the classification");

        const procgen::BiomeMap twin = procgen::classifyBiomes(terrain, climateField, params);
        check(procgen::foldBiomeMap(map) == procgen::foldBiomeMap(twin), "classification is reproducible");

        // A world this varied should reach most of the palette. Six of the twelve
        // were unreachable at any setting before the climate model was fixed.
        check(distinct >= 9u, "a full world reaches most of the biome palette");

        std::printf("    ");
        for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
            if (counts[i] != 0u)
                std::printf("%s=%u ", procgen::biomeName(static_cast<procgen::BiomeId>(i)), counts[i]);
        std::printf("\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- wave function collapse --\n");
    {
        const procgen::TileSet tiles = procgen::makeTerrainTileSet();
        check(tiles.valid(), "the terrain tile set is well formed");

        procgen::WfcParams params;
        params.width = 24u;
        params.depth = 24u;
        params.seed = 1337u;

        const procgen::WfcResult a = procgen::solveWfc(tiles, params);
        check(a.solved, "WFC solves the grid");
        check(procgen::countAdjacencyViolations(a.tiles, tiles) == 0u, "the solution violates no adjacency rule");

        const procgen::WfcResult b = procgen::solveWfc(tiles, params);
        bool identical = a.tiles.cellCount() == b.tiles.cellCount();
        for (core::u32 i = 0u; identical && i < a.tiles.cellCount(); ++i)
            identical = a.tiles[i] == b.tiles[i];
        check(identical, "the same seed reproduces the arrangement");

        procgen::WfcParams other = params;
        other.seed = 2024u;
        const procgen::WfcResult c = procgen::solveWfc(tiles, other);
        bool differs = false;
        for (core::u32 i = 0u; i < a.tiles.cellCount() && i < c.tiles.cellCount(); ++i)
            differs = differs || a.tiles[i] != c.tiles[i];
        check(differs, "a different seed gives a different arrangement");

        // Border constraints: the seam mechanism for chunked worlds.
        procgen::TileGrid preset{params.width, params.depth, procgen::kNoTile};
        for (core::u32 z = 0u; z < params.depth; ++z)
            preset.at(0u, z) = static_cast<core::u8>(procgen::TerrainTile::Water);
        const procgen::WfcResult constrained = procgen::solveWfc(tiles, params, &preset);
        check(constrained.solved, "WFC solves around pinned cells");
        bool borderHeld = true;
        for (core::u32 z = 0u; z < params.depth; ++z)
            borderHeld =
                borderHeld && constrained.tiles.at(0u, z) == static_cast<core::u8>(procgen::TerrainTile::Water);
        check(borderHeld, "pinned cells survive the solve");
        check(procgen::countAdjacencyViolations(constrained.tiles, tiles) == 0u,
              "the constrained solution is still legal");

        std::printf("    attempts=%u contradictions=%u localRepairs=%u\n", a.attempts, a.contradictions,
                    a.localRepairs);
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- dungeons --\n");
    {
        procgen::BspDungeonParams bsp;
        bsp.width = 64u;
        bsp.depth = 64u;
        bsp.seed = 1337u;
        lpl::pmr::vector<procgen::Room> rooms;
        procgen::DungeonMap dungeon = procgen::generateBspDungeon(bsp, &rooms);
        check(rooms.size() >= 4u, "BSP carves several rooms");
        check(procgen::isFullyConnected(dungeon), "BSP guarantees connectivity by construction");

        procgen::DungeonMap twin = procgen::generateBspDungeon(bsp);
        check(procgen::foldDungeon(dungeon) == procgen::foldDungeon(twin), "BSP is reproducible");

        procgen::CaveParams cave;
        cave.width = 64u;
        cave.depth = 64u;
        cave.seed = 1337u;
        procgen::DungeonMap caves = procgen::generateCellularCave(cave);
        check(procgen::isFullyConnected(caves), "caves are connected after the repair pass");

        core::u32 floor = 0u;
        for (core::u32 i = 0u; i < caves.cellCount(); ++i)
            if (caves[i] != procgen::DungeonCell::Wall)
                ++floor;
        check(floor > caves.cellCount() / 10u, "caves leave a usable amount of open space");

        procgen::DungeonMap cavesTwin = procgen::generateCellularCave(cave);
        check(procgen::foldDungeon(caves) == procgen::foldDungeon(cavesTwin), "caves are reproducible");

        procgen::DrunkardParams walk;
        walk.width = 48u;
        walk.depth = 48u;
        walk.seed = 1337u;
        procgen::DungeonMap galleries = procgen::generateDrunkardWalk(walk);
        check(procgen::isFullyConnected(galleries), "the drunkard's galleries are connected");
        check(procgen::foldDungeon(galleries) == procgen::foldDungeon(procgen::generateDrunkardWalk(walk)),
              "the walk is reproducible");

        // Erosion must not break the guarantee the generators just established.
        procgen::DungeonMap eroded = dungeon;
        procgen::erodeEdges(eroded, 1337u, 0.25f);
        check(procgen::foldDungeon(eroded) != procgen::foldDungeon(dungeon), "edge erosion changes the map");
        check(procgen::isFullyConnected(eroded), "edge erosion preserves connectivity");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- world builder --\n");
    {
        ecs::Registry registry;
        const procgen::BuiltWorldStats stats = procgen::WorldBuilder{1337u}
                                                   .terrain(48u, 48u)
                                                   .normalize(-4.0f, 28.0f)
                                                   .erode()
                                                   .rivers()
                                                   .biomes()
                                                   .scatterInBiome(procgen::BiomeId::Forest, 0.15f)
                                                   .scatterInBiome(procgen::BiomeId::Grassland, 0.05f)
                                                   .materialize(registry);

        check(stats.terrainCells == 48u * 48u, "the builder built the requested terrain");
        check(stats.terrainEntities == stats.terrainCells, "every ground cell became an entity");
        check(stats.propEntities > 0u, "scatter rules placed props");
        check(stats.riverCells > 0u, "the builder carved rivers");

        ecs::Registry twinRegistry;
        const procgen::BuiltWorldStats twin = procgen::WorldBuilder{1337u}
                                                  .terrain(48u, 48u)
                                                  .normalize(-4.0f, 28.0f)
                                                  .erode()
                                                  .rivers()
                                                  .biomes()
                                                  .scatterInBiome(procgen::BiomeId::Forest, 0.15f)
                                                  .scatterInBiome(procgen::BiomeId::Grassland, 0.05f)
                                                  .materialize(twinRegistry);
        check(twin.heightSignature == stats.heightSignature && twin.biomeSignature == stats.biomeSignature &&
                  twin.propEntities == stats.propEntities,
              "the whole pipeline is reproducible end to end");

        ecs::Registry otherRegistry;
        const procgen::BuiltWorldStats other = procgen::WorldBuilder{2024u}
                                                   .terrain(48u, 48u)
                                                   .normalize(-4.0f, 28.0f)
                                                   .erode()
                                                   .rivers()
                                                   .biomes()
                                                   .materialize(otherRegistry);
        check(other.heightSignature != stats.heightSignature, "a different seed builds a different world");

        // The short form must not be a different engine from the long one.
        procgen::WorldBuilder lazy{4242u};
        const procgen::BuiltWorldStats implied = lazy.biomes().bakeGrids();
        check(implied.terrainCells > 0u, "asking for biomes alone still produces terrain first");
        check(!lazy.biomeMap().empty(), "dependencies run in the right order automatically");

        std::printf("    ground=%u props=%u rivers=%u height_sig=0x%08X biome_sig=0x%08X\n", stats.terrainEntities,
                    stats.propEntities, stats.riverCells, stats.heightSignature, stats.biomeSignature);
    }

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
