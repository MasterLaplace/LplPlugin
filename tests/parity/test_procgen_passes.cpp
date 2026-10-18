#include <lpl/ecs/Registry.hpp>
#include <lpl/math/Random.hpp>
#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/procgen/Erosion.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/Hydrology.hpp>
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

/**
 * @brief Average steepness over a field — the number erosion should reduce.
 */
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

/**
 * @brief The random stream: the same seed replays, a different seed diverges, and the draws spread evenly.
 *
 * A stream that leans on its low bits would show in how evenly below() fills its buckets.
 */
void checkTheRandomStreamReplaysAndSpreads()
{
    std::printf("-- random stream --\n");
    lpl::math::Random a{1337u};
    lpl::math::Random b{1337u};
    bool identical = true;
    for (lpl::core::u32 i = 0u; i < 1000u; ++i)
        identical = identical && a.next() == b.next();
    check(identical, "the same seed replays the same stream");

    lpl::math::Random c{1338u};
    lpl::math::Random d{1337u};
    check(c.next() != d.next(), "a different seed diverges");

    lpl::math::Random spread{99u};
    lpl::core::u32 buckets[4] = {0u, 0u, 0u, 0u};
    for (lpl::core::u32 i = 0u; i < 4000u; ++i)
        ++buckets[spread.below(4u)];
    bool balanced = true;
    for (lpl::core::u32 i = 0u; i < 4u; ++i)
        balanced = balanced && buckets[i] > 700u && buckets[i] < 1300u;
    check(balanced, "below() is roughly uniform");

    check(lpl::math::deriveStream(1337u, 1u).state() != lpl::math::deriveStream(1337u, 2u).state(),
          "different salts give independent streams");
}

/**
 * @brief The heightfield: reproducible, sensitive to its seed, and shaped by normalisation and smoothing as asked.
 */
void checkTheHeightfieldIsReproducibleAndShapedAsAsked()
{
    std::printf("\n-- heightfield --\n");
    const lpl::procgen::Heightfield a = makeTestTerrain(1337u);
    const lpl::procgen::Heightfield b = makeTestTerrain(1337u);
    const lpl::procgen::Heightfield c = makeTestTerrain(2024u);
    check(lpl::procgen::foldHeightfield(a) == lpl::procgen::foldHeightfield(b), "same seed reproduces the terrain");
    check(lpl::procgen::foldHeightfield(a) != lpl::procgen::foldHeightfield(c), "a different seed changes the terrain");

    lpl::math::Fixed32 low{}, high{};
    check(lpl::procgen::heightRange(a, low, high) && low < high, "the terrain has relief");

    lpl::procgen::Heightfield normalized = a;
    lpl::procgen::normalizeHeights(normalized, lpl::math::Fixed32::zero(), lpl::math::Fixed32::fromInt(10));
    lpl::math::Fixed32 nlow{}, nhigh{};
    (void) lpl::procgen::heightRange(normalized, nlow, nhigh);
    check(nlow == lpl::math::Fixed32::zero() && nhigh == lpl::math::Fixed32::fromInt(10),
          "normalisation hits the requested range exactly");

    lpl::procgen::Heightfield smoothed = a;
    lpl::procgen::smoothHeights(smoothed, 3u);
    check(averageSlope(smoothed) < averageSlope(a), "smoothing reduces steepness");
}

/**
 * @brief Erosion: reproducible, and it moves material where the slope says it should.
 *
 * The scan-order artefact the thermal pass is built to avoid: eroding in place would make the result depend on
 * direction, so a mirrored input would not give a mirrored output. Cheap proxy: the fold must not depend on how many
 * iterations are split across calls.
 *
 * THE invariant for the hydraulic pass, and the one that was missing. A slope-blind implementation folds perfectly
 * and reproduces perfectly while doing the wrong thing: measured on this very terrain it moved marginally MORE
 * material on flat cells than on steep ones, so it lowered the ground uniformly and carved nothing. Transport capacity
 * goes as the slope, so steep ground has to change more than flat ground.
 */
void checkErosionIsReproducibleAndFollowsTheSlope()
{
    std::printf("\n-- erosion --\n");
    const lpl::procgen::Heightfield original = makeTestTerrain(1337u);

    lpl::procgen::Heightfield thermal = original;
    lpl::procgen::ThermalErosionParams tp;
    tp.iterations = 12u;
    tp.talus = 0.5f;
    const lpl::math::Fixed32 moved = lpl::procgen::thermalErode(thermal, tp);
    check(moved > lpl::math::Fixed32::zero(), "thermal erosion moves material");
    check(averageSlope(thermal) < averageSlope(original), "thermal erosion reduces steepness");

    lpl::procgen::Heightfield thermalTwin = original;
    lpl::procgen::thermalErode(thermalTwin, tp);
    check(lpl::procgen::foldHeightfield(thermal) == lpl::procgen::foldHeightfield(thermalTwin),
          "thermal erosion is reproducible");

    lpl::procgen::Heightfield split = original;
    lpl::procgen::ThermalErosionParams half = tp;
    half.iterations = 6u;
    lpl::procgen::thermalErode(split, half);
    lpl::procgen::thermalErode(split, half);
    check(lpl::procgen::foldHeightfield(split) == lpl::procgen::foldHeightfield(thermal),
          "12 iterations equal 6 then 6 (no per-call state)");

    lpl::procgen::Heightfield hydraulic = original;
    lpl::procgen::HydraulicErosionParams hp;
    hp.iterations = 10u;
    const lpl::math::Fixed32 displaced = lpl::procgen::hydraulicErode(hydraulic, hp);
    check(displaced > lpl::math::Fixed32::zero(), "hydraulic erosion displaces material");

    lpl::procgen::Heightfield hydraulicTwin = original;
    lpl::procgen::hydraulicErode(hydraulicTwin, hp);
    check(lpl::procgen::foldHeightfield(hydraulic) == lpl::procgen::foldHeightfield(hydraulicTwin),
          "hydraulic erosion is reproducible");
    check(lpl::procgen::foldHeightfield(hydraulic) != lpl::procgen::foldHeightfield(original),
          "hydraulic erosion actually changed the terrain");

    double steepChange = 0.0;
    double flatChange = 0.0;
    lpl::core::u32 steepCells = 0u;
    lpl::core::u32 flatCells = 0u;
    const lpl::math::Fixed32 slopeCut = lpl::math::Fixed32::fromFloat(0.6f);
    for (lpl::core::u32 z = 0u; z < original.depth(); ++z)
        for (lpl::core::u32 x = 0u; x < original.width(); ++x)
        {
            const double delta = static_cast<double>((hydraulic.at(x, z) - original.at(x, z)).abs().toFloat());
            if (lpl::procgen::slopeAt(original, x, z) > slopeCut)
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

/**
 * @brief The river network keeps its density at every map size.
 *
 * The third time this module has been bitten by a threshold judged against a distribution that moves: a share of
 * the LARGEST flow made 34% of a 24x24 world river and 3.7% of a 128x128 one, from the same recipe. A quantile of the
 * accumulation distribution is scale-free, and that is what this pins — a drainage network, not a water table.
 *
 * The recipe asks for the default density, 4%, and the share of each map is measured per mille: anything from 1.5%
 * to 7% is accepted. A band, not a point, because ties are kept whole and a tiny map has coarse granularity.
 */
void checkRiverDensityIsScaleFree()
{
    const lpl::procgen::RiverParams asked;
    bool densityHolds = true;
    lpl::core::u32 shares[4] = {0u, 0u, 0u, 0u};
    const lpl::core::u32 sizes[4] = {24u, 48u, 64u, 128u};
    for (lpl::core::u32 s = 0u; s < 4u; ++s)
    {
        lpl::procgen::NoiseParams noise;
        noise.seed = 1337u;
        noise.frequency = 0.15f;
        noise.octaves = 4u;
        lpl::procgen::Heightfield field = lpl::procgen::generateNoiseHeightfield(sizes[s], sizes[s], noise);
        lpl::procgen::normalizeHeights(field, lpl::math::Fixed32::fromFloat(-8.0f),
                                       lpl::math::Fixed32::fromFloat(16.0f));
        const lpl::procgen::DrainageNetwork drainage = lpl::procgen::computeDrainage(field);
        const lpl::core::u32 marked = lpl::procgen::carveRivers(field, drainage, asked);
        shares[s] = (marked * 1000u) / (sizes[s] * sizes[s]);
        densityHolds = densityHolds && shares[s] >= 15u && shares[s] <= 70u;
    }
    check(densityHolds, "river density is scale-free: the same recipe wets the same share at any map size");
    std::printf("    river share per mille: 24^2=%u 48^2=%u 64^2=%u 128^2=%u\n", shares[0], shares[1], shares[2],
                shares[3]);
}

/**
 * @brief Hydrology: drainage, rivers and moisture, reproducible and physically consistent.
 *
 * Every cell draining strictly downhill is the invariant that makes it hydrology and not decoration. It is judged on
 * the FILLED surface, which is the one routing was computed over: a cell inside a filled basin drains to a neighbour
 * lower than itself on that surface, and need not be lower on the raw terrain.
 *
 * Depression filling is what makes the whole layer mean anything: without it a noise terrain routes into several
 * hundred separate basins and the largest carries a couple of percent of the map whatever its size. The filled surface
 * must never sit below the terrain: filling raises ground into lakes, it does not excavate.
 *
 * Moisture spanning its full range is the whole point of normalising against the land: the thresholds a biome
 * classifier applies have to be reachable. A ceiling of 0.6 on land made forest, rainforest and marsh impossible
 * however they were configured.
 *
 * Rain shadow: a windward ridge has to dry the land behind it. It is compared against the same world with the shadow
 * switched off, so only the shadow term differs. Wind direction has to matter, or the shadow is not orographic.
 *
 * Climate is the lowest-frequency thing in a world, so its scale is expressed relative to the map: doubling the map
 * must not double the number of wet and dry belts a player crosses.
 */
void checkHydrologyIsReproducibleAndPhysical()
{
    std::printf("\n-- hydrology --\n");
    const lpl::procgen::Heightfield terrain = makeTestTerrain(1337u);
    const lpl::procgen::DrainageNetwork network = lpl::procgen::computeDrainage(terrain);

    check(network.maxAccumulation > 1u, "drainage accumulates flow");
    check(network.maxAccumulation <= terrain.cellCount(), "no cell drains more than the whole map (conservation)");

    bool alwaysDownhill = true;
    for (lpl::core::u32 z = 0u; z < terrain.depth() && alwaysDownhill; ++z)
    {
        for (lpl::core::u32 x = 0u; x < terrain.width(); ++x)
        {
            const lpl::core::u8 direction = network.direction.at(x, z);
            if (direction == lpl::procgen::kNoFlow)
                continue;
            const lpl::core::i32 nx = static_cast<lpl::core::i32>(x) + lpl::procgen::kNeighbor8X[direction];
            const lpl::core::i32 nz = static_cast<lpl::core::i32>(z) + lpl::procgen::kNeighbor8Z[direction];
            if (network.filled.at(static_cast<lpl::core::u32>(nx), static_cast<lpl::core::u32>(nz)) >=
                network.filled.at(x, z))
            {
                alwaysDownhill = false;
                break;
            }
        }
    }
    check(alwaysDownhill, "every cell drains strictly downhill");

    lpl::core::u32 interiorSinks = 0u;
    for (lpl::core::u32 z = 1u; z + 1u < terrain.depth(); ++z)
        for (lpl::core::u32 x = 1u; x + 1u < terrain.width(); ++x)
            if (network.direction.at(x, z) == lpl::procgen::kNoFlow)
                ++interiorSinks;
    check(interiorSinks == 0u, "depression filling leaves no interior sink");
    check(network.maxAccumulation > terrain.cellCount() / 8u,
          "the trunk drains a real share of the map, not one basin");

    bool fillOnlyRaises = true;
    for (lpl::core::u32 i = 0u; i < terrain.cellCount(); ++i)
        fillOnlyRaises = fillOnlyRaises && network.filled[i] >= terrain[i];
    check(fillOnlyRaises, "filling only ever raises ground");

    const lpl::procgen::DrainageNetwork twin = lpl::procgen::computeDrainage(terrain);
    check(twin.maxAccumulation == network.maxAccumulation, "drainage is reproducible");
    check(lpl::procgen::foldHeightfield(twin.filled) == lpl::procgen::foldHeightfield(network.filled),
          "the filled surface is reproducible");

    lpl::procgen::Heightfield carved = terrain;
    const lpl::core::u32 riverCells = lpl::procgen::carveRivers(carved, network, lpl::procgen::RiverParams{});
    check(riverCells > 0u, "rivers are carved");
    check(lpl::procgen::foldHeightfield(carved) != lpl::procgen::foldHeightfield(terrain), "carving lowers the bed");

    checkRiverDensityIsScaleFree();

    const lpl::procgen::Heightfield moisture =
        lpl::procgen::computeMoisture(terrain, network, lpl::procgen::MoistureParams{});
    bool inRange = true;
    for (lpl::core::u32 i = 0u; i < moisture.cellCount(); ++i)
        inRange = inRange && moisture[i] >= lpl::math::Fixed32::zero() && moisture[i] <= lpl::math::Fixed32::one();
    check(inRange, "moisture stays within [0, 1]");

    lpl::math::Fixed32 wettest = lpl::math::Fixed32::zero();
    lpl::math::Fixed32 driest = lpl::math::Fixed32::one();
    for (lpl::core::u32 i = 0u; i < moisture.cellCount(); ++i)
    {
        if (moisture[i] > wettest)
            wettest = moisture[i];
        if (moisture[i] < driest)
            driest = moisture[i];
    }
    check(wettest > lpl::math::Fixed32::fromFloat(0.9f) && driest < lpl::math::Fixed32::fromFloat(0.1f),
          "moisture spans its full range, so thresholds are reachable");

    lpl::procgen::MoistureParams shadowed;
    shadowed.rainShadow = 0.8f;
    lpl::procgen::MoistureParams unshadowed = shadowed;
    unshadowed.rainShadow = 0.0f;
    check(lpl::procgen::foldHeightfield(lpl::procgen::computeMoisture(terrain, network, shadowed)) !=
              lpl::procgen::foldHeightfield(lpl::procgen::computeMoisture(terrain, network, unshadowed)),
          "the rain shadow changes the climate");

    lpl::procgen::MoistureParams eastward = shadowed;
    eastward.windDirection = 0u;
    lpl::procgen::MoistureParams westward = shadowed;
    westward.windDirection = 1u;
    check(lpl::procgen::foldHeightfield(lpl::procgen::computeMoisture(terrain, network, eastward)) !=
              lpl::procgen::foldHeightfield(lpl::procgen::computeMoisture(terrain, network, westward)),
          "reversing the wind moves the dry side");

    const lpl::procgen::Heightfield small =
        lpl::procgen::generateNoiseHeightfield(48u, 48u, lpl::procgen::NoiseParams{});
    const lpl::procgen::Heightfield large =
        lpl::procgen::generateNoiseHeightfield(96u, 96u, lpl::procgen::NoiseParams{});
    const auto beltCrossings = [](const lpl::procgen::Heightfield &field) {
        const lpl::procgen::DrainageNetwork net = lpl::procgen::computeDrainage(field);
        const lpl::procgen::Heightfield wet = lpl::procgen::computeMoisture(field, net, lpl::procgen::MoistureParams{});
        lpl::core::u32 crossings = 0u;
        const lpl::core::u32 row = field.depth() / 2u;
        bool above = wet.at(0u, row) > lpl::math::Fixed32::half();
        for (lpl::core::u32 x = 1u; x < field.width(); ++x)
        {
            const bool now = wet.at(x, row) > lpl::math::Fixed32::half();
            if (now != above)
                ++crossings;
            above = now;
        }
        return crossings;
    };
    const lpl::core::u32 smallBelts = beltCrossings(small);
    const lpl::core::u32 largeBelts = beltCrossings(large);
    check(largeBelts <= smallBelts + 3u, "climate belts do not multiply with map size");
    std::printf("    belts: 48x48 -> %u crossings, 96x96 -> %u crossings\n", smallBelts, largeBelts);
}

/**
 * @brief Biomes: classified from altitude and climate, reproducibly, across most of the palette.
 *
 * The thresholds are absolute world units, so the terrain has to be put in the range they describe. Reading the
 * range off the params rather than hardcoding one is the point: a test that assumed sea level was zero would pass or
 * fail on the default rather than on the classifier.
 *
 * Altitude must dominate: nothing below sea level may be dry land.
 *
 * The lapse rate is what bands a mountain. Without it a summit is classified like the valley it rises from, so the
 * same world with the rate at zero must come out differently — and warmer up high.
 *
 * A world this varied should reach most of the palette. Six of the twelve were unreachable at any setting before
 * the climate model was fixed.
 */
void checkBiomesFollowAltitudeAndClimate()
{
    std::printf("\n-- biomes --\n");
    const lpl::procgen::BiomeParams params;

    lpl::procgen::Heightfield terrain = makeTestTerrain(1337u);
    lpl::procgen::normalizeHeights(terrain, lpl::math::Fixed32::fromFloat(params.seaLevel - 4.0f),
                                   lpl::math::Fixed32::fromFloat(params.snowHeight + 2.0f));

    lpl::procgen::MoistureParams climate;
    climate.seaLevel = params.seaLevel;
    const lpl::procgen::DrainageNetwork network = lpl::procgen::computeDrainage(terrain);
    const lpl::procgen::Heightfield moisture = lpl::procgen::computeMoisture(terrain, network, climate);

    lpl::procgen::ClimateParams axes;
    axes.seaLevel = params.seaLevel;
    const lpl::procgen::ClimateField climateField = lpl::procgen::computeClimate(terrain, moisture, network, axes);

    const lpl::procgen::BiomeMap map = lpl::procgen::classifyBiomes(terrain, climateField, params);
    check(!map.empty(), "biomes are classified");

    lpl::core::u32 counts[static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count)] = {};
    lpl::procgen::countBiomes(map, counts);

    lpl::core::u32 distinct = 0u;
    for (lpl::core::u32 i = 0u; i < static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count); ++i)
        if (counts[i] != 0u)
            ++distinct;
    check(distinct >= 3u, "a varied terrain yields several biomes");

    bool oceanConsistent = true;
    for (lpl::core::u32 i = 0u; i < map.cellCount(); ++i)
        if (terrain[i] <= lpl::math::Fixed32::fromFloat(params.seaLevel) && map[i] != lpl::procgen::BiomeId::Ocean)
            oceanConsistent = false;
    check(oceanConsistent, "everything at or below sea level is ocean");

    lpl::procgen::ClimateParams flatAxes = axes;
    flatAxes.lapseRate = 0.0f;
    const lpl::procgen::ClimateField flatField = lpl::procgen::computeClimate(terrain, moisture, network, flatAxes);
    check(lpl::procgen::foldBiomeMap(lpl::procgen::classifyBiomes(terrain, flatField, params)) !=
              lpl::procgen::foldBiomeMap(map),
          "the lapse rate changes the classification");

    const lpl::procgen::BiomeMap twin = lpl::procgen::classifyBiomes(terrain, climateField, params);
    check(lpl::procgen::foldBiomeMap(map) == lpl::procgen::foldBiomeMap(twin), "classification is reproducible");

    check(distinct >= 9u, "a full world reaches most of the biome palette");

    std::printf("    ");
    for (lpl::core::u32 i = 0u; i < static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count); ++i)
        if (counts[i] != 0u)
            std::printf("%s=%u ", lpl::procgen::biomeName(static_cast<lpl::procgen::BiomeId>(i)), counts[i]);
    std::printf("\n");
}

/**
 * @brief Wave function collapse: it solves without breaking a rule, reproducibly, and around pinned cells.
 *
 * Border constraints are the seam mechanism for chunked worlds.
 */
void checkWaveFunctionCollapseKeepsItsRules()
{
    std::printf("\n-- wave function collapse --\n");
    const lpl::procgen::TileSet tiles = lpl::procgen::makeTerrainTileSet();
    check(tiles.valid(), "the terrain tile set is well formed");

    lpl::procgen::WfcParams params;
    params.width = 24u;
    params.depth = 24u;
    params.seed = 1337u;

    const lpl::procgen::WfcResult a = lpl::procgen::solveWfc(tiles, params);
    check(a.solved, "WFC solves the grid");
    check(lpl::procgen::countAdjacencyViolations(a.tiles, tiles) == 0u, "the solution violates no adjacency rule");

    const lpl::procgen::WfcResult b = lpl::procgen::solveWfc(tiles, params);
    bool identical = a.tiles.cellCount() == b.tiles.cellCount();
    for (lpl::core::u32 i = 0u; identical && i < a.tiles.cellCount(); ++i)
        identical = a.tiles[i] == b.tiles[i];
    check(identical, "the same seed reproduces the arrangement");

    lpl::procgen::WfcParams other = params;
    other.seed = 2024u;
    const lpl::procgen::WfcResult c = lpl::procgen::solveWfc(tiles, other);
    bool differs = false;
    for (lpl::core::u32 i = 0u; i < a.tiles.cellCount() && i < c.tiles.cellCount(); ++i)
        differs = differs || a.tiles[i] != c.tiles[i];
    check(differs, "a different seed gives a different arrangement");

    lpl::procgen::TileGrid preset{params.width, params.depth, lpl::procgen::kNoTile};
    for (lpl::core::u32 z = 0u; z < params.depth; ++z)
        preset.at(0u, z) = static_cast<lpl::core::u8>(lpl::procgen::TerrainTile::Water);
    const lpl::procgen::WfcResult constrained = lpl::procgen::solveWfc(tiles, params, &preset);
    check(constrained.solved, "WFC solves around pinned cells");
    bool borderHeld = true;
    for (lpl::core::u32 z = 0u; z < params.depth; ++z)
        borderHeld =
            borderHeld && constrained.tiles.at(0u, z) == static_cast<lpl::core::u8>(lpl::procgen::TerrainTile::Water);
    check(borderHeld, "pinned cells survive the solve");
    check(lpl::procgen::countAdjacencyViolations(constrained.tiles, tiles) == 0u,
          "the constrained solution is still legal");

    std::printf("    attempts=%u contradictions=%u localRepairs=%u\n", a.attempts, a.contradictions, a.localRepairs);
}

/**
 * @brief Dungeons: every generator is reproducible and leaves a connected map.
 *
 * Erosion must not break the guarantee the generators just established.
 */
void checkDungeonsAreReproducibleAndConnected()
{
    std::printf("\n-- dungeons --\n");
    lpl::procgen::BspDungeonParams bsp;
    bsp.width = 64u;
    bsp.depth = 64u;
    bsp.seed = 1337u;
    lpl::pmr::vector<lpl::procgen::Room> rooms;
    lpl::procgen::DungeonMap dungeon = lpl::procgen::generateBspDungeon(bsp, &rooms);
    check(rooms.size() >= 4u, "BSP carves several rooms");
    check(lpl::procgen::isFullyConnected(dungeon), "BSP guarantees connectivity by construction");

    lpl::procgen::DungeonMap twin = lpl::procgen::generateBspDungeon(bsp);
    check(lpl::procgen::foldDungeon(dungeon) == lpl::procgen::foldDungeon(twin), "BSP is reproducible");

    lpl::procgen::CaveParams cave;
    cave.width = 64u;
    cave.depth = 64u;
    cave.seed = 1337u;
    lpl::procgen::DungeonMap caves = lpl::procgen::generateCellularCave(cave);
    check(lpl::procgen::isFullyConnected(caves), "caves are connected after the repair pass");

    lpl::core::u32 floor = 0u;
    for (lpl::core::u32 i = 0u; i < caves.cellCount(); ++i)
        if (caves[i] != lpl::procgen::DungeonCell::Wall)
            ++floor;
    check(floor > caves.cellCount() / 10u, "caves leave a usable amount of open space");

    lpl::procgen::DungeonMap cavesTwin = lpl::procgen::generateCellularCave(cave);
    check(lpl::procgen::foldDungeon(caves) == lpl::procgen::foldDungeon(cavesTwin), "caves are reproducible");

    lpl::procgen::DrunkardParams walk;
    walk.width = 48u;
    walk.depth = 48u;
    walk.seed = 1337u;
    lpl::procgen::DungeonMap galleries = lpl::procgen::generateDrunkardWalk(walk);
    check(lpl::procgen::isFullyConnected(galleries), "the drunkard's galleries are connected");
    check(lpl::procgen::foldDungeon(galleries) == lpl::procgen::foldDungeon(lpl::procgen::generateDrunkardWalk(walk)),
          "the walk is reproducible");

    lpl::procgen::DungeonMap eroded = dungeon;
    lpl::procgen::erodeEdges(eroded, 1337u, 0.25f);
    check(lpl::procgen::foldDungeon(eroded) != lpl::procgen::foldDungeon(dungeon), "edge erosion changes the map");
    check(lpl::procgen::isFullyConnected(eroded), "edge erosion preserves connectivity");
}

/**
 * @brief The world builder: the whole pipeline, reproducible end to end.
 *
 * The short form must not be a different engine from the long one.
 */
void checkTheWorldBuilderIsReproducibleEndToEnd()
{
    std::printf("\n-- world builder --\n");
    lpl::ecs::Registry registry;
    const lpl::procgen::BuiltWorldStats stats = lpl::procgen::WorldBuilder{1337u}
                                                    .terrain(48u, 48u)
                                                    .normalize(-4.0f, 28.0f)
                                                    .erode()
                                                    .rivers()
                                                    .biomes()
                                                    .scatterInBiome(lpl::procgen::BiomeId::Forest, 0.15f)
                                                    .scatterInBiome(lpl::procgen::BiomeId::Grassland, 0.05f)
                                                    .materialize(registry);

    check(stats.terrainCells == 48u * 48u, "the builder built the requested terrain");
    check(stats.terrainEntities == stats.terrainCells, "every ground cell became an entity");
    check(stats.propEntities > 0u, "scatter rules placed props");
    check(stats.riverCells > 0u, "the builder carved rivers");

    lpl::ecs::Registry twinRegistry;
    const lpl::procgen::BuiltWorldStats twin = lpl::procgen::WorldBuilder{1337u}
                                                   .terrain(48u, 48u)
                                                   .normalize(-4.0f, 28.0f)
                                                   .erode()
                                                   .rivers()
                                                   .biomes()
                                                   .scatterInBiome(lpl::procgen::BiomeId::Forest, 0.15f)
                                                   .scatterInBiome(lpl::procgen::BiomeId::Grassland, 0.05f)
                                                   .materialize(twinRegistry);
    check(twin.heightSignature == stats.heightSignature && twin.biomeSignature == stats.biomeSignature &&
              twin.propEntities == stats.propEntities,
          "the whole pipeline is reproducible end to end");

    lpl::ecs::Registry otherRegistry;
    const lpl::procgen::BuiltWorldStats other = lpl::procgen::WorldBuilder{2024u}
                                                    .terrain(48u, 48u)
                                                    .normalize(-4.0f, 28.0f)
                                                    .erode()
                                                    .rivers()
                                                    .biomes()
                                                    .materialize(otherRegistry);
    check(other.heightSignature != stats.heightSignature, "a different seed builds a different world");

    lpl::procgen::WorldBuilder lazy{4242u};
    const lpl::procgen::BuiltWorldStats implied = lazy.biomes().bakeGrids();
    check(implied.terrainCells > 0u, "asking for biomes alone still produces terrain first");
    check(!lazy.biomeMap().empty(), "dependencies run in the right order automatically");

    std::printf("    ground=%u props=%u rivers=%u height_sig=0x%08X biome_sig=0x%08X\n", stats.terrainEntities,
                stats.propEntities, stats.riverCells, stats.heightSignature, stats.biomeSignature);
}

} // namespace

/**
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
 */
int main()
{
    std::printf("== procedural passes: determinism and invariants ==\n\n");

    checkTheRandomStreamReplaysAndSpreads();
    checkTheHeightfieldIsReproducibleAndShapedAsAsked();
    checkErosionIsReproducibleAndFollowsTheSlope();
    checkHydrologyIsReproducibleAndPhysical();
    checkBiomesFollowAltitudeAndClimate();
    checkWaveFunctionCollapseKeepsItsRules();
    checkDungeonsAreReproducibleAndConnected();
    checkTheWorldBuilderIsReproducibleEndToEnd();

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
