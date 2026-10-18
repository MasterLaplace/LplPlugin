#include <lpl/procgen/Aggregation.hpp>
#include <lpl/procgen/Chunking.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/procgen/Extrusion.hpp>
#include <lpl/procgen/LSystem.hpp>
#include <lpl/procgen/QualityGate.hpp>
#include <lpl/procgen/Settlement.hpp>
#include <lpl/procgen/Voronoi.hpp>
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
 * @brief Voronoi: a partition of the plane into the regions of the nearest sites, reproducible and sensitive to
 *        its seed.
 *
 * The defining property: every cell belongs to exactly one region, and that region is the nearest site. A partition
 * that misses cells is not a partition.
 *
 * The 3x3 search is only valid if it agrees with an exhaustive one.
 */
void checkVoronoiPartitionsThePlane()
{
    std::printf("-- voronoi --\n");
    lpl::procgen::VoronoiParams params;
    params.width = 64u;
    params.depth = 64u;
    params.seed = 1337u;
    params.cellSize = 12u;

    const lpl::procgen::VoronoiDiagram diagram = lpl::procgen::computeVoronoi(params);
    check(diagram.regionCount > 1u, "the plane is partitioned into regions");

    bool everyCellClaimed = true;
    for (lpl::core::u32 i = 0u; i < diagram.regions.cellCount(); ++i)
        if (diagram.regions[i] == lpl::procgen::kNoRegion)
            everyCellClaimed = false;
    check(everyCellClaimed, "every cell is claimed by a region");

    bool matchesBruteForce = true;
    for (lpl::core::u32 z = 0u; z < params.depth && matchesBruteForce; z += 7u)
    {
        for (lpl::core::u32 x = 0u; x < params.width; x += 7u)
        {
            lpl::math::Fixed32 best = lpl::math::Fixed32::max();
            lpl::core::u16 bestRegion = lpl::procgen::kNoRegion;
            for (lpl::core::u32 s = 0u; s < diagram.sites.size(); ++s)
            {
                const lpl::math::Fixed32 dx =
                    lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(x)) - diagram.sites[s].x;
                const lpl::math::Fixed32 dz =
                    lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(z)) - diagram.sites[s].z;
                const lpl::math::Fixed32 d2 = dx * dx + dz * dz;
                if (d2 < best)
                {
                    best = d2;
                    bestRegion = static_cast<lpl::core::u16>(s);
                }
            }
            if (diagram.regions.at(x, z) != bestRegion)
            {
                matchesBruteForce = false;
                break;
            }
        }
    }
    check(matchesBruteForce, "the 3x3 site search finds the true nearest site");

    const lpl::procgen::VoronoiDiagram twin = lpl::procgen::computeVoronoi(params);
    check(lpl::procgen::foldRegionMap(diagram.regions) == lpl::procgen::foldRegionMap(twin.regions),
          "the partition is reproducible");

    lpl::procgen::VoronoiParams other = params;
    other.seed = 2024u;
    check(lpl::procgen::foldRegionMap(lpl::procgen::computeVoronoi(other).regions) !=
              lpl::procgen::foldRegionMap(diagram.regions),
          "a different seed moves the sites");

    const lpl::procgen::Grid<lpl::core::u8> borders = lpl::procgen::regionBorders(diagram);
    lpl::core::u32 borderCells = 0u;
    for (lpl::core::u32 i = 0u; i < borders.cellCount(); ++i)
        borderCells += borders[i];
    check(borderCells > 0u && borderCells < borders.cellCount() / 2u, "borders exist and are a minority of the map");
}

/**
 * @brief Diffusion-limited aggregation: a cluster that grows, stays in one piece and branches.
 *
 * The property that makes it DLA rather than a blob: sticking on contact means the whole cluster is one connected
 * piece.
 *
 * A dendrite is sparse: it reaches far while occupying little. A blob filling its bounding box would fail this.
 */
void checkAggregationGrowsOneBranchingStructure()
{
    std::printf("\n-- diffusion-limited aggregation --\n");
    lpl::procgen::DlaParams params;
    params.width = 48u;
    params.depth = 48u;
    params.seed = 1337u;
    params.particles = 300u;

    lpl::procgen::DlaReport report;
    const lpl::procgen::DungeonMap cluster = lpl::procgen::generateDlaCave(params, &report);
    check(report.stuck > 0u, "particles aggregate onto the cluster");
    check(report.openCells > 1u, "the cluster grows beyond its seed");
    check(report.extent > 1u, "the cluster spreads out from the centre");

    check(lpl::procgen::isFullyConnected(cluster), "the aggregate is a single connected structure");

    const lpl::core::u32 boundingArea = (report.extent * 2u + 1u) * (report.extent * 2u + 1u);
    check(report.openCells < boundingArea / 2u, "the growth is branching, not a filled blob");

    lpl::procgen::DungeonMap twin = lpl::procgen::generateDlaCave(params);
    check(lpl::procgen::foldDungeon(cluster) == lpl::procgen::foldDungeon(twin), "aggregation is reproducible");

    std::printf("    stuck=%u abandoned=%u open=%u extent=%u\n", report.stuck, report.abandoned, report.openCells,
                report.extent);
}

/**
 * @brief L-systems: rewriting grows deterministically within its cap, and the turtle draws what it rewrote.
 *
 * Exponential growth must respect the cap, or a kernel heap decides how this ends.
 *
 * The turtle starts at the bottom of the canvas heading "up" the grid. Brackets must restore state: without a
 * working stack the turtle wanders off instead of returning to the branch point.
 *
 * More rewriting must draw MORE, and it did the opposite: the turtle walked off the canvas and every later symbol
 * was silently clipped, so 173 draw symbols painted 43 cells while 4156 painted 32. Growth running backwards is
 * invisible to a fold and to any "did it draw something" check — only the trend shows it.
 */
void checkLSystemsGrowAndDrawWhatTheyRewrite()
{
    std::printf("\n-- L-systems --\n");
    const lpl::procgen::LSystemParams grammar = lpl::procgen::makeBranchingGrammar();
    const lpl::pmr::string expanded = lpl::procgen::expandLSystem(grammar);
    check(expanded.size() > grammar.axiom.size(), "rewriting grows the string");

    const lpl::pmr::string twin = lpl::procgen::expandLSystem(grammar);
    check(expanded == twin, "rewriting is deterministic");

    lpl::procgen::LSystemParams runaway = grammar;
    runaway.iterations = 20u;
    runaway.maxLength = 4096u;
    check(lpl::procgen::expandLSystem(runaway).size() <= runaway.maxLength, "expansion respects its length cap");

    lpl::procgen::Grid<lpl::core::u8> canvas{64u, 64u, 0u};
    lpl::procgen::TurtleParams turtle;
    turtle.startX = 32u;
    turtle.startZ = 60u;
    turtle.startDirection = 12u;
    turtle.stepLength = 3u;
    turtle.turnAmount = 2u;
    const lpl::core::u32 drawn = lpl::procgen::drawTurtle(expanded, turtle, canvas);
    check(drawn > 0u, "the turtle draws the structure");

    lpl::core::u32 filled = 0u;
    for (lpl::core::u32 i = 0u; i < canvas.cellCount(); ++i)
        filled += canvas[i];
    check(filled == drawn, "every drawn cell is accounted for");

    lpl::procgen::Grid<lpl::core::u8> noBranches{64u, 64u, 0u};
    const lpl::core::u32 straight = lpl::procgen::drawTurtle(lpl::pmr::string{"FFFF"}, turtle, noBranches);
    check(straight > 0u && straight < drawn, "a straight run draws less than a branching one");

    const lpl::procgen::LSystemParams roads = lpl::procgen::makeRoadGrammar();
    check(lpl::procgen::expandLSystem(roads).size() > 0u, "the road grammar expands");

    lpl::procgen::LSystemParams growing = lpl::procgen::makeRoadGrammar();
    lpl::core::u32 previous = 0u;
    bool growsWithIterations = true;
    lpl::core::u32 drawnPerRound[4] = {0u, 0u, 0u, 0u};
    for (lpl::core::u32 round = 0u; round < 4u; ++round)
    {
        growing.iterations = 4u + round;
        lpl::procgen::Grid<lpl::core::u8> sheet{64u, 64u, 0u};
        lpl::procgen::TurtleParams walker;
        walker.startX = 32u;
        walker.startZ = 32u;
        walker.stepLength = 3u;
        drawnPerRound[round] = lpl::procgen::drawTurtle(lpl::procgen::expandLSystem(growing), walker, sheet);
        growsWithIterations = growsWithIterations && drawnPerRound[round] >= previous;
        previous = drawnPerRound[round];
    }
    check(growsWithIterations, "a longer expansion draws more, not less (the turtle stays on the canvas)");
    std::printf("    road grammar draws: %u %u %u %u cells at 4..7 rounds\n", drawnPerRound[0], drawnPerRound[1],
                drawnPerRound[2], drawnPerRound[3]);
}

/**
 * @brief Settlements: a town whose streets join up, whose buildings face a street, and that keeps off cliffs and water.
 *
 * A single connected street network is the property the whole ordering exists to guarantee. A plot that faces no
 * street is a building nobody can enter.
 *
 * On terrain, nothing may stand on a cliff or in water.
 */
void checkSettlementsJoinUpAndStayOnSoundGround()
{
    std::printf("\n-- settlements --\n");
    lpl::procgen::SettlementParams params;
    params.width = 96u;
    params.depth = 96u;
    params.seed = 1337u;

    lpl::pmr::vector<lpl::procgen::BuildingPlot> plots;
    lpl::procgen::SettlementReport report;
    const lpl::procgen::SettlementMap town = lpl::procgen::generateSettlement(params, &plots, &report);

    check(report.districts > 1u, "the town is divided into districts");
    check(report.roadCells > 0u, "streets are laid");
    check(report.plots > 0u, "buildings are placed");
    check(plots.size() == report.plots, "every plot is reported");

    check(report.roadsConnected, "the street network is a single connected piece");

    bool everyPlotReachable = true;
    for (lpl::core::u32 p = 0u; p < plots.size(); ++p)
    {
        const lpl::procgen::BuildingPlot &plot = plots[p];
        bool touchesStreet = false;
        for (lpl::core::i32 dz = -1; dz <= static_cast<lpl::core::i32>(plot.depth) && !touchesStreet; ++dz)
            for (lpl::core::i32 dx = -1; dx <= static_cast<lpl::core::i32>(plot.width); ++dx)
            {
                const lpl::core::i32 cx = static_cast<lpl::core::i32>(plot.x) + dx;
                const lpl::core::i32 cz = static_cast<lpl::core::i32>(plot.z) + dz;
                if (!town.contains(cx, cz))
                    continue;
                const lpl::procgen::SettlementCell cell =
                    town.at(static_cast<lpl::core::u32>(cx), static_cast<lpl::core::u32>(cz));
                if (cell == lpl::procgen::SettlementCell::Road || cell == lpl::procgen::SettlementCell::Plaza)
                {
                    touchesStreet = true;
                    break;
                }
            }
        if (!touchesStreet)
            everyPlotReachable = false;
    }
    check(everyPlotReachable, "every building faces a street");

    check(lpl::procgen::foldSettlement(town) == lpl::procgen::foldSettlement(lpl::procgen::generateSettlement(params)),
          "the layout is reproducible");

    lpl::procgen::NoiseParams noise;
    noise.seed = 1337u;
    noise.frequency = 0.05f;
    noise.amplitude = 16.0f;
    lpl::procgen::Heightfield ground = lpl::procgen::generateNoiseHeightfield(params.width, params.depth, noise);
    lpl::procgen::normalizeHeights(ground, lpl::math::Fixed32::fromInt(-2), lpl::math::Fixed32::fromInt(18));

    lpl::procgen::SettlementReport terrainReport;
    const lpl::procgen::SettlementMap onTerrain =
        lpl::procgen::generateSettlementOnTerrain(params, ground, nullptr, &terrainReport);
    check(terrainReport.blockedCells > 0u, "unbuildable ground is recognised");

    bool nothingOnBadGround = true;
    const lpl::math::Fixed32 minHeight = lpl::math::Fixed32::fromFloat(params.minHeight);
    const lpl::math::Fixed32 maxSlope = lpl::math::Fixed32::fromFloat(params.maxSlope);
    for (lpl::core::u32 z = 0u; z < onTerrain.depth(); ++z)
        for (lpl::core::u32 x = 0u; x < onTerrain.width(); ++x)
        {
            if (onTerrain.at(x, z) == lpl::procgen::SettlementCell::Empty ||
                onTerrain.at(x, z) == lpl::procgen::SettlementCell::Blocked)
                continue;
            if (ground.at(x, z) < minHeight || lpl::procgen::slopeAt(ground, x, z) > maxSlope)
                nothingOnBadGround = false;
        }
    check(nothingOnBadGround, "nothing is built on a cliff or under water");

    std::printf("    districts=%u roads=%u plots=%u blocked=%u\n", report.districts, report.roadCells, report.plots,
                terrainReport.blockedCells);
}

/**
 * @brief 2.5D extrusion: a solved plan becomes a volume with an interior.
 *
 * Water is a hole, each later tile is taller: the plan's gradient becomes relief. The heights are water 0, sand 1,
 * grass 2, forest 3 and rock 5.
 *
 * A surface smaller than the solid count is what proves the volume has an interior — that is, that it is a shape
 * and not a shell.
 *
 * Hollow mode must cost less than solid mode.
 */
void checkExtrusionTurnsAPlanIntoAShape()
{
    std::printf("\n-- 2.5D extrusion --\n");
    const lpl::procgen::TileSet tiles = lpl::procgen::makeTerrainTileSet();
    lpl::procgen::WfcParams wfc;
    wfc.width = 24u;
    wfc.depth = 24u;
    wfc.seed = 1337u;
    const lpl::procgen::WfcResult plan = lpl::procgen::solveWfc(tiles, wfc);
    check(plan.solved, "the 2D plan solves");

    lpl::pmr::vector<lpl::core::u8> heights;
    heights.push_back(0u);
    heights.push_back(1u);
    heights.push_back(2u);
    heights.push_back(3u);
    heights.push_back(5u);

    lpl::procgen::ExtrusionParams extrusion;
    extrusion.levels = 8u;
    const lpl::procgen::VoxelVolume volume = lpl::procgen::extrudeTilePlan(plan.tiles, heights, extrusion);

    check(!volume.empty(), "the plan extrudes to a volume");
    check(volume.width == wfc.width && volume.depth == wfc.depth, "the volume keeps the plan's footprint");
    check(volume.levels == extrusion.levels, "the volume has the requested height");

    const lpl::core::u32 solid = lpl::procgen::countSolidVoxels(volume);
    check(solid > 0u && solid < volume.voxelCount(), "the volume is neither empty nor full");

    const lpl::core::u32 surface = lpl::procgen::countSurfaceVoxels(volume);
    check(surface > 0u && surface <= solid, "the volume has a coherent surface");

    check(lpl::procgen::foldVolume(volume) ==
              lpl::procgen::foldVolume(lpl::procgen::extrudeTilePlan(plan.tiles, heights, extrusion)),
          "extrusion is reproducible");

    lpl::procgen::ExtrusionParams hollow = extrusion;
    hollow.solidBelow = false;
    check(lpl::procgen::countSolidVoxels(lpl::procgen::extrudeTilePlan(plan.tiles, heights, hollow)) < solid,
          "surface-only extrusion writes fewer voxels");

    std::printf("    solid=%u surface=%u of %u voxels\n", solid, surface, volume.voxelCount());
}

/**
 * @brief Chunked worlds: a chunk depends on its coordinates alone, and neighbours agree on their seams.
 *
 * The whole scheme rests on this: a chunk depends on nothing but its coordinates, so it is identical however and
 * whenever it is built.
 *
 * Seams: the shared edge must agree, in every direction, including across the origin where the coordinates change
 * sign. The world function and the chunk must agree cell for cell.
 *
 * Constraint-solved chunks need their borders pinned explicitly.
 */
void checkChunksAgreeWithTheWorldAndAcrossSeams()
{
    std::printf("\n-- chunked worlds --\n");
    lpl::procgen::ChunkParams params;
    params.size = 32u;
    params.worldSeed = 1337u;
    params.noise.frequency = 0.03f;
    params.noise.amplitude = 20.0f;

    const lpl::procgen::Heightfield origin = lpl::procgen::generateChunkTerrain(params, lpl::procgen::ChunkCoord{0, 0});
    check(origin.width() == params.size, "a chunk is generated at its requested size");

    const lpl::procgen::Heightfield again = lpl::procgen::generateChunkTerrain(params, lpl::procgen::ChunkCoord{0, 0});
    check(lpl::procgen::foldHeightfield(origin) == lpl::procgen::foldHeightfield(again), "a chunk is reproducible");

    const lpl::procgen::Heightfield east = lpl::procgen::generateChunkTerrain(params, lpl::procgen::ChunkCoord{1, 0});
    check(lpl::procgen::foldHeightfield(east) != lpl::procgen::foldHeightfield(origin), "neighbouring chunks differ");

    check(lpl::procgen::countSeamMismatches(params, lpl::procgen::ChunkCoord{0, 0}, lpl::procgen::ChunkCoord{1, 0}) ==
              0u,
          "the eastern seam matches");
    check(lpl::procgen::countSeamMismatches(params, lpl::procgen::ChunkCoord{0, 0}, lpl::procgen::ChunkCoord{0, 1}) ==
              0u,
          "the southern seam matches");
    check(lpl::procgen::countSeamMismatches(params, lpl::procgen::ChunkCoord{0, 0}, lpl::procgen::ChunkCoord{-1, 0}) ==
              0u,
          "the western seam matches");
    check(lpl::procgen::countSeamMismatches(params, lpl::procgen::ChunkCoord{-1, -1},
                                            lpl::procgen::ChunkCoord{0, -1}) == 0u,
          "seams hold at negative coordinates");

    bool agreesWithWorld = true;
    for (lpl::core::u32 i = 0u; i < params.size; ++i)
        if (origin.at(i, 0u).raw() != lpl::procgen::sampleWorldHeight(params, static_cast<lpl::core::i32>(i), 0).raw())
            agreesWithWorld = false;
    check(agreesWithWorld, "a chunk equals the world function over its own cells");

    check(lpl::procgen::chunkSeed(params, lpl::procgen::ChunkCoord{1, 0}) !=
              lpl::procgen::chunkSeed(params, lpl::procgen::ChunkCoord{0, 1}),
          "transposed chunk coordinates get different seeds");

    const lpl::procgen::TileSet tiles = lpl::procgen::makeTerrainTileSet();
    lpl::procgen::WfcParams wfc;
    wfc.width = 16u;
    wfc.depth = 16u;
    wfc.seed = lpl::procgen::chunkSeed(params, lpl::procgen::ChunkCoord{0, 0});
    const lpl::procgen::WfcResult first = lpl::procgen::solveWfc(tiles, wfc);
    check(first.solved, "the first chunk solves");

    const lpl::procgen::TileGrid preset = lpl::procgen::borderConstraintsFrom(16u, first.tiles, 1u);
    lpl::procgen::WfcParams nextParams = wfc;
    nextParams.seed = lpl::procgen::chunkSeed(params, lpl::procgen::ChunkCoord{1, 0});
    const lpl::procgen::WfcResult second = lpl::procgen::solveWfc(tiles, nextParams, &preset);
    check(second.solved, "the neighbouring chunk solves against the seam");

    bool seamAgrees = true;
    for (lpl::core::u32 i = 0u; i < 16u; ++i)
        if (second.tiles.at(0u, i) != first.tiles.at(15u, i))
            seamAgrees = false;
    check(seamAgrees, "the WFC seam is continuous across chunks");
}

/**
 * @brief Playability gates: a generated level is measured, accepted when playable and rejected when not.
 *
 * The gate has to reject as well as accept, or it is decoration.
 *
 * A distance map is a flow field: descending it must always approach.
 */
void checkPlayabilityGatesAcceptAndReject()
{
    std::printf("\n-- playability gates --\n");
    lpl::procgen::BspDungeonParams bsp;
    bsp.width = 64u;
    bsp.depth = 64u;
    bsp.seed = 1337u;
    const lpl::procgen::DungeonMap level = lpl::procgen::generateBspDungeon(bsp);

    lpl::core::u32 startX = 0u, startZ = 0u, goalX = 0u, goalZ = 0u;
    check(lpl::procgen::findFarthestPair(level, startX, startZ, goalX, goalZ),
          "entrance and exit are found automatically");

    const lpl::procgen::LevelQuality quality = lpl::procgen::evaluateLevel(level, startX, startZ, goalX, goalZ);
    check(quality.goalReachable, "the exit is reachable from the entrance");
    check(quality.fullyConnected, "no part of the level is walled off");
    check(quality.pathLength > 0u, "the exit is not on top of the entrance");
    check(quality.reachableCells == quality.walkableCells, "every walkable cell is reached");
    check(quality.junctions > 0u, "the level branches rather than being one corridor");

    check(lpl::procgen::passesGate(quality, lpl::procgen::GateCriteria{}), "a BSP dungeon passes the default gate");

    lpl::procgen::DungeonMap sealed{32u, 32u, lpl::procgen::DungeonCell::Wall};
    sealed.at(1u, 1u) = lpl::procgen::DungeonCell::Floor;
    sealed.at(30u, 30u) = lpl::procgen::DungeonCell::Floor;
    const lpl::procgen::LevelQuality bad = lpl::procgen::evaluateLevel(sealed, 1u, 1u, 30u, 30u);
    check(!bad.goalReachable, "a sealed-off exit is detected");
    check(!lpl::procgen::passesGate(bad, lpl::procgen::GateCriteria{}), "an unplayable level is rejected");

    const lpl::procgen::DistanceMap distances = lpl::procgen::computeDistanceMap(level, startX, startZ);
    bool monotone = true;
    for (lpl::core::u32 z = 0u; z < level.depth() && monotone; ++z)
        for (lpl::core::u32 x = 0u; x < level.width(); ++x)
        {
            const lpl::core::u32 here = distances.at(x, z);
            if (here == lpl::procgen::kUnreachable || here == 0u)
                continue;
            bool hasCloserNeighbour = false;
            for (lpl::core::u32 n = 0u; n < 4u; ++n)
            {
                const lpl::core::i32 nx = static_cast<lpl::core::i32>(x) + lpl::procgen::kNeighbor4X[n];
                const lpl::core::i32 nz = static_cast<lpl::core::i32>(z) + lpl::procgen::kNeighbor4Z[n];
                if (distances.contains(nx, nz) &&
                    distances.at(static_cast<lpl::core::u32>(nx), static_cast<lpl::core::u32>(nz)) == here - 1u)
                    hasCloserNeighbour = true;
            }
            if (!hasCloserNeighbour)
            {
                monotone = false;
                break;
            }
        }
    check(monotone, "every reachable cell has a step toward the entrance");

    std::printf("    walkable=%u path=%u deadEnds=%u junctions=%u longest=%u\n", quality.walkableCells,
                quality.pathLength, quality.deadEnds, quality.junctions, quality.longestDistance);
}

/**
 * @brief The builder reaches every generator.
 *
 * Each of these three had a working implementation, a passing test, and no
 * caller: nothing a world was ever built from went through them. A generator
 * reachable only from its own test is not part of the module, so what this
 * block pins is the wiring rather than the algorithm.
 *
 * Roads: an L-system steered by a tensor field, vetoed by the ground. Local constraints actually bind: no road on
 * ground it refused.
 *
 * Extrusion: the plan gains height, and the town is taller than its streets. The underground is a volume too: rock
 * is solid, floor is open.
 *
 * Chunking: the same seed describes one endless world, and a chunk of it agrees with the whole map about the ground
 * they share. The right edge of chunk (1,-2) is asked twice, of the builder's chunk and of the world sampler.
 */
void checkTheBuilderReachesEveryGenerator()
{
    std::printf("\n-- builder wiring --\n");

    lpl::procgen::SettlementParams town;
    town.districtSize = 12u;

    lpl::procgen::WorldBuilder city{4242u};
    city.terrain(64u, 64u).normalize(-8.0f, 16.0f).biomes().regions(12u).settlement(town).roads();
    const lpl::procgen::BuiltWorldStats built = city.bakeGrids();
    check(built.roadCells > 0u, "the builder grows a road network");
    check(city.roadMap().width() == 64u && city.roadMap().depth() == 64u, "the road mask covers the world");

    lpl::procgen::RoadParams strict;
    strict.maxSlope = 0.05f;
    lpl::procgen::WorldBuilder steep{4242u};
    steep.terrain(64u, 64u).normalize(-8.0f, 16.0f).biomes().regions(12u).settlement(town).roads(strict);
    (void) steep.bakeGrids();
    bool slopeRespected = true;
    for (lpl::core::u32 z = 0u; z < 64u && slopeRespected; ++z)
        for (lpl::core::u32 x = 0u; x < 64u; ++x)
            if (steep.roadMap().at(x, z) != 0u &&
                lpl::procgen::slopeAt(steep.heightfield(), x, z) > lpl::math::Fixed32::fromFloat(strict.maxSlope))
            {
                slopeRespected = false;
                break;
            }
    check(slopeRespected, "the ground vetoes roads the grammar proposed on slopes it refuses");

    lpl::procgen::ExtrusionParams volume;
    volume.levels = 8u;
    city.extrudeTown(volume);
    const lpl::procgen::VoxelVolume &raised = city.townVolume();
    check(!raised.empty() && raised.levels == 8u, "the builder raises the settlement into a volume");
    check(lpl::procgen::countSolidVoxels(raised) > 0u, "the raised town has solid voxels");

    lpl::core::u32 tallestPlot = 0u;
    lpl::core::u32 tallestRoad = 0u;
    for (lpl::core::u32 z = 0u; z < raised.depth; ++z)
        for (lpl::core::u32 x = 0u; x < raised.width; ++x)
        {
            lpl::core::u32 column = 0u;
            for (lpl::core::u32 y = 0u; y < raised.levels; ++y)
                if (raised.at(x, y, z) != 0u)
                    column = y + 1u;
            const lpl::procgen::SettlementCell kind = city.settlementMap().at(x, z);
            if (kind == lpl::procgen::SettlementCell::Plot && column > tallestPlot)
                tallestPlot = column;
            if (kind == lpl::procgen::SettlementCell::Road && column > tallestRoad)
                tallestRoad = column;
        }
    check(tallestPlot > tallestRoad, "buildings stand above their streets");

    lpl::procgen::CaveParams caves;
    caves.width = 48u;
    caves.depth = 48u;
    lpl::procgen::WorldBuilder below{4242u};
    below.terrain(48u, 48u).caves(caves).extrudeUnderground(volume);
    const lpl::procgen::VoxelVolume &cavern = below.undergroundVolume();
    check(!cavern.empty(), "the builder raises the underground into a volume");
    bool floorIsOpen = true;
    for (lpl::core::u32 z = 0u; z < cavern.depth && floorIsOpen; ++z)
        for (lpl::core::u32 x = 0u; x < cavern.width; ++x)
            if (below.dungeonMap().at(x, z) != lpl::procgen::DungeonCell::Wall && cavern.at(x, 0u, z) != 0u)
            {
                floorIsOpen = false;
                break;
            }
    check(floorIsOpen, "open cells stay open through the extrusion");

    lpl::procgen::NoiseParams noise;
    noise.seed = 4242u;
    noise.frequency = 0.05f;
    lpl::procgen::WorldBuilder tile{4242u};
    tile.chunk(lpl::procgen::ChunkCoord{1, -2}).terrain(32u, 32u, noise);
    lpl::procgen::WorldBuilder sameTile{4242u};
    sameTile.chunk(lpl::procgen::ChunkCoord{1, -2}).terrain(32u, 32u, noise);
    lpl::procgen::WorldBuilder otherTile{4242u};
    otherTile.chunk(lpl::procgen::ChunkCoord{2, -2}).terrain(32u, 32u, noise);
    check(lpl::procgen::foldHeightfield(tile.heightfield()) == lpl::procgen::foldHeightfield(sameTile.heightfield()),
          "the same chunk of the same world is the same terrain");
    check(lpl::procgen::foldHeightfield(tile.heightfield()) != lpl::procgen::foldHeightfield(otherTile.heightfield()),
          "a different chunk is different terrain");

    bool seamless = true;
    for (lpl::core::u32 z = 0u; z < 32u; ++z)
    {
        const lpl::math::Fixed32 mine = tile.heightfield().at(31u, z);
        const lpl::math::Fixed32 theirs = lpl::procgen::sampleWorldHeight(
            lpl::procgen::ChunkParams{32u, 4242u, noise}, 1 * 32 + 31, -2 * 32 + static_cast<lpl::core::i32>(z));
        if (mine.raw() != theirs.raw())
        {
            seamless = false;
            break;
        }
    }
    check(seamless, "a builder chunk agrees with the world sampler on every shared cell");

    std::printf("    roads=%u townVoxels=%u caveVoxels=%u\n", built.roadCells, lpl::procgen::countSolidVoxels(raised),
                lpl::procgen::countSolidVoxels(cavern));
}

} // namespace

/**
 * @brief The structural generators: partitions, growth, grammars, towns,
 *        volume, chunks and playability.
 *
 * Same discipline as the terrain passes: reproducibility, sensitivity, and an
 * invariant per generator that would have to hold for it to have done its job.
 * The invariants here are mostly about STRUCTURE rather than shape — a Voronoi
 * partition must actually partition, an L-system must produce something
 * connected, a town's streets must join up, chunks must agree on their seams.
 * Those are the properties that a fold cannot see.
 */
int main()
{
    std::printf("== procedural structures ==\n\n");

    checkVoronoiPartitionsThePlane();
    checkAggregationGrowsOneBranchingStructure();
    checkLSystemsGrowAndDrawWhatTheyRewrite();
    checkSettlementsJoinUpAndStayOnSoundGround();
    checkExtrusionTurnsAPlanIntoAShape();
    checkChunksAgreeWithTheWorldAndAcrossSeams();
    checkPlayabilityGatesAcceptAndReject();
    checkTheBuilderReachesEveryGenerator();

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
