/**
 * @file test_procgen_structures.cpp
 * @brief The structural generators: partitions, growth, grammars, towns,
 *        volume, chunks and playability.
 *
 * Same discipline as the terrain passes: reproducibility, sensitivity, and an
 * invariant per generator that would have to hold for it to have done its job.
 * The invariants here are mostly about STRUCTURE rather than shape — a Voronoi
 * partition must actually partition, an L-system must produce something
 * connected, a town's streets must join up, chunks must agree on their seams.
 * Those are the properties that a fold cannot see.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

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

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== procedural structures ==\n\n");

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("-- voronoi --\n");
    {
        procgen::VoronoiParams params;
        params.width = 64u;
        params.depth = 64u;
        params.seed = 1337u;
        params.cellSize = 12u;

        const procgen::VoronoiDiagram diagram = procgen::computeVoronoi(params);
        check(diagram.regionCount > 1u, "the plane is partitioned into regions");

        // The defining property: every cell belongs to exactly one region, and
        // that region is the nearest site. A partition that misses cells is not
        // a partition.
        bool everyCellClaimed = true;
        for (core::u32 i = 0u; i < diagram.regions.cellCount(); ++i)
            if (diagram.regions[i] == procgen::kNoRegion)
                everyCellClaimed = false;
        check(everyCellClaimed, "every cell is claimed by a region");

        // The 3x3 search is only valid if it agrees with an exhaustive one.
        bool matchesBruteForce = true;
        for (core::u32 z = 0u; z < params.depth && matchesBruteForce; z += 7u)
        {
            for (core::u32 x = 0u; x < params.width; x += 7u)
            {
                math::Fixed32 best = math::Fixed32::max();
                core::u16 bestRegion = procgen::kNoRegion;
                for (core::u32 s = 0u; s < diagram.sites.size(); ++s)
                {
                    const math::Fixed32 dx = math::Fixed32::fromInt(static_cast<core::i32>(x)) - diagram.sites[s].x;
                    const math::Fixed32 dz = math::Fixed32::fromInt(static_cast<core::i32>(z)) - diagram.sites[s].z;
                    const math::Fixed32 d2 = dx * dx + dz * dz;
                    if (d2 < best)
                    {
                        best = d2;
                        bestRegion = static_cast<core::u16>(s);
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

        const procgen::VoronoiDiagram twin = procgen::computeVoronoi(params);
        check(procgen::foldRegionMap(diagram.regions) == procgen::foldRegionMap(twin.regions),
              "the partition is reproducible");

        procgen::VoronoiParams other = params;
        other.seed = 2024u;
        check(procgen::foldRegionMap(procgen::computeVoronoi(other).regions) != procgen::foldRegionMap(diagram.regions),
              "a different seed moves the sites");

        const procgen::Grid<core::u8> borders = procgen::regionBorders(diagram);
        core::u32 borderCells = 0u;
        for (core::u32 i = 0u; i < borders.cellCount(); ++i)
            borderCells += borders[i];
        check(borderCells > 0u && borderCells < borders.cellCount() / 2u,
              "borders exist and are a minority of the map");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- diffusion-limited aggregation --\n");
    {
        procgen::DlaParams params;
        params.width = 48u;
        params.depth = 48u;
        params.seed = 1337u;
        params.particles = 300u;

        procgen::DlaReport report;
        const procgen::DungeonMap cluster = procgen::generateDlaCave(params, &report);
        check(report.stuck > 0u, "particles aggregate onto the cluster");
        check(report.openCells > 1u, "the cluster grows beyond its seed");
        check(report.extent > 1u, "the cluster spreads out from the centre");

        // The property that makes it DLA rather than a blob: sticking on
        // contact means the whole cluster is one connected piece.
        check(procgen::isFullyConnected(cluster), "the aggregate is a single connected structure");

        // A dendrite is sparse: it reaches far while occupying little. A blob
        // filling its bounding box would fail this.
        const core::u32 boundingArea = (report.extent * 2u + 1u) * (report.extent * 2u + 1u);
        check(report.openCells < boundingArea / 2u, "the growth is branching, not a filled blob");

        procgen::DungeonMap twin = procgen::generateDlaCave(params);
        check(procgen::foldDungeon(cluster) == procgen::foldDungeon(twin), "aggregation is reproducible");

        std::printf("    stuck=%u abandoned=%u open=%u extent=%u\n", report.stuck, report.abandoned, report.openCells,
                    report.extent);
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- L-systems --\n");
    {
        const procgen::LSystemParams grammar = procgen::makeBranchingGrammar();
        const lpl::pmr::string expanded = procgen::expandLSystem(grammar);
        check(expanded.size() > grammar.axiom.size(), "rewriting grows the string");

        const lpl::pmr::string twin = procgen::expandLSystem(grammar);
        check(expanded == twin, "rewriting is deterministic");

        // Exponential growth must respect the cap, or a kernel heap decides how
        // this ends.
        procgen::LSystemParams runaway = grammar;
        runaway.iterations = 20u;
        runaway.maxLength = 4096u;
        check(procgen::expandLSystem(runaway).size() <= runaway.maxLength, "expansion respects its length cap");

        procgen::Grid<core::u8> canvas{64u, 64u, 0u};
        procgen::TurtleParams turtle;
        turtle.startX = 32u;
        turtle.startZ = 60u;
        turtle.startDirection = 12u; // heading "up" the grid
        turtle.stepLength = 3u;
        turtle.turnAmount = 2u;
        const core::u32 drawn = procgen::drawTurtle(expanded, turtle, canvas);
        check(drawn > 0u, "the turtle draws the structure");

        // A branching structure is connected: it grew from one stem.
        core::u32 filled = 0u;
        for (core::u32 i = 0u; i < canvas.cellCount(); ++i)
            filled += canvas[i];
        check(filled == drawn, "every drawn cell is accounted for");

        // Brackets must restore state: without a working stack the turtle
        // wanders off instead of returning to the branch point.
        procgen::Grid<core::u8> noBranches{64u, 64u, 0u};
        const core::u32 straight = procgen::drawTurtle(lpl::pmr::string{"FFFF"}, turtle, noBranches);
        check(straight > 0u && straight < drawn, "a straight run draws less than a branching one");

        const procgen::LSystemParams roads = procgen::makeRoadGrammar();
        check(procgen::expandLSystem(roads).size() > 0u, "the road grammar expands");

        // More rewriting must draw MORE, and it did the opposite: the turtle
        // walked off the canvas and every later symbol was silently clipped, so
        // 173 draw symbols painted 43 cells while 4156 painted 32. Growth running
        // backwards is invisible to a fold and to any "did it draw something"
        // check — only the trend shows it.
        procgen::LSystemParams growing = procgen::makeRoadGrammar();
        core::u32 previous = 0u;
        bool growsWithIterations = true;
        core::u32 drawnPerRound[4] = {0u, 0u, 0u, 0u};
        for (core::u32 round = 0u; round < 4u; ++round)
        {
            growing.iterations = 4u + round;
            procgen::Grid<core::u8> sheet{64u, 64u, 0u};
            procgen::TurtleParams walker;
            walker.startX = 32u;
            walker.startZ = 32u;
            walker.stepLength = 3u;
            drawnPerRound[round] = procgen::drawTurtle(procgen::expandLSystem(growing), walker, sheet);
            growsWithIterations = growsWithIterations && drawnPerRound[round] >= previous;
            previous = drawnPerRound[round];
        }
        check(growsWithIterations, "a longer expansion draws more, not less (the turtle stays on the canvas)");
        std::printf("    road grammar draws: %u %u %u %u cells at 4..7 rounds\n", drawnPerRound[0], drawnPerRound[1],
                    drawnPerRound[2], drawnPerRound[3]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- settlements --\n");
    {
        procgen::SettlementParams params;
        params.width = 96u;
        params.depth = 96u;
        params.seed = 1337u;

        lpl::pmr::vector<procgen::BuildingPlot> plots;
        procgen::SettlementReport report;
        const procgen::SettlementMap town = procgen::generateSettlement(params, &plots, &report);

        check(report.districts > 1u, "the town is divided into districts");
        check(report.roadCells > 0u, "streets are laid");
        check(report.plots > 0u, "buildings are placed");
        check(plots.size() == report.plots, "every plot is reported");

        // The property the whole ordering exists to guarantee.
        check(report.roadsConnected, "the street network is a single connected piece");

        // A plot that faces no street is a building nobody can enter.
        bool everyPlotReachable = true;
        for (core::u32 p = 0u; p < plots.size(); ++p)
        {
            const procgen::BuildingPlot &plot = plots[p];
            bool touchesStreet = false;
            for (core::i32 dz = -1; dz <= static_cast<core::i32>(plot.depth) && !touchesStreet; ++dz)
                for (core::i32 dx = -1; dx <= static_cast<core::i32>(plot.width); ++dx)
                {
                    const core::i32 cx = static_cast<core::i32>(plot.x) + dx;
                    const core::i32 cz = static_cast<core::i32>(plot.z) + dz;
                    if (!town.contains(cx, cz))
                        continue;
                    const procgen::SettlementCell cell =
                        town.at(static_cast<core::u32>(cx), static_cast<core::u32>(cz));
                    if (cell == procgen::SettlementCell::Road || cell == procgen::SettlementCell::Plaza)
                    {
                        touchesStreet = true;
                        break;
                    }
                }
            if (!touchesStreet)
                everyPlotReachable = false;
        }
        check(everyPlotReachable, "every building faces a street");

        check(procgen::foldSettlement(town) == procgen::foldSettlement(procgen::generateSettlement(params)),
              "the layout is reproducible");

        // On terrain, nothing may stand on a cliff or in water.
        procgen::NoiseParams noise;
        noise.seed = 1337u;
        noise.frequency = 0.05f;
        noise.amplitude = 16.0f;
        procgen::Heightfield ground = procgen::generateNoiseHeightfield(params.width, params.depth, noise);
        procgen::normalizeHeights(ground, math::Fixed32::fromInt(-2), math::Fixed32::fromInt(18));

        procgen::SettlementReport terrainReport;
        const procgen::SettlementMap onTerrain =
            procgen::generateSettlementOnTerrain(params, ground, nullptr, &terrainReport);
        check(terrainReport.blockedCells > 0u, "unbuildable ground is recognised");

        bool nothingOnBadGround = true;
        const math::Fixed32 minHeight = math::Fixed32::fromFloat(params.minHeight);
        const math::Fixed32 maxSlope = math::Fixed32::fromFloat(params.maxSlope);
        for (core::u32 z = 0u; z < onTerrain.depth(); ++z)
            for (core::u32 x = 0u; x < onTerrain.width(); ++x)
            {
                if (onTerrain.at(x, z) == procgen::SettlementCell::Empty ||
                    onTerrain.at(x, z) == procgen::SettlementCell::Blocked)
                    continue;
                if (ground.at(x, z) < minHeight || procgen::slopeAt(ground, x, z) > maxSlope)
                    nothingOnBadGround = false;
            }
        check(nothingOnBadGround, "nothing is built on a cliff or under water");

        std::printf("    districts=%u roads=%u plots=%u blocked=%u\n", report.districts, report.roadCells, report.plots,
                    terrainReport.blockedCells);
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- 2.5D extrusion --\n");
    {
        const procgen::TileSet tiles = procgen::makeTerrainTileSet();
        procgen::WfcParams wfc;
        wfc.width = 24u;
        wfc.depth = 24u;
        wfc.seed = 1337u;
        const procgen::WfcResult plan = procgen::solveWfc(tiles, wfc);
        check(plan.solved, "the 2D plan solves");

        // Water is a hole, each later tile is taller: the plan's gradient
        // becomes relief.
        lpl::pmr::vector<core::u8> heights;
        heights.push_back(0u); // water
        heights.push_back(1u); // sand
        heights.push_back(2u); // grass
        heights.push_back(3u); // forest
        heights.push_back(5u); // rock

        procgen::ExtrusionParams extrusion;
        extrusion.levels = 8u;
        const procgen::VoxelVolume volume = procgen::extrudeTilePlan(plan.tiles, heights, extrusion);

        check(!volume.empty(), "the plan extrudes to a volume");
        check(volume.width == wfc.width && volume.depth == wfc.depth, "the volume keeps the plan's footprint");
        check(volume.levels == extrusion.levels, "the volume has the requested height");

        const core::u32 solid = procgen::countSolidVoxels(volume);
        check(solid > 0u && solid < volume.voxelCount(), "the volume is neither empty nor full");

        // A surface smaller than the solid count is what proves the volume has
        // an interior — that is, that it is a shape and not a shell.
        const core::u32 surface = procgen::countSurfaceVoxels(volume);
        check(surface > 0u && surface <= solid, "the volume has a coherent surface");

        check(procgen::foldVolume(volume) ==
                  procgen::foldVolume(procgen::extrudeTilePlan(plan.tiles, heights, extrusion)),
              "extrusion is reproducible");

        // Hollow mode must cost less than solid mode.
        procgen::ExtrusionParams hollow = extrusion;
        hollow.solidBelow = false;
        check(procgen::countSolidVoxels(procgen::extrudeTilePlan(plan.tiles, heights, hollow)) < solid,
              "surface-only extrusion writes fewer voxels");

        std::printf("    solid=%u surface=%u of %u voxels\n", solid, surface, volume.voxelCount());
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- chunked worlds --\n");
    {
        procgen::ChunkParams params;
        params.size = 32u;
        params.worldSeed = 1337u;
        params.noise.frequency = 0.03f;
        params.noise.amplitude = 20.0f;

        const procgen::Heightfield origin = procgen::generateChunkTerrain(params, procgen::ChunkCoord{0, 0});
        check(origin.width() == params.size, "a chunk is generated at its requested size");

        // The whole scheme rests on this: a chunk depends on nothing but its
        // coordinates, so it is identical however and whenever it is built.
        const procgen::Heightfield again = procgen::generateChunkTerrain(params, procgen::ChunkCoord{0, 0});
        check(procgen::foldHeightfield(origin) == procgen::foldHeightfield(again), "a chunk is reproducible");

        const procgen::Heightfield east = procgen::generateChunkTerrain(params, procgen::ChunkCoord{1, 0});
        check(procgen::foldHeightfield(east) != procgen::foldHeightfield(origin), "neighbouring chunks differ");

        // Seams: the shared edge must agree, in every direction, including
        // across the origin where the coordinates change sign.
        check(procgen::countSeamMismatches(params, procgen::ChunkCoord{0, 0}, procgen::ChunkCoord{1, 0}) == 0u,
              "the eastern seam matches");
        check(procgen::countSeamMismatches(params, procgen::ChunkCoord{0, 0}, procgen::ChunkCoord{0, 1}) == 0u,
              "the southern seam matches");
        check(procgen::countSeamMismatches(params, procgen::ChunkCoord{0, 0}, procgen::ChunkCoord{-1, 0}) == 0u,
              "the western seam matches");
        check(procgen::countSeamMismatches(params, procgen::ChunkCoord{-1, -1}, procgen::ChunkCoord{0, -1}) == 0u,
              "seams hold at negative coordinates");

        // The world function and the chunk must agree cell for cell.
        bool agreesWithWorld = true;
        for (core::u32 i = 0u; i < params.size; ++i)
            if (origin.at(i, 0u).raw() != procgen::sampleWorldHeight(params, static_cast<core::i32>(i), 0).raw())
                agreesWithWorld = false;
        check(agreesWithWorld, "a chunk equals the world function over its own cells");

        check(procgen::chunkSeed(params, procgen::ChunkCoord{1, 0}) !=
                  procgen::chunkSeed(params, procgen::ChunkCoord{0, 1}),
              "transposed chunk coordinates get different seeds");

        // Constraint-solved chunks need their borders pinned explicitly.
        const procgen::TileSet tiles = procgen::makeTerrainTileSet();
        procgen::WfcParams wfc;
        wfc.width = 16u;
        wfc.depth = 16u;
        wfc.seed = procgen::chunkSeed(params, procgen::ChunkCoord{0, 0});
        const procgen::WfcResult first = procgen::solveWfc(tiles, wfc);
        check(first.solved, "the first chunk solves");

        const procgen::TileGrid preset = procgen::borderConstraintsFrom(16u, first.tiles, 1u);
        procgen::WfcParams nextParams = wfc;
        nextParams.seed = procgen::chunkSeed(params, procgen::ChunkCoord{1, 0});
        const procgen::WfcResult second = procgen::solveWfc(tiles, nextParams, &preset);
        check(second.solved, "the neighbouring chunk solves against the seam");

        bool seamAgrees = true;
        for (core::u32 i = 0u; i < 16u; ++i)
            if (second.tiles.at(0u, i) != first.tiles.at(15u, i))
                seamAgrees = false;
        check(seamAgrees, "the WFC seam is continuous across chunks");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- playability gates --\n");
    {
        procgen::BspDungeonParams bsp;
        bsp.width = 64u;
        bsp.depth = 64u;
        bsp.seed = 1337u;
        const procgen::DungeonMap level = procgen::generateBspDungeon(bsp);

        core::u32 startX = 0u, startZ = 0u, goalX = 0u, goalZ = 0u;
        check(procgen::findFarthestPair(level, startX, startZ, goalX, goalZ),
              "entrance and exit are found automatically");

        const procgen::LevelQuality quality = procgen::evaluateLevel(level, startX, startZ, goalX, goalZ);
        check(quality.goalReachable, "the exit is reachable from the entrance");
        check(quality.fullyConnected, "no part of the level is walled off");
        check(quality.pathLength > 0u, "the exit is not on top of the entrance");
        check(quality.reachableCells == quality.walkableCells, "every walkable cell is reached");
        check(quality.junctions > 0u, "the level branches rather than being one corridor");

        check(procgen::passesGate(quality, procgen::GateCriteria{}), "a BSP dungeon passes the default gate");

        // The gate has to reject as well as accept, or it is decoration.
        procgen::DungeonMap sealed{32u, 32u, procgen::DungeonCell::Wall};
        sealed.at(1u, 1u) = procgen::DungeonCell::Floor;
        sealed.at(30u, 30u) = procgen::DungeonCell::Floor;
        const procgen::LevelQuality bad = procgen::evaluateLevel(sealed, 1u, 1u, 30u, 30u);
        check(!bad.goalReachable, "a sealed-off exit is detected");
        check(!procgen::passesGate(bad, procgen::GateCriteria{}), "an unplayable level is rejected");

        // A distance map is a flow field: descending it must always approach.
        const procgen::DistanceMap distances = procgen::computeDistanceMap(level, startX, startZ);
        bool monotone = true;
        for (core::u32 z = 0u; z < level.depth() && monotone; ++z)
            for (core::u32 x = 0u; x < level.width(); ++x)
            {
                const core::u32 here = distances.at(x, z);
                if (here == procgen::kUnreachable || here == 0u)
                    continue;
                bool hasCloserNeighbour = false;
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor4Z[n];
                    if (distances.contains(nx, nz) &&
                        distances.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) == here - 1u)
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

    // ── The builder reaches every generator ─────────────────────────────────
    //
    // Each of these three had a working implementation, a passing test, and no
    // caller: nothing a world was ever built from went through them. A generator
    // reachable only from its own test is not part of the module, so what this
    // block pins is the wiring rather than the algorithm.
    {
        std::printf("\n-- builder wiring --\n");

        // Roads: an L-system steered by a tensor field, vetoed by the ground.
        procgen::SettlementParams town;
        town.districtSize = 12u;

        procgen::WorldBuilder city{4242u};
        city.terrain(64u, 64u).normalize(-8.0f, 16.0f).biomes().regions(12u).settlement(town).roads();
        const procgen::BuiltWorldStats built = city.bakeGrids();
        check(built.roadCells > 0u, "the builder grows a road network");
        check(city.roadMap().width() == 64u && city.roadMap().depth() == 64u, "the road mask covers the world");

        // Local constraints actually bind: no road on ground it refused.
        procgen::RoadParams strict;
        strict.maxSlope = 0.05f;
        procgen::WorldBuilder steep{4242u};
        steep.terrain(64u, 64u).normalize(-8.0f, 16.0f).biomes().regions(12u).settlement(town).roads(strict);
        (void) steep.bakeGrids();
        bool slopeRespected = true;
        for (core::u32 z = 0u; z < 64u && slopeRespected; ++z)
            for (core::u32 x = 0u; x < 64u; ++x)
                if (steep.roadMap().at(x, z) != 0u &&
                    procgen::slopeAt(steep.heightfield(), x, z) > math::Fixed32::fromFloat(strict.maxSlope))
                {
                    slopeRespected = false;
                    break;
                }
        check(slopeRespected, "the ground vetoes roads the grammar proposed on slopes it refuses");

        // Extrusion: the plan gains height, and the town is taller than its streets.
        procgen::ExtrusionParams volume;
        volume.levels = 8u;
        city.extrudeTown(volume);
        const procgen::VoxelVolume &raised = city.townVolume();
        check(!raised.empty() && raised.levels == 8u, "the builder raises the settlement into a volume");
        check(procgen::countSolidVoxels(raised) > 0u, "the raised town has solid voxels");

        core::u32 tallestPlot = 0u;
        core::u32 tallestRoad = 0u;
        for (core::u32 z = 0u; z < raised.depth; ++z)
            for (core::u32 x = 0u; x < raised.width; ++x)
            {
                core::u32 column = 0u;
                for (core::u32 y = 0u; y < raised.levels; ++y)
                    if (raised.at(x, y, z) != 0u)
                        column = y + 1u;
                const procgen::SettlementCell kind = city.settlementMap().at(x, z);
                if (kind == procgen::SettlementCell::Plot && column > tallestPlot)
                    tallestPlot = column;
                if (kind == procgen::SettlementCell::Road && column > tallestRoad)
                    tallestRoad = column;
            }
        check(tallestPlot > tallestRoad, "buildings stand above their streets");

        // The underground is a volume too: rock is solid, floor is open.
        procgen::CaveParams caves;
        caves.width = 48u;
        caves.depth = 48u;
        procgen::WorldBuilder below{4242u};
        below.terrain(48u, 48u).caves(caves).extrudeUnderground(volume);
        const procgen::VoxelVolume &cavern = below.undergroundVolume();
        check(!cavern.empty(), "the builder raises the underground into a volume");
        bool floorIsOpen = true;
        for (core::u32 z = 0u; z < cavern.depth && floorIsOpen; ++z)
            for (core::u32 x = 0u; x < cavern.width; ++x)
                if (below.dungeonMap().at(x, z) != procgen::DungeonCell::Wall && cavern.at(x, 0u, z) != 0u)
                {
                    floorIsOpen = false;
                    break;
                }
        check(floorIsOpen, "open cells stay open through the extrusion");

        // Chunking: the same seed describes one endless world, and a chunk of it
        // agrees with the whole map about the ground they share.
        procgen::NoiseParams noise;
        noise.seed = 4242u;
        noise.frequency = 0.05f;
        procgen::WorldBuilder tile{4242u};
        tile.chunk(procgen::ChunkCoord{1, -2}).terrain(32u, 32u, noise);
        procgen::WorldBuilder sameTile{4242u};
        sameTile.chunk(procgen::ChunkCoord{1, -2}).terrain(32u, 32u, noise);
        procgen::WorldBuilder otherTile{4242u};
        otherTile.chunk(procgen::ChunkCoord{2, -2}).terrain(32u, 32u, noise);
        check(procgen::foldHeightfield(tile.heightfield()) == procgen::foldHeightfield(sameTile.heightfield()),
              "the same chunk of the same world is the same terrain");
        check(procgen::foldHeightfield(tile.heightfield()) != procgen::foldHeightfield(otherTile.heightfield()),
              "a different chunk is different terrain");

        bool seamless = true;
        for (core::u32 z = 0u; z < 32u; ++z)
        {
            // The right edge of (1,-2) and the left edge of (2,-2) are the same
            // world cells, asked twice.
            const math::Fixed32 mine = tile.heightfield().at(31u, z);
            const math::Fixed32 theirs = procgen::sampleWorldHeight(procgen::ChunkParams{32u, 4242u, noise},
                                                                    1 * 32 + 31, -2 * 32 + static_cast<core::i32>(z));
            if (mine.raw() != theirs.raw())
            {
                seamless = false;
                break;
            }
        }
        check(seamless, "a builder chunk agrees with the world sampler on every shared cell");

        std::printf("    roads=%u townVoxels=%u caveVoxels=%u\n", built.roadCells, procgen::countSolidVoxels(raised),
                    procgen::countSolidVoxels(cavern));
    }

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
