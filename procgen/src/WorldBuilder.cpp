/**
 * @file WorldBuilder.cpp
 * @brief Implementation of the fluent world composition API.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/WorldBuilder.hpp>

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/Random.hpp>

namespace lpl::procgen {

namespace {

using FVec3 = math::Vec3<math::Fixed32>;

/// Default terrain when a caller says only "give me a world".
constexpr core::u32 kDefaultWidth = 64u;
constexpr core::u32 kDefaultDepth = 64u;

} // namespace

WorldBuilder &WorldBuilder::cellSize(core::f32 size) noexcept
{
    _cellSize = size > 0.0f ? size : 1.0f;
    return *this;
}

// ── Terrain ─────────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::terrain(core::u32 width, core::u32 depth)
{
    NoiseParams noise;
    noise.seed = _seed;
    return terrain(width, depth, noise);
}

WorldBuilder &WorldBuilder::chunk(ChunkCoord coord) noexcept
{
    _chunk = coord;
    _isChunk = true;
    return *this;
}

WorldBuilder &WorldBuilder::terrain(core::u32 width, core::u32 depth, const NoiseParams &noise)
{
    NoiseParams seeded = noise;
    // A caller that leaves the noise seed at its default still gets a world
    // that follows the builder's seed — otherwise every world would be the same
    // one until someone noticed the second seed.
    if (seeded.seed == NoiseParams{}.seed)
        seeded.seed = _seed;

    if (_isChunk)
    {
        // A chunk samples the SAME function as a whole map, only at absolute
        // world coordinates. That is what makes the seam free: the cells two
        // neighbours share are one query with one answer, so there is nothing to
        // stitch and nothing that can disagree.
        ChunkParams params;
        params.size = width;
        params.worldSeed = seeded.seed;
        params.noise = seeded;
        _height = generateChunkTerrain(params, _chunk);
        // A non-square chunk is not a chunk: the coordinate system would stop
        // being a grid. Fall back to the plain path rather than lie about it.
        if (width != depth)
            _height = generateNoiseHeightfield(width, depth, seeded);
    }
    else
    {
        _height = generateNoiseHeightfield(width, depth, seeded);
    }

    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

WorldBuilder &WorldBuilder::addLayer(const NoiseParams &noise)
{
    ensureTerrain();
    NoiseParams seeded = noise;
    if (seeded.seed == NoiseParams{}.seed)
        seeded.seed = math::deriveStream(_seed, 0x1A4E4u).state();
    addNoiseLayer(_height, seeded);
    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

WorldBuilder &WorldBuilder::normalize(core::f32 low, core::f32 high)
{
    ensureTerrain();
    normalizeHeights(_height, math::Fixed32::fromFloat(low), math::Fixed32::fromFloat(high));
    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

WorldBuilder &WorldBuilder::terraces(core::u32 steps)
{
    ensureTerrain();
    terrace(_height, steps);
    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

// ── Erosion ─────────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::erode()
{
    erodeThermal(ThermalErosionParams{});
    return erodeHydraulic(HydraulicErosionParams{});
}

WorldBuilder &WorldBuilder::erodeThermal(const ThermalErosionParams &params)
{
    ensureTerrain();
    (void) thermalErode(_height, params);
    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

WorldBuilder &WorldBuilder::erodeHydraulic(const HydraulicErosionParams &params)
{
    ensureTerrain();
    (void) hydraulicErode(_height, params);
    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

// ── Water ───────────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::rivers() { return rivers(RiverParams{}); }

WorldBuilder &WorldBuilder::rivers(const RiverParams &params)
{
    ensureTerrain();
    ensureDrainage();
    _riverCells = carveRivers(_height, _drainage, params);

    // Carving changed the terrain, so the drainage that produced it is stale.
    // Recomputing here rather than lazily is what lets moisture reflect the
    // valleys the rivers just cut, instead of the slopes that predated them.
    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    ensureDrainage();

    // Keep the mask, not just the count. Where the water runs is what a
    // phreatophyte scatter rule and a road router both need to know, and
    // recomputing it from a stale drainage would answer about the wrong terrain.
    _rivers = riverMask(_drainage, params.density);
    return *this;
}

WorldBuilder &WorldBuilder::seaLevel(core::f32 level)
{
    ensureTerrain();
    clampToSeaLevel(_height, math::Fixed32::fromFloat(level));
    _biomeParams.seaLevel = level;
    _drainageReady = false;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

// ── Climate ─────────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::biomes() { return biomes(_biomeParams); }

WorldBuilder &WorldBuilder::biomes(const BiomeParams &params)
{
    _biomeParams = params;
    ensureClimate();

    // The lakes come free with the drainage: Priority-Flood already raised every
    // basin to its spill level, so the difference between the filled surface and
    // the terrain is standing water. Reporting only a raised-cell count and
    // classifying those cells as meadow was throwing the answer away.
    _lakes = lakeMask(_drainage, _height);
    _lakeCells = 0u;
    for (core::u32 i = 0u; i < _lakes.cellCount(); ++i)
        _lakeCells += _lakes[i] != 0u ? 1u : 0u;

    _biomes = classifyBiomes(_height, _climate, _biomeParams, _lakes.empty() ? nullptr : &_lakes);
    _biomesReady = true;
    return *this;
}

WorldBuilder &WorldBuilder::climateAxes(const ClimateParams &params)
{
    _climateParams = params;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

WorldBuilder &WorldBuilder::climate(const MoistureParams &params)
{
    _moistureParams = params;
    _moistureReady = false;
    _climateReady = false;
    _biomesReady = false;
    return *this;
}

// ── Population ──────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::scatterInBiome(BiomeId biome, core::f32 density)
{
    ScatterRule rule;
    rule.biome = biome;
    rule.density = density;
    return scatter(rule);
}

WorldBuilder &WorldBuilder::scatter(const ScatterRule &rule)
{
    _scatterRules.push_back(rule);
    return *this;
}

// ── Underground ─────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::dungeon(const BspDungeonParams &params)
{
    BspDungeonParams seeded = params;
    if (seeded.seed == BspDungeonParams{}.seed)
        seeded.seed = math::deriveStream(_seed, 0xD065u).state();
    _dungeon = generateBspDungeon(seeded);
    const DungeonReport report = connectRegions(_dungeon, 1u);
    _dungeonFloor = report.floorCells;
    _dungeonConnected = report.connected;
    return *this;
}

WorldBuilder &WorldBuilder::caves(const CaveParams &params)
{
    CaveParams seeded = params;
    if (seeded.seed == CaveParams{}.seed)
        seeded.seed = math::deriveStream(_seed, 0xCA4E5u).state();
    _dungeon = generateCellularCave(seeded);
    _dungeonFloor = 0u;
    for (core::u32 i = 0u; i < _dungeon.cellCount(); ++i)
        if (_dungeon[i] != DungeonCell::Wall)
            ++_dungeonFloor;
    _dungeonConnected = isFullyConnected(_dungeon);
    return *this;
}

WorldBuilder &WorldBuilder::dlaCaves(const DlaParams &params)
{
    DlaParams seeded = params;
    if (seeded.seed == DlaParams{}.seed)
        seeded.seed = math::deriveStream(_seed, 0xD1AAu).state();
    DlaReport report;
    _dungeon = generateDlaCave(seeded, &report);
    _dungeonFloor = report.openCells;
    _dungeonConnected = isFullyConnected(_dungeon);
    return *this;
}

WorldBuilder &WorldBuilder::regions(core::u32 regionSize)
{
    VoronoiParams params;
    params.cellSize = regionSize == 0u ? 12u : regionSize;
    return regions(params);
}

WorldBuilder &WorldBuilder::regions(const VoronoiParams &params)
{
    ensureTerrain();
    VoronoiParams seeded = params;
    if (seeded.seed == VoronoiParams{}.seed)
        seeded.seed = math::deriveStream(_seed, 0x4E610u).state();
    // The partition covers the terrain, so its extent is taken from the
    // heightfield rather than trusted from the caller.
    seeded.width = _height.width();
    seeded.depth = _height.depth();
    _regions = computeVoronoi(seeded);
    return *this;
}

WorldBuilder &WorldBuilder::tiles(const TileSet &tileSet, const WfcParams &params)
{
    ensureTerrain();
    WfcParams seeded = params;
    if (seeded.seed == WfcParams{}.seed)
        seeded.seed = math::deriveStream(_seed, 0x7F1Eu).state();
    seeded.width = _height.width();
    seeded.depth = _height.depth();

    // Pin what the world has already decided. A tile solve with no preset is a
    // second, unrelated world of the same dimensions; with one it is a decoration
    // of the world that exists.
    TileGrid preset{seeded.width, seeded.depth, kNoTile};
    bool anyPinned = false;
    if (_biomesReady && _biomes.width() == seeded.width && _biomes.depth() == seeded.depth && tileSet.tileCount >= 5u)
    {
        for (core::u32 i = 0u; i < preset.cellCount(); ++i)
        {
            switch (_biomes[i])
            {
            case BiomeId::Ocean:
                preset[i] = static_cast<core::u8>(TerrainTile::Water);
                anyPinned = true;
                break;
            case BiomeId::Beach:
                preset[i] = static_cast<core::u8>(TerrainTile::Sand);
                anyPinned = true;
                break;
            case BiomeId::Lake:
                preset[i] = static_cast<core::u8>(TerrainTile::Water);
                anyPinned = true;
                break;
            case BiomeId::Rock:
            case BiomeId::Snow:
                preset[i] = static_cast<core::u8>(TerrainTile::Rock);
                anyPinned = true;
                break;
            default: break;
            }
        }

        // A preset can contradict the very rules the solve must satisfy: a rock
        // summit whose neighbour is a lake pins two tiles the tile set forbids
        // from touching, and no arrangement exists. The solver cannot report that
        // usefully — it exhausts its attempts and hands back a grid that violates
        // the rules, silently.
        //
        // So the preset is made SATISFIABLE before it is handed over: where two
        // pinned neighbours may not touch, the later one in scan order is
        // released, and the solver fills it with whatever transition the tile set
        // provides. Releasing rather than refusing is the right call here because
        // the pins are a preference (this cell looks like rock) while the
        // adjacency rules are a constraint (rock never abuts water).
        for (core::u32 z = 0u; z < preset.depth(); ++z)
        {
            for (core::u32 x = 0u; x < preset.width(); ++x)
            {
                const core::u8 here = preset.at(x, z);
                if (here == kNoTile)
                    continue;
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                    if (!preset.contains(nx, nz))
                        continue;
                    const core::u8 other = preset.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                    if (other == kNoTile)
                        continue;
                    const core::u64 mask = tileSet.allowed[here * 4u + n];
                    if ((mask & (core::u64{1} << other)) == 0u)
                    {
                        preset.at(x, z) = kNoTile;
                        break;
                    }
                }
            }
        }

        anyPinned = false;
        for (core::u32 i = 0u; i < preset.cellCount(); ++i)
            if (preset[i] != kNoTile)
                anyPinned = true;
    }

    WfcResult result = solveWfc(tileSet, seeded, anyPinned ? &preset : nullptr);

    // A solve that did not succeed hands back a grid anyway, and that grid
    // violates the very rules the pass exists to satisfy — measured at seven
    // violations on a 48x48 world before this check existed, with every cell
    // filled so nothing looked wrong. Keeping it was the bug: a caller has no way
    // to tell a solved arrangement from a failed one by looking at it.
    //
    // A preset can be locally consistent and still globally unsatisfiable, so
    // when the pinned solve fails, fall back to an unpinned one. That loses the
    // agreement with the biome map, which is a real loss — and it is strictly
    // better than an arrangement that breaks the adjacency rules, because the
    // rules are what the tiles MEAN.
    if (!result.solved && anyPinned)
        result = solveWfc(tileSet, seeded, nullptr);

    _tilesSolved = result.solved;
    _tiles = result.tiles;
    return *this;
}

WorldBuilder &WorldBuilder::validate(const GateCriteria &criteria)
{
    // A layered system fills _caveSystem and leaves _dungeon empty, so judging the
    // flat map would report zero open cells and fail a world that is perfectly
    // navigable. The builder knows which underground it generated; asking the right
    // one is its business rather than the caller's, or every caller repeats the test
    // and one of them gets it wrong.
    if (_dungeon.empty() && _caveSystem.layerCount != 0u)
    {
        _quality = evaluateCaveSystem(_caveSystem);
        _gatePassed = passesGate(_quality, criteria);
        return *this;
    }

    if (_dungeon.empty())
    {
        // Nothing to judge. Saying "passed" would be a claim about a level that was
        // never generated, so the verdict stays false and the measurements empty.
        _quality = LevelQuality{};
        _gatePassed = false;
        return *this;
    }

    core::u32 startX = 0u;
    core::u32 startZ = 0u;
    core::u32 goalX = 0u;
    core::u32 goalZ = 0u;
    if (!findFarthestPair(_dungeon, startX, startZ, goalX, goalZ))
    {
        _quality = LevelQuality{};
        _gatePassed = false;
        return *this;
    }

    _quality = evaluateLevel(_dungeon, startX, startZ, goalX, goalZ);
    _gatePassed = passesGate(_quality, criteria);
    return *this;
}

WorldBuilder &WorldBuilder::settlement(const SettlementParams &params)
{
    ensureTerrain();
    SettlementParams seeded = params;
    if (seeded.seed == SettlementParams{}.seed)
        seeded.seed = math::deriveStream(_seed, 0x70A4u).state();
    // A settlement must match the terrain it stands on, so its extent is taken
    // from the heightfield rather than trusted from the caller.
    seeded.width = _height.width();
    seeded.depth = _height.depth();

    SettlementReport report;
    _plots.clear();
    _settlement = generateSettlementOnTerrain(seeded, _height, &_plots, &report);
    _settlementPlots = report.plots;
    _settlementConnected = report.roadsConnected;
    return *this;
}

// ── Roads ───────────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::roads() { return roads(RoadParams{}); }

WorldBuilder &WorldBuilder::roads(const RoadParams &params)
{
    ensureTerrain();
    if (_height.empty())
        return *this;

    const core::u32 width = _height.width();
    const core::u32 depth = _height.depth();
    math::Random random = math::deriveStream(params.seed != 0u ? params.seed : _seed, 0x20AD5u);

    // ── Global goals: where roads want to run ────────────────────────────────
    //
    // Anchored on the districts the settlement already laid out when there is
    // one. A road network exists to serve something; inventing its own centres
    // while a town sits next to it is how a generator ends up drawing two
    // unrelated worlds on the same grid.
    lpl::pmr::vector<FieldRegion> regions;
    if (_regions.regionCount != 0u && !_regions.sites.empty())
    {
        for (core::u32 i = 0u; i < _regions.sites.size(); ++i)
        {
            FieldRegion region;
            // The first districts get the radial web of an old centre; the rest
            // get an imposed bearing, which is the planned grid at the edges.
            region.pattern = i < params.gridDistricts ? FieldPattern::Grid : FieldPattern::Radial;
            region.centerX = static_cast<core::u32>(_regions.sites[i].x.toInt());
            region.centerZ = static_cast<core::u32>(_regions.sites[i].z.toInt());
            region.bearing = static_cast<core::u32>(random.below(16u));
            region.strength = 1.0f;
            region.falloff = 0.04f;
            regions.push_back(region);
        }
    }
    else
    {
        FieldRegion centre;
        centre.pattern = FieldPattern::Radial;
        centre.centerX = width / 2u;
        centre.centerZ = depth / 2u;
        regions.push_back(centre);
    }

    const HeadingField field = bakeHeadingField(width, depth, regions);

    _roads = Grid<core::u8>{width, depth, 0u};

    // ── Arterials first: routed, not grown ───────────────────────────────────
    //
    // A grammar has no destination, so the network it produces connects whatever
    // it happens to reach. The trunk roads are therefore routed between the
    // places that exist — district centres — paying for slope and water, and the
    // grammar then adds texture over a skeleton that already goes somewhere.
    if (params.arterials)
    {
        lpl::pmr::vector<core::u32> places;
        if (_regions.regionCount != 0u && !_regions.sites.empty())
        {
            for (const VoronoiSite &site : _regions.sites)
            {
                const core::i32 sx = site.x.toInt();
                const core::i32 sz = site.z.toInt();
                if (_height.contains(sx, sz))
                    places.push_back(_height.index(static_cast<core::u32>(sx), static_cast<core::u32>(sz)));
            }
        }

        if (places.size() >= 2u)
        {
            RoutingParams routing;
            routing.waterLevel = params.minHeight;
            (void) connectPlaces(_height, places, routing, _roads);
        }
    }

    LSystemParams grammar = makeRoadGrammar();
    grammar.iterations = params.iterations;
    grammar.seed = random.next();

    TurtleParams turtle;
    turtle.startX = width / 2u;
    turtle.startZ = depth / 2u;
    turtle.stepLength = params.stepLength;

    (void) drawTurtleInField(expandLSystem(grammar), turtle, field, params.conform, _roads);

    // ── Local constraints: the ground has the last word ──────────────────────
    //
    // Erased after drawing rather than avoided during, because a turtle that
    // refuses a step has to decide what to do instead, and every answer to that
    // question is a heuristic. Erasing is honest: the grammar proposed, the
    // terrain disposed.
    const math::Fixed32 maxSlope = math::Fixed32::fromFloat(params.maxSlope);
    const math::Fixed32 minHeight = math::Fixed32::fromFloat(params.minHeight);
    _roadCells = 0u;
    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            core::u8 &cell = _roads.at(x, z);
            if (cell == 0u)
                continue;
            if (_height.at(x, z) < minHeight || slopeAt(_height, x, z) > maxSlope)
            {
                cell = 0u;
                continue;
            }
            ++_roadCells;
        }
    }
    return *this;
}

// ── Volume ──────────────────────────────────────────────────────────────────

WorldBuilder &WorldBuilder::extrudeTown(const ExtrusionParams &params)
{
    _townVolume = VoxelVolume{};
    _townVoxels = 0u;
    if (_settlement.empty())
        return *this;

    // One material per cell kind, and a height rule per material. Plots are
    // taller than roads by construction rather than by a special case, which is
    // what makes the plan and the volume one relationship instead of two.
    Grid<core::u8> plan{_settlement.width(), _settlement.depth(), 0u};
    for (core::u32 i = 0u; i < _settlement.cellCount(); ++i)
        plan[i] = static_cast<core::u8>(_settlement[i]);

    lpl::pmr::vector<core::u8> heights(static_cast<core::usize>(SettlementCell::Blocked) + 1u, core::u8{0});
    heights[static_cast<core::usize>(SettlementCell::Empty)] = 0u;
    heights[static_cast<core::usize>(SettlementCell::Road)] = 1u;
    heights[static_cast<core::usize>(SettlementCell::Plaza)] = 1u;
    heights[static_cast<core::usize>(SettlementCell::Plot)] = 3u;
    heights[static_cast<core::usize>(SettlementCell::Blocked)] = 0u;

    _townVolume = extrudeTilePlan(plan, heights, params);

    // A bigger footprint is a taller building: the plan alone cannot say that,
    // because every plot cell looks the same to it. The footprints can.
    //
    // Only over cells the map still calls Plot, though. A footprint is a
    // rectangle and the map is the truth: parts of that rectangle came back as
    // road, plaza or unbuildable, and raising the rectangle wholesale puts a
    // six-storey block on top of the street that serves it. Measured before the
    // guard: the tallest ROAD column was as tall as the tallest building.
    const core::u8 plotMaterial = static_cast<core::u8>(static_cast<core::u32>(SettlementCell::Plot) + 1u);
    for (const BuildingPlot &plot : _plots)
    {
        const core::u32 storeys = 2u + (plot.width * plot.depth) / 4u;
        const core::u32 top = storeys < _townVolume.levels ? storeys : _townVolume.levels;
        for (core::u32 z = plot.z; z < plot.z + plot.depth && z < _townVolume.depth; ++z)
            for (core::u32 x = plot.x; x < plot.x + plot.width && x < _townVolume.width; ++x)
            {
                if (_settlement.at(x, z) != SettlementCell::Plot)
                    continue;
                for (core::u32 y = params.baseLevel; y < top; ++y)
                    _townVolume.at(x, y, z) = plotMaterial;
            }
    }

    _townVoxels = countSolidVoxels(_townVolume);
    return *this;
}

WorldBuilder &WorldBuilder::extrudeUnderground(const ExtrusionParams &params)
{
    _undergroundVolume = VoxelVolume{};
    _undergroundVoxels = 0u;
    if (_dungeon.empty())
        return *this;

    // Rock is solid to the ceiling, floor is open. Read as a plan the two are a
    // picture; read as a volume they are a cave with walls you can stand next to.
    Grid<core::u8> plan{_dungeon.width(), _dungeon.depth(), 0u};
    for (core::u32 i = 0u; i < _dungeon.cellCount(); ++i)
        plan[i] = _dungeon[i] == DungeonCell::Wall ? 1u : 0u;

    lpl::pmr::vector<core::u8> heights(2u, core::u8{0});
    heights[0] = 0u;                                   // open cell: nothing above it
    heights[1] = static_cast<core::u8>(params.levels); // rock: full height

    _undergroundVolume = extrudeTilePlan(plan, heights, params);
    _undergroundVoxels = countSolidVoxels(_undergroundVolume);
    return *this;
}

WorldBuilder &WorldBuilder::caveSystem(const CaveSystemParams &params)
{
    ensureBiomes();

    CaveSystemParams local = params;
    local.width = _height.width();
    local.depth = _height.depth();
    if (local.seed == CaveSystemParams{}.seed)
        local.seed = math::deriveStream(_seed, 0xCA7E5u).state();

    _caveSystem = generateCaveSystem(local, _height, _biomes.empty() ? nullptr : &_biomes);
    _undergroundVolume = caveVolume(_caveSystem, local, 1u);
    _undergroundVoxels = countSolidVoxels(_undergroundVolume);
    return *this;
}

WorldBuilder &WorldBuilder::buildings(const BuildingGrammarParams &params)
{
    if (_settlement.empty())
        return *this;

    const core::u32 seed = params.seed != 0u ? params.seed : _seed ^ 0xB1D6E5u;
    // Tall enough for the tallest building the parameters can produce, so a
    // storey is never silently clipped: a roof that vanished because the volume
    // was one level short is a bug that looks like a design choice.
    const core::u32 levels =
        params.baseHeight + params.maxFloors * (params.floorHeight == 0u ? 1u : params.floorHeight) + params.roofHeight;

    _townVolume = buildTown(_settlement, _plots, params, seed, levels);
    _townVoxels = countSolidVoxels(_townVolume);
    return *this;
}

WorldBuilder &WorldBuilder::roadside(const char *grammarText, core::u32 levels)
{
    _roadsideModules = 0u;
    _roadsideVolume = VoxelVolume{};
    if (_roads.empty())
        return *this;

    SequenceGrammar grammar;
    if (!parseSequenceGrammar(grammarText, grammar))
        return *this; // A refused grammar decorates nothing; it does not half-decorate.

    _roadsideVolume = decoratePath(_roads, grammar, _seed ^ 0xF3CE1u, levels, _roadsideModules);
    return *this;
}

// ── Scatter ─────────────────────────────────────────────────────────────────

lpl::pmr::vector<core::u32> WorldBuilder::eligibleCells(const ScatterRule &rule) const
{
    lpl::pmr::vector<core::u32> cells;
    const math::Fixed32 maxSlope = math::Fixed32::fromFloat(rule.maxSlope);
    const math::Fixed32 minMoisture = math::Fixed32::fromFloat(rule.minMoisture);
    const math::Fixed32 maxMoisture = math::Fixed32::fromFloat(rule.maxMoisture);
    const bool haveMoisture = _moisture.width() == _height.width() && _moisture.depth() == _height.depth();

    // ── Endemism: which regions this species was granted ────────────────────
    //
    // Drawn once per rule from the world seed and the rule's tag, so the answer
    // is a property of the world rather than of when the rule happened to run.
    const bool haveRegions = _regions.regionCount > 0u && _regions.regions.width() == _height.width() &&
                             _regions.regions.depth() == _height.depth();
    const bool endemic = rule.endemicShare < 1.0f && haveRegions;
    lpl::pmr::vector<core::u8> allowedRegion;
    if (endemic)
    {
        allowedRegion.resize(_regions.regionCount, core::u8{0});
        math::Random draw{_seed ^ (0xE7DE9B1Fu * (rule.tag + 1u))};
        // The raw Q16.16 word IS the share scaled by 65536, so it is the threshold
        // already. Multiplying by fromInt(0x10000) would have been the natural
        // spelling and would have overflowed Fixed32 to zero — granting the
        // species nothing, everywhere, silently.
        const core::i32 rawShare = math::Fixed32::fromFloat(rule.endemicShare).raw();
        const core::u32 threshold = rawShare <= 0 ? 0u : static_cast<core::u32>(rawShare);
        core::u32 granted = 0u;
        for (core::u32 r = 0u; r < _regions.regionCount; ++r)
        {
            allowedRegion[r] = (draw.next() & 0xFFFFu) < threshold ? 1u : 0u;
            granted += allowedRegion[r];
        }
        // A species granted nowhere is not endemic, it is extinct — and silently
        // dropping a rule is exactly the kind of nothing-happened bug that hides
        // for months. Give it the region its own draw favoured most.
        if (granted == 0u)
            allowedRegion[draw.next() % _regions.regionCount] = 1u;
    }

    // ── Hygrometry: distance to running water, only when a rule asks ────────
    Grid<core::u32> riverDistance;
    const bool haveRiverTest =
        rule.maxRiverDistance > 0u && _rivers.width() == _height.width() && _rivers.depth() == _height.depth();
    if (haveRiverTest)
        riverDistance = chamferDistance(_rivers);

    // The tree line is deliberately NOT tested here. A cell above it is still
    // eligible; it just ends up with a larger exclusion radius in
    // @ref selectBlueNoise, so the stand thins. Rejecting outright would draw a
    // contour line across the mountainside.
    for (core::u32 z = 0u; z < _height.depth(); ++z)
    {
        for (core::u32 x = 0u; x < _height.width(); ++x)
        {
            const BiomeId biome = _biomes.at(x, z);
            if (biome != rule.biome || !isHabitable(biome))
                continue;
            // Nothing stands on a cliff. Checking slope here rather than letting
            // physics sort it out keeps the world plausible before a single tick
            // has run.
            if (slopeAt(_height, x, z) > maxSlope)
                continue;
            if (haveMoisture)
            {
                const math::Fixed32 wetness = _moisture.at(x, z);
                if (wetness < minMoisture || wetness > maxMoisture)
                    continue;
            }
            const core::u32 index = _height.index(x, z);
            if (endemic && allowedRegion[_regions.regions[index]] == 0u)
                continue;
            if (haveRiverTest && riverDistance[index] > rule.maxRiverDistance)
                continue;
            cells.push_back(index);
        }
    }
    return cells;
}

void WorldBuilder::selectBlueNoise(const ScatterRule &rule, lpl::pmr::vector<core::u32> &cells, math::Random random) const
{
    if (cells.empty())
        return;

    const core::u32 target = static_cast<core::u32>(
        (math::Fixed32::fromFloat(rule.density) * math::Fixed32::fromInt(static_cast<core::i32>(cells.size())))
            .toInt());
    if (target == 0u)
    {
        cells.clear();
        return;
    }

    // Spacing that would put `target` points in this many cells if they were on a
    // square lattice. Deriving it from the requested density rather than asking for
    // a radius keeps the parameter meaning "how much of the ground is covered",
    // which is the thing a caller actually knows.
    const core::u32 baseSpacing = math::integerSqrt(static_cast<core::u32>(cells.size()) / target);
    const core::u32 spacing = baseSpacing < 1u ? 1u : baseSpacing;
    const bool haveMoisture = _moisture.width() == _height.width() && _moisture.depth() == _height.depth();
    const math::Fixed32 affinity = math::Fixed32::fromFloat(rule.moistureAffinity);

    math::Fixed32 lowest{};
    math::Fixed32 highest{};
    (void) heightRange(_height, lowest, highest);
    const math::Fixed32 span = highest - lowest;
    const math::Fixed32 treeLine = math::Fixed32::fromFloat(rule.treeLine);
    const math::Fixed32 falloff = math::Fixed32::fromFloat(rule.altitudeFalloff);
    const bool haveTreeLine = rule.treeLine < 1.0f && span.raw() != 0;

    // Dart-throwing against an occupancy grid. Candidates are drawn from the
    // eligible set in a shuffled order, so the result does not favour the top-left
    // corner the way a raster sweep does, and each is accepted only if no accepted
    // point lies within its exclusion radius.
    for (core::u32 i = static_cast<core::u32>(cells.size()); i-- > 1u;)
    {
        const core::u32 j = random.below(i + 1u);
        const core::u32 swap = cells[i];
        cells[i] = cells[j];
        cells[j] = swap;
    }

    Grid<core::u8> taken{_height.width(), _height.depth(), 0u};
    lpl::pmr::vector<core::u32> accepted;
    accepted.reserve(target);

    for (core::u32 i = 0u; i < cells.size() && accepted.size() < target; ++i)
    {
        const core::u32 cell = cells[i];
        const core::u32 x = cell % _height.width();
        const core::u32 z = cell / _height.width();

        // Variable radius: wetter ground supports a denser stand, and thin cold
        // air supports a sparser one. This is the Lipschitz-constrained density
        // function the flora survey describes, reduced to the two inputs that
        // matter here.
        math::Fixed32 stretch = math::Fixed32::one();
        if (haveMoisture && affinity.raw() > 0)
        {
            const math::Fixed32 dryness = math::Fixed32::one() - _moisture.at(x, z);
            stretch = stretch + dryness * affinity;
        }
        if (haveTreeLine && falloff.raw() > 0)
        {
            // Past the tree line the radius grows with how far past it the cell
            // sits, so the stand THINS instead of stopping at a drawn contour.
            // A hard cut in `eligibleCells` would put a straight edge across a
            // mountainside, which no forest has ever done.
            const math::Fixed32 relative = (_height[cell] - lowest) / span;
            if (relative > treeLine)
                stretch = stretch + (relative - treeLine) * falloff;
        }

        const math::Fixed32 stretched = math::Fixed32::fromInt(static_cast<core::i32>(spacing)) * stretch;
        const core::i32 widened = stretched.toInt();
        const core::u32 radius = widened < 1 ? 1u : static_cast<core::u32>(widened);

        bool clear = true;
        const core::i32 reach = static_cast<core::i32>(radius);
        for (core::i32 dz = -reach; dz <= reach && clear; ++dz)
        {
            for (core::i32 dx = -reach; dx <= reach && clear; ++dx)
            {
                // Squared distance, so the exclusion zone is a disc rather than the
                // square a Chebyshev test would give — a square one leaves visible
                // rows and columns in the result.
                if (dx * dx + dz * dz > reach * reach)
                    continue;
                const core::i32 nx = static_cast<core::i32>(x) + dx;
                const core::i32 nz = static_cast<core::i32>(z) + dz;
                if (taken.contains(nx, nz) && taken.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) != 0u)
                    clear = false;
            }
        }
        if (!clear)
            continue;

        taken.at(x, z) = 1u;
        accepted.push_back(cell);
    }

    // Emit in scan order rather than in the shuffled draw order: the entities are
    // written to chunks in this sequence, so a stable order keeps the world fold
    // independent of how the shuffle happened to land.
    //
    // Read it back off the occupancy grid rather than sorting the accepted list.
    // The grid is already indexed by position, so one sweep gives scan order for
    // free — sorting would be the one quadratic step in an otherwise linear pass.
    cells.clear();
    for (core::u32 i = 0u; i < taken.cellCount(); ++i)
        if (taken[i] != 0u)
            cells.push_back(i);
}

// ── Dependencies ────────────────────────────────────────────────────────────

void WorldBuilder::ensureTerrain()
{
    if (_height.empty())
        terrain(kDefaultWidth, kDefaultDepth);
}

void WorldBuilder::ensureDrainage()
{
    if (_drainageReady)
        return;
    ensureTerrain();
    _drainage = computeDrainage(_height);
    _drainageReady = true;
}

void WorldBuilder::ensureMoisture()
{
    if (_moistureReady)
        return;
    ensureDrainage();
    // Climate has to agree with the classification about where the sea is, or the
    // coastal term would measure the distance to a shoreline the biome map does
    // not have.
    MoistureParams params = _moistureParams;
    params.seaLevel = _biomeParams.seaLevel;
    _moisture = computeMoisture(_height, _drainage, params);
    _moistureReady = true;
}

void WorldBuilder::ensureClimate()
{
    if (_climateReady)
        return;
    ensureMoisture();
    // One sea level for the whole pipeline. The moisture's coast term, the
    // continentalness axis and the ocean test all have to agree about where the
    // water is, or the map has three shorelines in three different places.
    ClimateParams params = _climateParams;
    params.seaLevel = _biomeParams.seaLevel;
    _climate = computeClimate(_height, _moisture, _drainage, params);
    _climateReady = true;
}

void WorldBuilder::ensureBiomes()
{
    if (_biomesReady)
        return;
    biomes(_biomeParams);
}

// ── Output ──────────────────────────────────────────────────────────────────

BuiltWorldStats WorldBuilder::bakeGrids()
{
    ensureTerrain();
    ensureClimate();
    ensureBiomes();

    BuiltWorldStats stats;
    stats.terrainCells = _height.cellCount();
    stats.riverCells = _riverCells;
    stats.dungeonFloor = _dungeonFloor;
    stats.dungeonConnected = _dungeonConnected;
    stats.regionCount = _regions.regionCount;
    stats.settlementPlots = _settlementPlots;
    stats.settlementConnected = _settlementConnected;
    stats.lakeCells = _lakeCells;
    stats.roadCells = _roadCells;
    stats.townVoxels = _townVoxels;
    stats.roadsideModules = _roadsideModules;
    stats.caveLayers = _caveSystem.layerCount;
    stats.caveEntrances = _caveSystem.entranceCount;
    stats.caveHollow = _caveSystem.hollowCells;
    stats.caveReachable = _caveSystem.reachableCells;
    stats.undergroundVoxels = _undergroundVoxels;
    stats.heightSignature = foldHeightfield(_height);
    stats.biomeSignature = foldBiomeMap(_biomes);
    stats.climateSignature = foldClimateField(_climate);
    return stats;
}

BuiltWorldStats WorldBuilder::materialize(ecs::Registry &registry)
{
    BuiltWorldStats stats = bakeGrids();
    if (_height.empty())
        return stats;

    lpl::pmr::vector<Placement> placements;
    placements.reserve(static_cast<core::usize>(_height.cellCount()));

    const math::Fixed32 spacing = math::Fixed32::fromFloat(_cellSize);
    const math::Fixed32 groundHalf = spacing * math::Fixed32::half();
    for (core::u32 z = 0u; z < _height.depth(); ++z)
    {
        for (core::u32 x = 0u; x < _height.width(); ++x)
        {
            math::Fixed32 worldX{};
            math::Fixed32 worldZ{};
            cellToWorld(x, z, worldX, worldZ);
            placements.push_back(Placement{worldX, _height.at(x, z), worldZ, groundHalf, false});
        }
    }
    stats.terrainEntities = static_cast<core::u32>(placements.size());

    stats.propEntities = collectProps(placements);
    emit(registry, placements, nullptr);
    return stats;
}

BuiltWorldStats WorldBuilder::materializeProps(ecs::Registry &registry, lpl::pmr::vector<ecs::EntityId> *outIds)
{
    BuiltWorldStats stats = bakeGrids();
    if (_height.empty())
        return stats;

    lpl::pmr::vector<Placement> placements;
    stats.propEntities = collectProps(placements);
    emit(registry, placements, outIds);
    return stats;
}

void WorldBuilder::cellToWorld(core::u32 x, core::u32 z, math::Fixed32 &outX, math::Fixed32 &outZ) const
{
    const math::Fixed32 spacing = math::Fixed32::fromFloat(_cellSize);
    const math::Fixed32 halfWidth =
        spacing * math::Fixed32::fromInt(static_cast<core::i32>(_height.width())) * math::Fixed32::half();
    const math::Fixed32 halfDepth =
        spacing * math::Fixed32::fromInt(static_cast<core::i32>(_height.depth())) * math::Fixed32::half();
    outX = spacing * math::Fixed32::fromInt(static_cast<core::i32>(x)) - halfWidth;
    outZ = spacing * math::Fixed32::fromInt(static_cast<core::i32>(z)) - halfDepth;
}

core::u32 WorldBuilder::collectProps(lpl::pmr::vector<Placement> &placements)
{
    const math::Fixed32 spacing = math::Fixed32::fromFloat(_cellSize);
    const math::Fixed32 groundHalf = spacing * math::Fixed32::half();

    core::u32 total = 0u;
    // One stream per rule so adding a rule cannot shift the others.
    for (core::u32 r = 0u; r < _scatterRules.size(); ++r)
    {
        const ScatterRule &rule = _scatterRules[r];
        const math::Fixed32 propHalf = math::Fixed32::fromFloat(rule.halfExtent);

        lpl::pmr::vector<core::u32> cells = eligibleCells(rule);
        selectBlueNoise(rule, cells, math::deriveStream(_seed, 0x5CA7u + r));

        for (core::u32 i = 0u; i < cells.size(); ++i)
        {
            const core::u32 x = cells[i] % _height.width();
            const core::u32 z = cells[i] / _height.width();
            math::Fixed32 worldX{};
            math::Fixed32 worldZ{};
            cellToWorld(x, z, worldX, worldZ);
            placements.push_back(
                Placement{worldX, _height.at(x, z) + propHalf + groundHalf, worldZ, propHalf, rule.collidable});
            ++total;
        }
    }
    return total;
}

void WorldBuilder::emit(ecs::Registry &registry, const lpl::pmr::vector<Placement> &placements,
                        lpl::pmr::vector<ecs::EntityId> *outIds)
{
    if (placements.empty())
        return;

    // Two archetypes, filled in two passes. A collidable prop needs Velocity and Mass
    // for the physics to see it at all — the solver skips any partition missing them —
    // so it cannot share an archetype with decoration, and interleaving the two would
    // scatter each kind across both partitions with no way to tell which row is which.
    const ecs::ComponentId sceneryIds[] = {ecs::ComponentId::Position, ecs::ComponentId::AABB};
    const ecs::ComponentId obstacleIds[] = {ecs::ComponentId::Position, ecs::ComponentId::Velocity,
                                            ecs::ComponentId::Mass, ecs::ComponentId::AABB};

    for (core::u32 pass = 0u; pass < 2u; ++pass)
    {
        const bool collidable = pass == 1u;
        lpl::pmr::vector<Placement> group;
        for (core::u32 i = 0u; i < placements.size(); ++i)
            if (placements[i].collidable == collidable)
                group.push_back(placements[i]);
        if (group.empty())
            continue;

        const ecs::Archetype archetype{collidable ? std::span<const ecs::ComponentId>{obstacleIds} :
                                                    std::span<const ecs::ComponentId>{sceneryIds}};
        lpl::pmr::vector<ecs::EntityId> created;
        created.reserve(group.size());
        for (core::u32 i = 0u; i < group.size(); ++i)
        {
            auto id = registry.createEntity(archetype);
            if (!id)
                continue;
            created.push_back(*id);
            if (outIds != nullptr)
                outIds->push_back(*id);
        }
        fillPlacements(registry, archetype, group, created);
    }
}

void WorldBuilder::fillPlacements(ecs::Registry &registry, const ecs::Archetype &archetype,
                                  const lpl::pmr::vector<Placement> &placements,
                                  const lpl::pmr::vector<ecs::EntityId> &created)
{
    // Write through IDENTITY, never through row order.
    //
    // A collidable prop's archetype is {Position, Velocity, Mass, AABB} — exactly a
    // loose body's, because that is what the solver requires to see either of them.
    // So props and bodies share a partition, and the earlier version of this
    // function walked that partition from its first row and wrote the first
    // `expected` of them. Whoever else owned those rows lost them.
    //
    // Measured, not reasoned about: with twenty-four bodies already in the world,
    // scattering twenty-three trees overwrote TWENTY-THREE of them — their mass
    // became a prop's zero and their position became a tree's. In the viewer that
    // is the reported symptom exactly: regenerate, and boulders hang motionless in
    // mid-air. They were never stuck. A body with zero mass is one gravity does not
    // apply to, and it stops where it last was.
    //
    // Resolving each created entity costs one lookup per prop and removes the
    // assumption entirely: an entity is written where the registry says it lives.
    for (core::u32 n = 0u; n < created.size() && n < placements.size(); ++n)
    {
        const auto ref = registry.resolve(created[n]);
        if (!ref)
            continue;

        for (const auto &partition : registry.partitions())
        {
            if (!partition || !(partition->archetype() == archetype))
                continue;
            const auto &chunks = partition->chunks();
            if (ref->chunkIndex >= chunks.size() || !chunks[ref->chunkIndex])
                continue;
            const auto &chunk = chunks[ref->chunkIndex];
            const core::u32 i = ref->localIndex;
            if (i >= chunk->count())
                continue;

            const Placement &placement = placements[n];
            const FVec3 point{placement.x, placement.y, placement.z};
            const math::Fixed32 size = placement.halfExtent * math::Fixed32::fromInt(2);
            const FVec3 extents{size, size, size};

            auto *position = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *positionRead =
                static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Position)));
            auto *aabb = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB));
            auto *aabbRead = static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::AABB)));
            if (position == nullptr)
                break;

            // Both buffers, like every other generator: the simulation reads the
            // front buffer on its first tick and would otherwise start from
            // uninitialised memory.
            position[i] = point;
            if (positionRead != nullptr)
                positionRead[i] = point;
            if (aabb != nullptr)
                aabb[i] = extents;
            if (aabbRead != nullptr)
                aabbRead[i] = extents;

            if (placement.collidable)
            {
                // Mass zero is the solver's word for immovable, and a zero velocity
                // that gravity never touches (it too is gated on mass) keeps it that
                // way. Nothing else is needed to make a tree an obstacle.
                auto *mass = static_cast<math::Fixed32 *>(chunk->writeComponent(ecs::ComponentId::Mass));
                auto *massRead =
                    static_cast<math::Fixed32 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Mass)));
                auto *velocity = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
                auto *velocityRead =
                    static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Velocity)));
                const FVec3 still{};
                if (velocity != nullptr)
                    velocity[i] = still;
                if (velocityRead != nullptr)
                    velocityRead[i] = still;
                if (mass != nullptr)
                    mass[i] = math::Fixed32::zero();
                if (massRead != nullptr)
                    massRead[i] = math::Fixed32::zero();
            }
            break;
        }
    }
}

} // namespace lpl::procgen
