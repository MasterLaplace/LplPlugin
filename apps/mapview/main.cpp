/**
 * @file main.cpp
 * @brief A solo client: the engine hosting a procedurally generated world, on screen.
 *
 * A generator you cannot look at is a generator you are debugging by folding
 * hashes. Every number in this module — eight thousand cells of drainage, twelve
 * biomes, a rain shadow — is a claim about a shape, and a shape is the one thing a
 * signature cannot tell you about. So: a viewport.
 *
 * This is not a viewer bolted onto the side of the engine. It is a real
 * `engine::Engine` hosting a real `engine::World`, exactly the way the client app
 * does, with the only difference being what gets drawn at the end:
 *
 *  - the **Engine** owns the fixed-timestep loop, so the simulation advances on its
 *    own clock and the frame rate is free;
 *  - the **World** owns the game — its `Registry`, its `SystemScheduler`, and the
 *    terrain that procgen built into it;
 *  - `Config::enablePhysics` puts the engine's own physics on that scheduler, and
 *    the World adds the one system that is the *game's* business rather than the
 *    host's: keeping things on top of the ground;
 *  - networking is off. This is a solo game, and there is no second party for the
 *    world to agree with.
 *
 * Presentation is raw X11 and GLX with fixed-function OpenGL, and deliberately so:
 * the engine's Vulkan renderer is not finished, and finishing it to look at a
 * heightfield would be the wrong order of work. Nothing here links `render/`. When
 * that renderer is ready, the viewport moves and everything above stays.
 *
 * Everything drawn is float. That is allowed and it is the point: the world state is
 * authoritative Fixed32, and this only ever *reads* it — no value computed here
 * flows back into the simulation.
 *
 * @warning **Build this in release.** Not as an optimisation — as a requirement. The
 *          physics tick costs about 23 ms in a debug build and 1.2 ms in release, and the
 *          engine's loop is fixed-step: a tick that overruns its budget makes the loop
 *          spend every frame catching up, so rendering is starved and the window sits at
 *          eight frames a second while reporting a seven-millisecond frame. The frame was
 *          never the problem. Unoptimised Fixed32 arithmetic was.
 *
 *          `xmake f -m release --mapview=y && xmake build lpl-mapview`
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ai/Personality.hpp>
#include <lpl/ai/StigmergyField.hpp>
#include <lpl/ai/Swarm.hpp>
#include <lpl/ecology/Genome.hpp>
#include <lpl/ecology/Populations.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Entity.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/System.hpp>
#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/engine/Engine.hpp>
#include <lpl/engine/World.hpp>
#include <lpl/image/Font8x16.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/CaveSystem.hpp>
#include <lpl/procgen/Climate.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/procgen/Extrusion.hpp>
#include <lpl/procgen/FixedMath.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/Hydrology.hpp>
#include <lpl/procgen/Liminal.hpp>
#include <lpl/procgen/Settlement.hpp>
#include <lpl/procgen/ShapeGrammar.hpp>
#include <lpl/procgen/Streaming.hpp>
#include <lpl/procgen/ValueNoise.hpp>
#include <lpl/procgen/Voronoi.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <lpl/core/Log.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <utility>
#include <vector>

namespace {

using namespace lpl;

// ─────────────────────────────────────────────────────────────────────────────
// What the viewer is currently asking the generator for.
// ─────────────────────────────────────────────────────────────────────────────

/// How the surface is coloured.
/// How the surface is coloured.
///
/// @c Climate is one mode rather than six, and the axis it shows is a separate
/// option: the six axes are the same kind of quantity, and cycling shading
/// through six near-identical entries would bury the five modes that are not.
enum class Shading : int {
    Biome = 0,
    Height,
    Moisture,
    Drainage,
    Region,
    Slope,
    Climate,
    Count
};

/// Which grid is on screen.
///
/// Cutaway exists because "underground" is not a separate world: a dungeon means
/// nothing without the ground it is under. Drawing the terrain translucent above it
/// is the only view in which the two read as one place.
/// @c Liminal is not a view onto the same world — it is a different generator
/// entirely, and it is here because a liminal sector is the one thing in the
/// module whose whole point is how it *feels* to stand in. A fold cannot report
/// that, and the pipeline order that produces it (zone, split, erode, repair,
/// dress) is only obviously right once you can walk the result.
enum class View : int {
    Surface = 0,
    Cutaway,
    Underground,
    Liminal,
    Count
};

struct Options {
    core::u32 seed{1337u};
    core::u32 size{128u};
    procgen::NoiseKind noise{procgen::NoiseKind::Fbm};
    bool erosion{true};
    bool rivers{true};
    bool warp{false};
    bool terraces{false};
    bool settlement{true};
    bool roads{true};
    bool vegetation{true};
    bool wireframe{false};
    bool water{true};
    core::u32 windDirection{0u};
    procgen::DistanceMetric metric{procgen::DistanceMetric::Euclidean};
    Shading shading{Shading::Biome};
    View view{View::Surface};
    core::u32 caveKind{3u};      ///< 0 cellular, 1 BSP, 2 DLA, 3 layered system.
    core::u32 climateAxis{0u};   ///< Which of the six axes @c Shading::Climate shows.
    bool grammarBuildings{true}; ///< Raise the town with the shape grammar rather than as prisms.
    bool living{true};           ///< Run the ai/ecology layer on top of the world.
    bool chunkOverlay{false};    ///< Draw the streaming plan around the camera.
};

/// Everything the generator produced, plus the timing it took.
struct TerrainData {
    procgen::Heightfield height;
    procgen::Heightfield moisture;
    procgen::BiomeMap biomes;
    procgen::DrainageNetwork drainage;
    procgen::VoronoiDiagram regions;
    procgen::SettlementMap settlement;
    procgen::Grid<core::u8> roads;
    procgen::DungeonMap dungeon;
    procgen::Grid<core::u8> riverMask;
    procgen::ClimateField climate;
    procgen::CaveSystem caveSystem;
    procgen::VoxelVolume townVolume;
    procgen::VoxelVolume roadsideVolume;
    procgen::LiminalSpace liminal;
    math::Fixed32 low{};
    math::Fixed32 high{};
    core::u32 riverCells{0u};
    core::u32 raisedCells{0u};
    core::u32 maxAccumulation{0u};
    core::u32 biomeCounts[static_cast<core::u32>(procgen::BiomeId::Count)] = {};
    lpl::pmr::vector<procgen::BuildingPlot> plots;
    core::u32 dungeonFloor{0u};
    core::u32 roadCells{0u};
    bool dungeonConnected{false};
    core::u32 caveLayers{0u};
    core::u32 caveEntrances{0u};
    core::u32 caveHollow{0u};
    core::u32 caveReachable{0u};
    core::u32 roadsideModules{0u};
    core::u32 liminalSectors{0u};
    float seaLevel{0.0f};
    double buildMilliseconds{0.0};
};

const char *noiseName(procgen::NoiseKind kind)
{
    switch (kind)
    {
    case procgen::NoiseKind::Fbm: return "fBm";
    case procgen::NoiseKind::Ridged: return "ridged";
    case procgen::NoiseKind::Billow: return "billow";
    }
    return "?";
}

const char *shadingName(Shading shading)
{
    switch (shading)
    {
    case Shading::Biome: return "biome";
    case Shading::Height: return "height";
    case Shading::Moisture: return "moisture";
    case Shading::Drainage: return "drainage";
    case Shading::Region: return "region";
    case Shading::Slope: return "slope";
    case Shading::Climate: return "climate";
    case Shading::Count: break;
    }
    return "?";
}

const char *metricName(procgen::DistanceMetric metric)
{
    switch (metric)
    {
    case procgen::DistanceMetric::Euclidean: return "euclidean";
    case procgen::DistanceMetric::Manhattan: return "manhattan";
    case procgen::DistanceMetric::Chebyshev: return "chebyshev";
    }
    return "?";
}

const char *windName(core::u32 direction)
{
    switch (direction % 4u)
    {
    case 0u: return "east";
    case 1u: return "west";
    case 2u: return "south";
    default: return "north";
    }
}

const char *viewName(View view)
{
    switch (view)
    {
    case View::Surface: return "surface";
    case View::Cutaway: return "cutaway";
    case View::Underground: return "underground";
    case View::Liminal: return "liminal";
    case View::Count: break;
    }
    return "?";
}

const char *caveName(core::u32 kind)
{
    switch (kind % 4u)
    {
    case 0u: return "cellular";
    case 1u: return "bsp";
    case 2u: return "dla";
    default: return "layered";
    }
}

const char *climateAxisName(core::u32 axis)
{
    switch (axis % procgen::kClimateAxisCount)
    {
    case 0u: return "temperature";
    case 1u: return "moisture";
    case 2u: return "continentalness";
    case 3u: return "erosion";
    case 4u: return "depth";
    default: return "weirdness";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generation
// ─────────────────────────────────────────────────────────────────────────────

double nowMilliseconds()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) * 1000.0 + static_cast<double>(ts.tv_nsec) / 1.0e6;
}

// A note on telling scenery from bodies.
//
// `readComponent` returns a pointer for every component id, allocated or not, so a null
// check does NOT mean "this archetype lacks the component" — every chunk answers for
// every component. The test that means what it says is `archetype().has(...)`, which is
// what the engine's own physics uses to decide what it may integrate.
//
// Getting this wrong is not subtle: every prop was drawn once as a plant and once as a
// boulder, and every boulder likewise, so the world carried twice the geometry it should
// and nothing could be told apart.

TerrainData generateTerrain(const Options &options, ecs::Registry *registry,
                            lpl::pmr::vector<ecs::EntityId> *outPropIds)
{
    const double started = nowMilliseconds();
    TerrainData world;

    procgen::NoiseParams noise;
    noise.seed = options.seed;
    noise.kind = options.noise;
    noise.warpStrength = options.warp ? 8.0f : 0.0f;

    procgen::MoistureParams climate;
    climate.windDirection = options.windDirection;

    procgen::WorldBuilder builder{options.seed};
    builder.terrain(options.size, options.size, noise).climate(climate);
    if (options.terraces)
        builder.terraces(8u);
    if (options.erosion)
        builder.erode();
    if (options.rivers)
        builder.rivers();

    // ── Put the world in the frame the physics expects, once, before anything is
    //    placed in it ────────────────────────────────────────────────────────────
    //
    // The built-in physics stops any body below kDefaultHalfHeight, half a unit above
    // the origin, and it knows nothing about a heightfield. A world whose ground dips
    // under that line therefore has two floors: the real one, and a flat invisible one
    // the bodies actually come to rest on.
    //
    // Shifting the terrain afterwards and then chasing everything already placed in it
    // is the wrong way round — and it does not work, because "everything" grows: props
    // today, spawn points and structures tomorrow, each needing to remember it lives in
    // a frame that moved. The world is shifted HERE, before a single entity exists, and
    // the biome thresholds are shifted with it so the classification still sees the same
    // relative elevations it was tuned for.
    math::Fixed32 rawLow{};
    math::Fixed32 rawHigh{};
    (void) procgen::heightRange(builder.heightfield(), rawLow, rawHigh);
    const float clearance = 1.5f;
    const float lift = clearance - rawLow.toFloat();
    if (lift > 0.0f)
    {
        // normalizeHeights maps min and max onto the two bounds linearly, so asking for
        // the same span translated is an exact shift, not a rescale.
        builder.normalize(rawLow.toFloat() + lift, rawHigh.toFloat() + lift);
    }

    procgen::BiomeParams biomeParams;
    if (lift > 0.0f)
    {
        biomeParams.seaLevel += lift;
        biomeParams.mountainHeight += lift;
        biomeParams.snowHeight += lift;
    }

    procgen::VoronoiParams provinces;
    provinces.cellSize = options.size / 6u == 0u ? 6u : options.size / 6u;
    provinces.metric = options.metric;
    provinces.warpStrength = options.warp ? 6.0f : 0.0f;
    builder.regions(provinces);

    // The six-axis hypercube, not just the two-axis Whittaker input. The
    // classifier already reads it; asking for it here is what makes it
    // inspectable, and the sea level has to travel with the lifted world for the
    // continentalness and depth axes to mean anything.
    procgen::ClimateParams axes;
    axes.seaLevel += lift > 0.0f ? lift : 0.0f;
    builder.climateAxes(axes);
    builder.biomes(biomeParams);

    if (options.settlement)
    {
        procgen::SettlementParams town;
        town.districtSize = 14u;
        // The buildable floor is an absolute height too, so it moves with the world.
        town.minHeight += lift > 0.0f ? lift : 0.0f;
        builder.settlement(town);
    }

    if (options.roads)
    {
        procgen::RoadParams highways;
        highways.iterations = 6u;
        highways.stepLength = 3u;
        // Same absolute-height caveat as the settlement: the world was lifted, so
        // a floor of zero would put roads under the new ground rather than above
        // the old water line.
        highways.minHeight += lift > 0.0f ? lift : 0.0f;
        builder.roads(highways);
    }

    // Underground, generated alongside so the view can be toggled without a rebuild.
    if (options.caveKind % 4u == 3u)
    {
        // The layered system: a stack of plans joined by shafts, at least one of
        // which pierces the surface. The three flat kinds below are all buried
        // voids with no way in — which is exactly what looking at them makes
        // obvious and what a floor count never did.
        procgen::CaveSystemParams system;
        system.width = options.size;
        system.depth = options.size;
        system.seed = options.seed;
        system.layers = 3u;
        system.entrances = 3u;
        builder.caveSystem(system);
    }
    else if (options.caveKind % 4u == 1u)
    {
        procgen::BspDungeonParams rooms;
        rooms.width = options.size;
        rooms.depth = options.size;
        rooms.seed = options.seed;
        builder.dungeon(rooms);
    }
    else if (options.caveKind % 4u == 2u)
    {
        procgen::DlaParams dla;
        dla.width = options.size;
        dla.depth = options.size;
        dla.seed = options.seed;
        dla.particles = options.size * 8u;
        builder.dlaCaves(dla);
    }
    else
    {
        procgen::CaveParams cave;
        cave.width = options.size;
        cave.depth = options.size;
        cave.seed = options.seed;
        builder.caves(cave);
    }

    // The town's third dimension. `extrudeTown` gives prisms and the viewer used to
    // draw its own boxes on top of the footprints; the grammar gives a base course,
    // storeys and a roof, which is the difference between a bar chart and something
    // that reads as architecture. Drawing the volume rather than a heuristic box also
    // means what is on screen is what the generator actually produced.
    if (options.settlement && options.grammarBuildings)
    {
        procgen::BuildingGrammarParams grammar;
        grammar.minFloors = 1u;
        grammar.maxFloors = 5u;
        grammar.roofHeight = 2u;
        builder.buildings(grammar);
    }
    if (options.roads && options.grammarBuildings)
    {
        // The linear form of the same grammar: two fence posts to one lamp, then a
        // gap. The point is that one parser serves both.
        builder.roadside("{[A,P]:2,[BL,P]:1}*,[G,P]", 3u);
    }

    // Vegetation: the same scatter the builder would place in a real game, with one
    // rule per biome so a forest is conifers and a savanna is scrub. This is what
    // `bakeGrids` alone never produced — it stops before any entity exists, so the map
    // came out bare and the scatter rules were never exercised at all.
    procgen::ScatterRule conifer;
    conifer.biome = procgen::BiomeId::Taiga;
    conifer.density = 0.16f;
    conifer.halfExtent = 0.42f;
    conifer.maxSlope = 1.6f;
    conifer.collidable = true;
    builder.scatter(conifer);

    procgen::ScatterRule broadleaf;
    broadleaf.biome = procgen::BiomeId::Forest;
    broadleaf.density = 0.18f;
    broadleaf.halfExtent = 0.5f;
    broadleaf.maxSlope = 1.6f;
    broadleaf.collidable = true;
    builder.scatter(broadleaf);

    procgen::ScatterRule jungle;
    jungle.biome = procgen::BiomeId::Rainforest;
    jungle.density = 0.22f;
    jungle.halfExtent = 0.55f;
    jungle.maxSlope = 1.8f;
    jungle.collidable = true;
    builder.scatter(jungle);

    procgen::ScatterRule scrub;
    scrub.biome = procgen::BiomeId::Savanna;
    scrub.density = 0.05f;
    scrub.halfExtent = 0.3f;
    builder.scatter(scrub);

    procgen::ScatterRule cactus;
    cactus.biome = procgen::BiomeId::Desert;
    cactus.density = 0.03f;
    cactus.halfExtent = 0.22f;
    builder.scatter(cactus);

    procgen::ScatterRule reed;
    reed.biome = procgen::BiomeId::Marsh;
    reed.density = 0.2f;
    reed.halfExtent = 0.25f;
    builder.scatter(reed);

    // Materialise the vegetation into the World's registry, so the props are real
    // entities the simulation owns rather than something the viewer draws on top.
    const procgen::BuiltWorldStats stats =
        registry != nullptr ? builder.materializeProps(*registry, outPropIds) : builder.bakeGrids();

    world.height = builder.heightfield();
    world.moisture = builder.moisture();
    world.biomes = builder.biomeMap();
    world.drainage = builder.drainage();
    world.regions = builder.regionMap();
    world.settlement = builder.settlementMap();
    world.roads = builder.roadMap();
    world.dungeon = builder.dungeonMap();
    world.riverMask = procgen::riverMask(world.drainage, 0.02f);
    world.climate = builder.climateField();
    world.caveSystem = builder.caves();
    world.townVolume = builder.townVolume();
    world.roadsideVolume = builder.roadsideVolume();
    world.caveLayers = stats.caveLayers;
    world.caveEntrances = stats.caveEntrances;
    world.caveHollow = stats.caveHollow;
    world.caveReachable = stats.caveReachable;
    world.roadsideModules = stats.roadsideModules;
    (void) procgen::heightRange(world.height, world.low, world.high);
    world.riverCells = stats.riverCells;
    world.roadCells = stats.roadCells;
    world.raisedCells = world.drainage.raisedCells;
    world.maxAccumulation = world.drainage.maxAccumulation;
    world.dungeonFloor = stats.dungeonFloor;
    world.dungeonConnected = stats.dungeonConnected;
    world.plots = builder.plots();
    procgen::countBiomes(world.biomes, world.biomeCounts);

    world.seaLevel = biomeParams.seaLevel;

    // The scenery was placed against the terrain BEFORE it was lifted, so it is a whole
    // offset out of frame — buried under the ground it was supposed to stand on.
    // Everything living in the simulation's frame has to move with it, and the props are
    // the only things written before the shift.
    world.buildMilliseconds = nowMilliseconds() - started;
    return world;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour
// ─────────────────────────────────────────────────────────────────────────────

struct Rgb {
    float r, g, b;
};

/// The palette. Chosen so a glance reads as a map, not as a debug ramp.
Rgb biomeColour(procgen::BiomeId biome)
{
    switch (biome)
    {
    case procgen::BiomeId::Ocean: return {0.07f, 0.20f, 0.42f};
    case procgen::BiomeId::Beach: return {0.83f, 0.77f, 0.55f};
    case procgen::BiomeId::Snow: return {0.94f, 0.95f, 0.97f};
    case procgen::BiomeId::Tundra: return {0.60f, 0.62f, 0.56f};
    case procgen::BiomeId::Taiga: return {0.20f, 0.38f, 0.31f};
    case procgen::BiomeId::Rock: return {0.44f, 0.42f, 0.40f};
    case procgen::BiomeId::Desert: return {0.85f, 0.72f, 0.42f};
    case procgen::BiomeId::Savanna: return {0.70f, 0.68f, 0.34f};
    case procgen::BiomeId::Grassland: return {0.42f, 0.60f, 0.30f};
    case procgen::BiomeId::Forest: return {0.20f, 0.45f, 0.22f};
    case procgen::BiomeId::Rainforest: return {0.11f, 0.36f, 0.18f};
    case procgen::BiomeId::Marsh: return {0.31f, 0.42f, 0.33f};
    case procgen::BiomeId::Lake: return {0.16f, 0.35f, 0.55f};
    case procgen::BiomeId::Count: break;
    }
    return {1.0f, 0.0f, 1.0f};
}

/// Amber-on-abyss ramp for the scalar views, so they read as instrumentation.
Rgb rampColour(float t)
{
    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;
    return {0.05f + 0.95f * t, 0.05f + 0.62f * t * t, 0.10f + 0.15f * t * t * t};
}

Rgb surfaceColour(const TerrainData &world, const Options &options, core::u32 x, core::u32 z)
{
    const float span = (world.high - world.low).toFloat();
    const float normalized = span > 0.0f ? (world.height.at(x, z) - world.low).toFloat() / span : 0.5f;

    switch (options.shading)
    {
    case Shading::Height: return rampColour(normalized);
    case Shading::Moisture:
        return world.moisture.empty() ? Rgb{0.5f, 0.5f, 0.5f} : rampColour(world.moisture.at(x, z).toFloat());
    case Shading::Drainage: {
        if (world.maxAccumulation == 0u)
            return {0.1f, 0.1f, 0.1f};
        // Logarithmic, for the same reason the moisture term is: accumulation
        // spans four orders of magnitude and a linear ramp shows only the trunk.
        const float flow = static_cast<float>(world.drainage.accumulation.at(x, z));
        const float scale = std::log(1.0f + flow) / std::log(1.0f + static_cast<float>(world.maxAccumulation));
        return rampColour(scale);
    }
    case Shading::Region: {
        if (world.regions.regions.empty())
            return {0.2f, 0.2f, 0.2f};
        const core::u16 region = world.regions.regions.at(x, z);
        // Hash the id so adjacent regions never share a shade.
        const core::u32 h = procgen::ValueNoise2D::hash2(static_cast<core::i32>(region), 7, 0x9E37u);
        return {0.25f + 0.7f * static_cast<float>((h >> 0) & 0xFFu) / 255.0f,
                0.25f + 0.7f * static_cast<float>((h >> 8) & 0xFFu) / 255.0f,
                0.25f + 0.7f * static_cast<float>((h >> 16) & 0xFFu) / 255.0f};
    }
    case Shading::Slope: return rampColour(procgen::slopeAt(world.height, x, z).toFloat() * 0.6f);
    case Shading::Climate: {
        if (world.climate.empty())
            return {0.2f, 0.2f, 0.2f};
        // Every axis is normalised to [0, 1] by the climate pass, so one ramp
        // reads all six — and the fact that it does is itself the claim being
        // looked at: an axis that came out flat or saturated is a bug you can
        // see here and nowhere in a signature.
        const core::u32 axis = options.climateAxis % procgen::kClimateAxisCount;
        return rampColour(world.climate.axes[axis].at(x, z).toFloat());
    }
    case Shading::Biome:
    case Shading::Count: break;
    }
    return biomeColour(world.biomes.at(x, z));
}

// ─────────────────────────────────────────────────────────────────────────────
// Text, drawn from the engine's console font as GL quads
// ─────────────────────────────────────────────────────────────────────────────

void drawGlyph(unsigned char character, float x, float y, float scale)
{
    const core::u8 *glyph = image::kFont8x16[character];
    glBegin(GL_QUADS);
    for (core::u32 row = 0u; row < image::kFontHeight; ++row)
    {
        const core::u8 bits = glyph[row];
        for (core::u32 column = 0u; column < image::kFontWidth; ++column)
        {
            if ((bits & (0x80u >> column)) == 0u)
                continue;
            const float px = x + static_cast<float>(column) * scale;
            const float py = y + static_cast<float>(row) * scale;
            glVertex2f(px, py);
            glVertex2f(px + scale, py);
            glVertex2f(px + scale, py + scale);
            glVertex2f(px, py + scale);
        }
    }
    glEnd();
}

void drawText(const char *text, float x, float y, float scale)
{
    float cursor = x;
    for (const char *c = text; *c != '\0'; ++c)
    {
        drawGlyph(static_cast<unsigned char>(*c), cursor, y, scale);
        cursor += static_cast<float>(image::kFontWidth) * scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh
// ─────────────────────────────────────────────────────────────────────────────

struct Vertex {
    float x, y, z;
    float nx, ny, nz;
    Rgb colour;
};

/// Builds the surface mesh once per regeneration, so drawing is only a walk.
std::vector<Vertex> buildSurfaceMesh(const TerrainData &world, const Options &options)
{
    std::vector<Vertex> vertices;
    if (world.height.empty())
        return vertices;

    const core::u32 width = world.height.width();
    const core::u32 depth = world.height.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;

    // Per-cell normals from the central difference, which is what makes the
    // relief read at all: flat shading on a heightfield is nearly invisible.
    std::vector<Vertex> grid(static_cast<std::size_t>(width) * depth);
    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            const float left = world.height.clamped(static_cast<core::i32>(x) - 1, static_cast<core::i32>(z)).toFloat();
            const float right =
                world.height.clamped(static_cast<core::i32>(x) + 1, static_cast<core::i32>(z)).toFloat();
            const float back = world.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z) - 1).toFloat();
            const float front =
                world.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z) + 1).toFloat();

            float nx = left - right;
            float ny = 2.0f;
            float nz = back - front;
            const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (length > 0.0f)
            {
                nx /= length;
                ny /= length;
                nz /= length;
            }

            Rgb colour = surfaceColour(world, options, x, z);
            // Rivers, drawn as part of the surface rather than over it: a river is
            // a property of the ground, and an overlay would z-fight with it.
            if (options.rivers && !world.riverMask.empty() && world.riverMask.at(x, z) != 0u)
                colour = {0.16f, 0.42f, 0.72f};
            if (options.settlement && !world.settlement.empty())
            {
                switch (world.settlement.at(x, z))
                {
                case procgen::SettlementCell::Road: colour = {0.35f, 0.32f, 0.30f}; break;
                case procgen::SettlementCell::Plaza: colour = {0.55f, 0.50f, 0.44f}; break;
                case procgen::SettlementCell::Plot: colour = {0.82f, 0.55f, 0.22f}; break;
                default: break;
                }
            }
            // The highway network, on top of the town's own streets: it is the
            // long-distance layer, grown by the grammar and steered by the field,
            // where the settlement's roads are district borders. Drawn last so a
            // road crossing a town reads as passing through it.
            if (options.roads && !world.roads.empty() && world.roads.at(x, z) != 0u)
                colour = {0.24f, 0.22f, 0.20f};

            grid[world.height.index(x, z)] = Vertex{static_cast<float>(x) - halfW,
                                                    world.height.at(x, z).toFloat(),
                                                    static_cast<float>(z) - halfD,
                                                    nx,
                                                    ny,
                                                    nz,
                                                    colour};
        }
    }

    vertices.reserve(static_cast<std::size_t>(width - 1u) * (depth - 1u) * 6u);
    for (core::u32 z = 0u; z + 1u < depth; ++z)
    {
        for (core::u32 x = 0u; x + 1u < width; ++x)
        {
            const Vertex &a = grid[world.height.index(x, z)];
            const Vertex &b = grid[world.height.index(x + 1u, z)];
            const Vertex &c = grid[world.height.index(x + 1u, z + 1u)];
            const Vertex &d = grid[world.height.index(x, z + 1u)];
            vertices.push_back(a);
            vertices.push_back(b);
            vertices.push_back(c);
            vertices.push_back(a);
            vertices.push_back(c);
            vertices.push_back(d);
        }
    }
    return vertices;
}

/**
 * @brief Builds the underground as a real volume: a floor, and walls around it.
 *
 * A flat sheet of coloured quads is not a cave. What makes a dungeon legible is the
 * WALLS — the boundary between the carved space and the rock — because that is the
 * only thing that shows a corridor as a corridor and a chamber as a chamber. So each
 * open cell contributes a floor quad, and each of its solid neighbours contributes a
 * vertical face: the mesh is the surface of the void, not a picture of it.
 *
 * Only the boundary faces are emitted. Rock is not drawn at all, which is both the
 * cheap thing and the right thing: filling the solid would hide the very space the
 * view exists to show.
 */
std::vector<Vertex> buildDungeonMesh(const TerrainData &world, float depthBelow)
{
    std::vector<Vertex> vertices;
    if (world.dungeon.empty())
        return vertices;

    const core::u32 width = world.dungeon.width();
    const core::u32 depth = world.dungeon.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;
    const float wallHeight = 1.6f;

    const Rgb floorColour{0.52f, 0.44f, 0.36f};
    const Rgb wallColour{0.30f, 0.26f, 0.24f};
    const Rgb capColour{0.66f, 0.44f, 0.20f};

    const auto solid = [&world](core::i32 x, core::i32 z) {
        // Outside the map counts as rock, so the cave is walled in rather than open
        // to nothing at the border.
        if (!world.dungeon.contains(x, z))
            return true;
        return world.dungeon.at(static_cast<core::u32>(x), static_cast<core::u32>(z)) == procgen::DungeonCell::Wall;
    };

    const auto quad = [&vertices](float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                                  float cz, float dx, float dy, float dz, float nx, float ny, float nz,
                                  const Rgb &colour) {
        const Vertex a{ax, ay, az, nx, ny, nz, colour};
        const Vertex b{bx, by, bz, nx, ny, nz, colour};
        const Vertex c{cx, cy, cz, nx, ny, nz, colour};
        const Vertex d{dx, dy, dz, nx, ny, nz, colour};
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
        vertices.push_back(a);
        vertices.push_back(c);
        vertices.push_back(d);
    };

    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            if (solid(static_cast<core::i32>(x), static_cast<core::i32>(z)))
                continue;

            // The cave hangs a fixed distance below the ground above it, so it follows
            // the terrain instead of lying on a flat plane the surface never touches.
            const float ground =
                world.height.empty() ?
                    0.0f :
                    world.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
            const float floorY = ground - depthBelow;
            const float ceilY = floorY + wallHeight;

            const float x0 = static_cast<float>(x) - halfW;
            const float x1 = x0 + 1.0f;
            const float z0 = static_cast<float>(z) - halfD;
            const float z1 = z0 + 1.0f;

            quad(x0, floorY, z0, x1, floorY, z0, x1, floorY, z1, x0, floorY, z1, 0.0f, 1.0f, 0.0f, floorColour);

            // A wall wherever the rock begins.
            if (solid(static_cast<core::i32>(x) + 1, static_cast<core::i32>(z)))
                quad(x1, floorY, z0, x1, ceilY, z0, x1, ceilY, z1, x1, floorY, z1, 1.0f, 0.0f, 0.0f, wallColour);
            if (solid(static_cast<core::i32>(x) - 1, static_cast<core::i32>(z)))
                quad(x0, floorY, z0, x0, ceilY, z0, x0, ceilY, z1, x0, floorY, z1, -1.0f, 0.0f, 0.0f, wallColour);
            if (solid(static_cast<core::i32>(x), static_cast<core::i32>(z) + 1))
                quad(x0, floorY, z1, x0, ceilY, z1, x1, ceilY, z1, x1, floorY, z1, 0.0f, 0.0f, 1.0f, wallColour);
            if (solid(static_cast<core::i32>(x), static_cast<core::i32>(z) - 1))
                quad(x0, floorY, z0, x0, ceilY, z0, x1, ceilY, z0, x1, floorY, z0, 0.0f, 0.0f, -1.0f, wallColour);

            // Cap the top of the walls, so from above the plan reads as a plan.
            if (solid(static_cast<core::i32>(x) + 1, static_cast<core::i32>(z)) ||
                solid(static_cast<core::i32>(x) - 1, static_cast<core::i32>(z)) ||
                solid(static_cast<core::i32>(x), static_cast<core::i32>(z) + 1) ||
                solid(static_cast<core::i32>(x), static_cast<core::i32>(z) - 1))
                quad(x0, ceilY, z0, x1, ceilY, z0, x1, ceilY, z1, x0, ceilY, z1, 0.0f, 1.0f, 0.0f, capColour);
        }
    }
    return vertices;
}

/**
 * @brief Raises the settlement: buildings as boxes on their footprints, roads as kerbs.
 *
 * Painting the plots onto the ground says where a town is. It does not put a town
 * there — a settlement is read by its silhouette, and a silhouette needs height. Each
 * footprint from @ref procgen::WorldBuilder::plots becomes one box, its height derived
 * from its own area and district so a quarter has a character rather than a uniform
 * skyline, and its base sunk to the lowest ground it covers so nothing floats over a
 * slope.
 */
std::vector<Vertex> buildTownMesh(const TerrainData &world, const lpl::pmr::vector<procgen::BuildingPlot> &plots)
{
    std::vector<Vertex> vertices;
    if (world.height.empty())
        return vertices;

    const float halfW = static_cast<float>(world.height.width()) * 0.5f;
    const float halfD = static_cast<float>(world.height.depth()) * 0.5f;

    const auto box = [&vertices](float x0, float x1, float y0, float y1, float z0, float z1, const Rgb &wall,
                                 const Rgb &roof) {
        const auto face = [&vertices](float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                                      float cz, float dx, float dy, float dz, float nx, float ny, float nz,
                                      const Rgb &colour) {
            const Vertex a{ax, ay, az, nx, ny, nz, colour};
            const Vertex b{bx, by, bz, nx, ny, nz, colour};
            const Vertex c{cx, cy, cz, nx, ny, nz, colour};
            const Vertex d{dx, dy, dz, nx, ny, nz, colour};
            vertices.push_back(a);
            vertices.push_back(b);
            vertices.push_back(c);
            vertices.push_back(a);
            vertices.push_back(c);
            vertices.push_back(d);
        };
        face(x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1, 0.0f, 1.0f, 0.0f, roof);
        face(x0, y0, z1, x0, y1, z1, x1, y1, z1, x1, y0, z1, 0.0f, 0.0f, 1.0f, wall);
        face(x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z0, 0.0f, 0.0f, -1.0f, wall);
        face(x1, y0, z1, x1, y1, z1, x1, y1, z0, x1, y0, z0, 1.0f, 0.0f, 0.0f, wall);
        face(x0, y0, z0, x0, y1, z0, x0, y1, z1, x0, y0, z1, -1.0f, 0.0f, 0.0f, wall);
    };

    for (core::usize p = 0u; p < plots.size(); ++p)
    {
        const procgen::BuildingPlot &plot = plots[p];

        // Sink the base to the lowest ground the footprint covers: a box placed at the
        // centre height would hang off the downhill corner.
        float lowest = 1.0e9f;
        for (core::u32 z = plot.z; z < plot.z + plot.depth; ++z)
            for (core::u32 x = plot.x; x < plot.x + plot.width; ++x)
            {
                const float h = world.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
                if (h < lowest)
                    lowest = h;
            }
        if (lowest > 1.0e8f)
            continue;

        // Height from the footprint's area and its district, hashed: a big plot on a
        // busy district gets a tall building, and the variation is reproducible.
        const core::u32 area = plot.width * plot.depth;
        const core::u32 h = procgen::ValueNoise2D::hash2(static_cast<core::i32>(plot.x), static_cast<core::i32>(plot.z),
                                                         static_cast<core::u32>(plot.district) + 1u);
        const float storeys = 1.0f + static_cast<float>(area) * 0.28f + static_cast<float>(h & 0x7u) * 0.55f;

        const float x0 = static_cast<float>(plot.x) - halfW + 0.12f;
        const float x1 = static_cast<float>(plot.x + plot.width) - halfW - 0.12f;
        const float z0 = static_cast<float>(plot.z) - halfD + 0.12f;
        const float z1 = static_cast<float>(plot.z + plot.depth) - halfD - 0.12f;
        const float y0 = lowest - 0.4f;
        const float y1 = lowest + storeys;

        const float tint = static_cast<float>((h >> 8) & 0x3Fu) / 63.0f;
        const Rgb wall{0.58f + 0.22f * tint, 0.48f + 0.16f * tint, 0.38f + 0.10f * tint};
        const Rgb roof{0.34f + 0.10f * tint, 0.22f + 0.06f * tint, 0.20f};
        box(x0, x1, y0, y1, z0, z1, wall, roof);
    }
    return vertices;
}

/**
 * @brief Meshes a voxel volume, emitting only the faces that border empty space.
 *
 * The generic mesher the grammar products need: the town raised by
 * @ref procgen::buildTown and the fences and lamps @ref procgen::decoratePath
 * leaves along the roads are both @ref procgen::VoxelVolume, so both arrive here.
 *
 * Interior faces are skipped, which is not an optimisation so much as a
 * correctness matter for a translucent pass: drawing the inside of a solid means
 * every wall is two coincident surfaces, and the z-fighting reads as noise on the
 * roofs rather than as the "too many quads" it is.
 *
 * @param volume    What to mesh.
 * @param world     The terrain, so the volume sits on the ground it was planned on.
 * @param baseLift  World units between the terrain and level 0 of the volume.
 * @param palette   Colour per material id; index 0 is never drawn.
 */
std::vector<Vertex> buildVoxelMesh(const procgen::VoxelVolume &volume, const TerrainData &world, float baseLift,
                                   const Rgb *palette, core::u32 paletteSize)
{
    std::vector<Vertex> vertices;
    if (volume.empty() || world.height.empty())
        return vertices;

    const float halfW = static_cast<float>(world.height.width()) * 0.5f;
    const float halfD = static_cast<float>(world.height.depth()) * 0.5f;
    const float cell = 1.0f;

    const auto solid = [&volume](core::i32 x, core::i32 y, core::i32 z) {
        if (x < 0 || y < 0 || z < 0 || static_cast<core::u32>(x) >= volume.width ||
            static_cast<core::u32>(y) >= volume.levels || static_cast<core::u32>(z) >= volume.depth)
            return false;
        return volume.at(static_cast<core::u32>(x), static_cast<core::u32>(y), static_cast<core::u32>(z)) != 0u;
    };

    const auto quad = [&vertices](float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                                  float cz, float dx, float dy, float dz, float nx, float ny, float nz,
                                  const Rgb &colour) {
        const Vertex a{ax, ay, az, nx, ny, nz, colour};
        const Vertex b{bx, by, bz, nx, ny, nz, colour};
        const Vertex c{cx, cy, cz, nx, ny, nz, colour};
        const Vertex d{dx, dy, dz, nx, ny, nz, colour};
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
        vertices.push_back(a);
        vertices.push_back(c);
        vertices.push_back(d);
    };

    for (core::u32 y = 0u; y < volume.levels; ++y)
        for (core::u32 z = 0u; z < volume.depth; ++z)
            for (core::u32 x = 0u; x < volume.width; ++x)
            {
                const core::u8 material = volume.at(x, y, z);
                if (material == 0u)
                    continue;
                const Rgb colour = palette[material < paletteSize ? material : 0u];

                // The ground under the whole footprint, not under this column: a
                // building sampled per column would follow the slope it stands on and
                // shear apart. The volume is a plan, and a plan sits on one datum.
                const float ground =
                    world.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
                const float y0 = ground + baseLift + static_cast<float>(y) * cell;
                const float y1 = y0 + cell;
                const float x0 = static_cast<float>(x) - halfW;
                const float x1 = x0 + cell;
                const float z0 = static_cast<float>(z) - halfD;
                const float z1 = z0 + cell;

                const core::i32 ix = static_cast<core::i32>(x);
                const core::i32 iy = static_cast<core::i32>(y);
                const core::i32 iz = static_cast<core::i32>(z);

                if (!solid(ix, iy + 1, iz))
                    quad(x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1, 0.0f, 1.0f, 0.0f, colour);
                if (!solid(ix, iy - 1, iz))
                    quad(x0, y0, z1, x1, y0, z1, x1, y0, z0, x0, y0, z0, 0.0f, -1.0f, 0.0f, colour);
                if (!solid(ix + 1, iy, iz))
                    quad(x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1, 1.0f, 0.0f, 0.0f, colour);
                if (!solid(ix - 1, iy, iz))
                    quad(x0, y0, z1, x0, y1, z1, x0, y1, z0, x0, y0, z0, -1.0f, 0.0f, 0.0f, colour);
                if (!solid(ix, iy, iz + 1))
                    quad(x0, y0, z1, x0, y1, z1, x1, y1, z1, x1, y0, z1, 0.0f, 0.0f, 1.0f, colour);
                if (!solid(ix, iy, iz - 1))
                    quad(x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z0, 0.0f, 0.0f, -1.0f, colour);
            }
    return vertices;
}

/**
 * @brief Meshes the layered cave system: every floor, and the shafts joining them.
 *
 * The flat underground could be drawn as one plan because it was one plan. A
 * system is a stack, so what has to read is the *vertical* relationship — which
 * layer sits under which, where a shaft drops from one to the next, and which
 * shafts come out on the surface. Entrances are drawn in amber for that reason:
 * an entrance is the difference between a cave and a sealed void, and it is the
 * single property the flat generator could not express at all.
 */
std::vector<Vertex> buildCaveSystemMesh(const TerrainData &world, float topDepth, float layerSpacing)
{
    std::vector<Vertex> vertices;
    const procgen::CaveSystem &system = world.caveSystem;
    if (system.layerCount == 0u || world.height.empty())
        return vertices;

    const float halfW = static_cast<float>(world.height.width()) * 0.5f;
    const float halfD = static_cast<float>(world.height.depth()) * 0.5f;

    const auto quad = [&vertices](float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                                  float cz, float dx, float dy, float dz, float nx, float ny, float nz,
                                  const Rgb &colour) {
        const Vertex a{ax, ay, az, nx, ny, nz, colour};
        const Vertex b{bx, by, bz, nx, ny, nz, colour};
        const Vertex c{cx, cy, cz, nx, ny, nz, colour};
        const Vertex d{dx, dy, dz, nx, ny, nz, colour};
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
        vertices.push_back(a);
        vertices.push_back(c);
        vertices.push_back(d);
    };

    for (core::u32 layer = 0u; layer < system.layerCount; ++layer)
    {
        const procgen::DungeonMap &plan = system.layer[layer];
        if (plan.empty())
            continue;

        // Deeper layers darken. Reading depth off a colour is crude and it works:
        // three identically lit floors stacked in a wireframe are unreadable.
        const float shade = 1.0f - 0.22f * static_cast<float>(layer);
        const Rgb floorColour{0.52f * shade, 0.44f * shade, 0.36f * shade};
        const Rgb wallColour{0.30f * shade, 0.26f * shade, 0.24f * shade};

        const auto rock = [&plan](core::i32 x, core::i32 z) {
            if (!plan.contains(x, z))
                return true;
            return !procgen::isWalkable(plan.at(static_cast<core::u32>(x), static_cast<core::u32>(z)));
        };

        for (core::u32 z = 0u; z < plan.depth(); ++z)
            for (core::u32 x = 0u; x < plan.width(); ++x)
            {
                if (rock(static_cast<core::i32>(x), static_cast<core::i32>(z)))
                    continue;

                const float ground =
                    world.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
                const float floorY = ground - topDepth - static_cast<float>(layer) * layerSpacing;
                const float ceilY = floorY + 1.6f;
                const float x0 = static_cast<float>(x) - halfW;
                const float x1 = x0 + 1.0f;
                const float z0 = static_cast<float>(z) - halfD;
                const float z1 = z0 + 1.0f;

                quad(x0, floorY, z0, x1, floorY, z0, x1, floorY, z1, x0, floorY, z1, 0.0f, 1.0f, 0.0f, floorColour);
                if (rock(static_cast<core::i32>(x) + 1, static_cast<core::i32>(z)))
                    quad(x1, floorY, z0, x1, ceilY, z0, x1, ceilY, z1, x1, floorY, z1, 1.0f, 0.0f, 0.0f, wallColour);
                if (rock(static_cast<core::i32>(x) - 1, static_cast<core::i32>(z)))
                    quad(x0, floorY, z0, x0, ceilY, z0, x0, ceilY, z1, x0, floorY, z1, -1.0f, 0.0f, 0.0f, wallColour);
                if (rock(static_cast<core::i32>(x), static_cast<core::i32>(z) + 1))
                    quad(x0, floorY, z1, x0, ceilY, z1, x1, ceilY, z1, x1, floorY, z1, 0.0f, 0.0f, 1.0f, wallColour);
                if (rock(static_cast<core::i32>(x), static_cast<core::i32>(z) - 1))
                    quad(x0, floorY, z0, x0, ceilY, z0, x1, ceilY, z0, x1, floorY, z0, 0.0f, 0.0f, -1.0f, wallColour);
            }
    }

    // The shafts, as square columns joining the two floors they connect. A surface
    // shaft runs all the way up to the ground and is drawn in amber.
    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const procgen::CaveShaft &shaft = system.shafts[i];
        const float ground =
            world.height.clamped(static_cast<core::i32>(shaft.x), static_cast<core::i32>(shaft.z)).toFloat();
        const float upper =
            shaft.surface ? ground + 0.4f : ground - topDepth - static_cast<float>(shaft.upperLayer) * layerSpacing;
        const float lower =
            ground - topDepth - static_cast<float>(shaft.surface ? shaft.upperLayer : shaft.lowerLayer) * layerSpacing;
        const Rgb colour = shaft.surface ? Rgb{0.95f, 0.62f, 0.14f} : Rgb{0.40f, 0.34f, 0.30f};

        const float x0 = static_cast<float>(shaft.x) - halfW + 0.25f;
        const float x1 = x0 + 0.5f;
        const float z0 = static_cast<float>(shaft.z) - halfD + 0.25f;
        const float z1 = z0 + 0.5f;
        quad(x0, lower, z0, x0, upper, z0, x1, upper, z0, x1, lower, z0, 0.0f, 0.0f, -1.0f, colour);
        quad(x1, lower, z1, x1, upper, z1, x0, upper, z1, x0, lower, z1, 0.0f, 0.0f, 1.0f, colour);
        quad(x1, lower, z0, x1, upper, z0, x1, upper, z1, x1, lower, z1, 1.0f, 0.0f, 0.0f, colour);
        quad(x0, lower, z1, x0, upper, z1, x0, upper, z0, x0, lower, z0, -1.0f, 0.0f, 0.0f, colour);
    }
    return vertices;
}

/**
 * @brief Meshes a liminal sector as walls on a flat floor, tinted by zone.
 *
 * Flat on purpose: the whole effect depends on the ceiling being uniform and the
 * light being even, and a liminal space that follows terrain stops being one.
 */
std::vector<Vertex> buildLiminalMesh(const procgen::LiminalSpace &space)
{
    std::vector<Vertex> vertices;
    if (space.map.empty())
        return vertices;

    const core::u32 width = space.map.width();
    const core::u32 depth = space.map.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;
    const float wallHeight = 2.6f;

    const auto zoneColour = [](procgen::LiminalZone zone) -> Rgb {
        switch (zone)
        {
        case procgen::LiminalZone::Corridor: return {0.76f, 0.72f, 0.55f};
        case procgen::LiminalZone::Office: return {0.82f, 0.79f, 0.62f};
        case procgen::LiminalZone::Hall: return {0.70f, 0.68f, 0.58f};
        case procgen::LiminalZone::Pool: return {0.55f, 0.72f, 0.74f};
        case procgen::LiminalZone::Count: break;
        }
        return {0.7f, 0.7f, 0.7f};
    };

    const auto quad = [&vertices](float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                                  float cz, float dx, float dy, float dz, float nx, float ny, float nz,
                                  const Rgb &colour) {
        const Vertex a{ax, ay, az, nx, ny, nz, colour};
        const Vertex b{bx, by, bz, nx, ny, nz, colour};
        const Vertex c{cx, cy, cz, nx, ny, nz, colour};
        const Vertex d{dx, dy, dz, nx, ny, nz, colour};
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
        vertices.push_back(a);
        vertices.push_back(c);
        vertices.push_back(d);
    };

    const auto rock = [&space](core::i32 x, core::i32 z) {
        if (!space.map.contains(x, z))
            return true;
        return !procgen::isWalkable(space.map.at(static_cast<core::u32>(x), static_cast<core::u32>(z)));
    };

    for (core::u32 z = 0u; z < depth; ++z)
        for (core::u32 x = 0u; x < width; ++x)
        {
            if (rock(static_cast<core::i32>(x), static_cast<core::i32>(z)))
                continue;
            const Rgb floorColour = zoneColour(space.zones.at(x, z));
            const Rgb wallColour{floorColour.r * 0.78f, floorColour.g * 0.78f, floorColour.b * 0.70f};

            const float x0 = static_cast<float>(x) - halfW;
            const float x1 = x0 + 1.0f;
            const float z0 = static_cast<float>(z) - halfD;
            const float z1 = z0 + 1.0f;

            quad(x0, 0.0f, z0, x1, 0.0f, z0, x1, 0.0f, z1, x0, 0.0f, z1, 0.0f, 1.0f, 0.0f, floorColour);
            if (rock(static_cast<core::i32>(x) + 1, static_cast<core::i32>(z)))
                quad(x1, 0.0f, z0, x1, wallHeight, z0, x1, wallHeight, z1, x1, 0.0f, z1, 1.0f, 0.0f, 0.0f, wallColour);
            if (rock(static_cast<core::i32>(x) - 1, static_cast<core::i32>(z)))
                quad(x0, 0.0f, z0, x0, wallHeight, z0, x0, wallHeight, z1, x0, 0.0f, z1, -1.0f, 0.0f, 0.0f, wallColour);
            if (rock(static_cast<core::i32>(x), static_cast<core::i32>(z) + 1))
                quad(x0, 0.0f, z1, x0, wallHeight, z1, x1, wallHeight, z1, x1, 0.0f, z1, 0.0f, 0.0f, 1.0f, wallColour);
            if (rock(static_cast<core::i32>(x), static_cast<core::i32>(z) - 1))
                quad(x0, 0.0f, z0, x0, wallHeight, z0, x1, wallHeight, z0, x1, 0.0f, z0, 0.0f, 0.0f, -1.0f, wallColour);
        }
    return vertices;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

struct Camera {
    float yaw{-0.7f};
    float pitch{0.85f};
    float distance{170.0f};
    float height{0.0f};
};

void applyPerspective(int width, int height)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const double aspect = height > 0 ? static_cast<double>(width) / static_cast<double>(height) : 1.0;
    // glFrustum rather than gluPerspective: GLU is not installed, and the two
    // lines below are the whole of what it would have done.
    const double nearPlane = 0.6;
    const double farPlane = 3000.0;
    const double top = nearPlane * 0.5773502692; // tan(30 degrees), i.e. a 60-degree field
    glFrustum(-top * aspect, top * aspect, -top, top, nearPlane, farPlane);
    glMatrixMode(GL_MODELVIEW);
}

void drawMesh(const std::vector<Vertex> &mesh, bool wireframe)
{
    if (mesh.empty())
        return;
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    // A single directional light, applied by hand. Fixed-function lighting would
    // work too, but doing the dot product here keeps the colours exactly the ones
    // chosen above rather than whatever the pipeline's material model implies.
    // One directional key light plus a dim fill from the opposite side, applied by
    // hand so the palette stays the one chosen above rather than whatever the
    // fixed-function material model would make of it.
    //
    // Two-sided: the lambert term is taken absolute. A heightfield's normals all
    // point up, so a light from above lights the top correctly — but the underside
    // would be pure ambient and read as a black void when the camera dips below the
    // horizon. Lighting both faces costs nothing and removes that cliff.
    const float keyX = 0.45f;
    const float keyY = 0.78f;
    const float keyZ = 0.44f;
    const float fillX = -0.55f;
    const float fillY = 0.35f;
    const float fillZ = -0.75f;

    glBegin(GL_TRIANGLES);
    for (const Vertex &v : mesh)
    {
        float key = v.nx * keyX + v.ny * keyY + v.nz * keyZ;
        if (key < 0.0f)
            key = -key;
        float fill = v.nx * fillX + v.ny * fillY + v.nz * fillZ;
        if (fill < 0.0f)
            fill = -fill;

        float shade = 0.38f + 0.72f * key + 0.18f * fill;
        if (shade > 1.35f)
            shade = 1.35f;

        float r = v.colour.r * shade;
        float g = v.colour.g * shade;
        float b = v.colour.b * shade;
        if (r > 1.0f)
            r = 1.0f;
        if (g > 1.0f)
            g = 1.0f;
        if (b > 1.0f)
            b = 1.0f;
        glColor3f(r, g, b);
        glVertex3f(v.x, v.y, v.z);
    }
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

/**
 * @class MeshList
 * @brief A static mesh compiled into a display list.
 *
 * Immediate mode re-sends every vertex across the driver boundary on every frame, and
 * the three static meshes here come to something like a hundred thousand triangles —
 * paid again sixty times a second for geometry that has not changed. A display list is
 * the fixed-function answer: submit once at build time, replay with a single call.
 *
 * Kept as a small owning type because a leaked list is a leak the driver never reports:
 * regenerating the world rebuilds all three meshes, so without this the lists would
 * accumulate one per keypress.
 */
class MeshList {
public:
    MeshList() = default;
    MeshList(const MeshList &) = delete;
    MeshList &operator=(const MeshList &) = delete;
    ~MeshList() { release(); }

    void release()
    {
        if (_list != 0u)
        {
            glDeleteLists(_list, 1);
            _list = 0u;
        }
        _triangles = 0u;
    }

    /// Compiles @p mesh, replacing whatever was here.
    void compile(const std::vector<Vertex> &mesh, void (*draw)(const std::vector<Vertex> &, bool), bool wireframe)
    {
        release();
        if (mesh.empty())
            return;
        _list = glGenLists(1);
        if (_list == 0u)
            return;
        glNewList(_list, GL_COMPILE);
        draw(mesh, wireframe);
        glEndList();
        _triangles = static_cast<core::u32>(mesh.size() / 3u);
    }

    void call() const
    {
        if (_list != 0u)
            glCallList(_list);
    }

    [[nodiscard]] core::u32 triangles() const noexcept { return _triangles; }
    [[nodiscard]] bool empty() const noexcept { return _list == 0u; }

private:
    GLuint _list{0u};
    core::u32 _triangles{0u};
};

/// The same mesh, drawn translucent — the ghosted ground of the cutaway view.
void drawMeshGhosted(const std::vector<Vertex> &mesh, bool /*wireframe*/)
{
    if (mesh.empty())
        return;
    const float alpha = 0.34f;

    const float keyX = 0.45f;
    const float keyY = 0.78f;
    const float keyZ = 0.44f;

    glBegin(GL_TRIANGLES);
    for (const Vertex &v : mesh)
    {
        float key = v.nx * keyX + v.ny * keyY + v.nz * keyZ;
        if (key < 0.0f)
            key = -key;
        const float shade = 0.45f + 0.65f * key;
        glColor4f(v.colour.r * shade, v.colour.g * shade, v.colour.b * shade, alpha);
        glVertex3f(v.x, v.y, v.z);
    }
    glEnd();
}

void drawWaterPlane(const TerrainData &world, float level)
{
    const float halfW = static_cast<float>(world.height.width()) * 0.5f;
    const float halfD = static_cast<float>(world.height.depth()) * 0.5f;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.12f, 0.34f, 0.62f, 0.55f);
    glBegin(GL_QUADS);
    glVertex3f(-halfW, level, -halfD);
    glVertex3f(halfW, level, -halfD);
    glVertex3f(halfW, level, halfD);
    glVertex3f(-halfW, level, halfD);
    glEnd();
    glDisable(GL_BLEND);
}

/**
 * @class LivingLayer
 * @brief The ai/ and ecology/ modules, running on top of the generated world.
 *
 * These two modules were folded and never looked at, which is the specific gap
 * this closes: a signature proves two machines agree about a number, and says
 * nothing about whether an ant colony converges on a route or a food web settles
 * instead of oscillating apart. Both of those are shapes over time, and a shape
 * is the one thing a fold cannot report.
 *
 * Three things run here, all of them on the world the generator just produced:
 *
 *  - a **stigmergy field** over the terrain footprint, with water and cliffs
 *    entered as obstacles, so the pheromone genuinely flows around the map's
 *    geometry rather than through it;
 *  - **agents** walking that field from a nest, depositing behind them — the
 *    trail that emerges is the whole claim of the module, and it either forms or
 *    it does not;
 *  - a **trophic web** stepping once per fixed tick, reported in the HUD.
 *
 * @warning Nothing here feeds back into the world. The layer reads the terrain
 *          and writes only its own field, so it cannot perturb the physics the
 *          rest of the viewer is showing.
 */
class LivingLayer {
public:
    /**
     * @struct Creature
     * @brief One animal with a body on the map.
     *
     * A boid for the movement, a genome for the body, and an identifier for the
     * temperament — which is *derived* from the id and never stored, so a herd of
     * four hundred costs four hundred integers rather than four hundred
     * personality structs. That is the module's own rule, and honouring it here is
     * what makes the herd affordable at all.
     */
    /**
     * @struct Plant
     * @brief One scattered plant, as the ecology sees it.
     *
     * The producer level of a food web is usually a number nobody can point at. It
     * does not have to be: the scatter already put several hundred plants on this
     * map, at known positions, and those are exactly what a herbivore eats. So the
     * producer population is not a parallel quantity that happens to be called
     * "plants" — it IS the standing vegetation, counted. Graze a valley bare and the
     * number in the HUD falls because the trees are gone, not because a differential
     * equation said so.
     */
    struct Plant {
        core::u32 cellX{0u};
        core::u32 cellZ{0u};
        core::u32 regrowth{0u}; ///< Ticks until it comes back; 0 when standing.
        bool standing{true};
    };

    struct Creature {
        ai::Boid body{};
        ecology::Genome genome{};
        core::u32 id{0u};
        core::u32 species{0u};                       ///< 0 herbivore, 1 predator.
        math::Fixed32 heading{math::Fixed32::one()}; ///< Unit facing, X.
        math::Fixed32 headingZ{};                    ///< Unit facing, Z.
        bool alive{true};
    };

    /// Rebuilds the layer against a freshly generated world.
    void reset(const TerrainData &world, core::u32 seed, const lpl::pmr::vector<math::Vec3<math::Fixed32>> &props)
    {
        _width = world.height.width();
        _depth = world.height.depth();
        _field = ai::StigmergyField{_width, _depth, 2u};
        if (_width == 0u || _depth == 0u)
            return;

        // Obstacles: below the water line, or too steep to walk. Marking them is
        // what makes the field flow around the map instead of through it — the
        // diffusion step refuses to cross a blocked cell in either direction.
        // ONE mask, three consumers: the field diffuses around it, the herd refuses
        // to walk into it, and the spawn refuses to start inside it. Three separate
        // notions of "blocked" is how a creature ends up standing in a wall that the
        // pheromone flows around.
        _blocked = procgen::Grid<core::u8>{_width, _depth, 0u};
        for (core::u32 z = 0u; z < _depth; ++z)
            for (core::u32 x = 0u; x < _width; ++x)
            {
                const bool drowned = world.height.at(x, z).toFloat() < world.seaLevel;
                const bool steep = procgen::slopeAt(world.height, x, z).toFloat() > 2.2f;
                _blocked.at(x, z) = (drowned || steep) ? 1u : 0u;
            }

        // Buildings. A plot is a footprint the grammar raised into a solid, so it is
        // as impassable as a cliff; a plaza and a street are not.
        if (!world.settlement.empty())
            for (core::u32 z = 0u; z < _depth && z < world.settlement.depth(); ++z)
                for (core::u32 x = 0u; x < _width && x < world.settlement.width(); ++x)
                    if (world.settlement.at(x, z) == procgen::SettlementCell::Plot)
                        _blocked.at(x, z) = 1u;

        // Trees and rocks: the scatter's own entities, at the resolution the herd
        // walks on. The physics solver already stops the BOULDERS against them; this
        // is the same obstacle set expressed where the creatures can read it.
        const float halfWidthCells = static_cast<float>(_width) * 0.5f;
        const float halfDepthCells = static_cast<float>(_depth) * 0.5f;
        for (core::usize i = 0u; i < props.size(); ++i)
        {
            const core::i32 px = static_cast<core::i32>(props[i].x.toFloat() + halfWidthCells);
            const core::i32 pz = static_cast<core::i32>(props[i].z.toFloat() + halfDepthCells);
            if (_blocked.contains(px, pz))
                _blocked.at(static_cast<core::u32>(px), static_cast<core::u32>(pz)) = 1u;
        }

        _field.setObstacles(_blocked);

        core::u32 blockedCells = 0u;
        for (core::u32 i = 0u; i < _blocked.cellCount(); ++i)
            if (_blocked[i] != 0u)
                ++blockedCells;
        std::printf("  living: %u of %u cells blocked (%.0f%%)\n", blockedCells, _blocked.cellCount(),
                    100.0 * static_cast<double>(blockedCells) / static_cast<double>(_blocked.cellCount()));

        // The nest goes where the town is, when there is one: a colony foraging out
        // of a settlement reads as something happening in the world rather than as a
        // demo running beside it.
        _nestX = _width / 2u;
        _nestZ = _depth / 2u;
        if (!world.settlement.empty())
            for (core::u32 z = 0u; z < _depth; ++z)
                for (core::u32 x = 0u; x < _width; ++x)
                    if (world.settlement.at(x, z) == procgen::SettlementCell::Plaza)
                    {
                        _nestX = x;
                        _nestZ = z;
                        z = _depth;
                        break;
                    }

        const core::u32 nest = _nestZ * _width + _nestX;
        ai::seedPheromoneField(_field, 0u, &nest, 1u, math::Fixed32::fromInt(60));

        _agentX.clear();
        _agentZ.clear();
        procgen::Random spawn{seed ^ 0xA47C0011u};
        for (core::u32 i = 0u; i < kAgents; ++i)
        {
            _agentX.push_back(_nestX);
            _agentZ.push_back(_nestZ);
            (void) spawn.next();
        }
        _stream = seed ^ 0xA57E0022u;

        // A four-level web with the same shape the parity recipe uses, so what the
        // HUD reports is the same dynamics the gate folds.
        _web = ecology::TrophicWeb{};
        ecology::SpeciesParams grass{};
        grass.level = ecology::TrophicLevel::Producer;
        grass.capacity = math::Fixed32::fromInt(1000);
        const core::u32 producer = _web.add(grass, math::Fixed32::fromInt(800), ecology::Species::kNoPrey);
        ecology::SpeciesParams herbivore{};
        herbivore.level = ecology::TrophicLevel::Primary;
        herbivore.capacity = math::Fixed32::fromInt(200);
        const core::u32 primary = _web.add(herbivore, math::Fixed32::fromInt(120), producer);
        ecology::SpeciesParams predator{};
        predator.level = ecology::TrophicLevel::Secondary;
        predator.capacity = math::Fixed32::fromInt(40);
        _web.add(predator, math::Fixed32::fromInt(24), primary);
        _ticks = 0u;

        // The bodies. Head counts come from the web rather than from a constant:
        // what is on screen is then the population the model says exists, and a
        // collapse is something you watch happen rather than read in a number.
        _creatures.clear();
        _heredity = seed ^ 0x5EED0033u;
        procgen::Random stock{seed ^ 0x11FE0044u};
        // The vegetation, taken from what the scatter actually placed. Capacity is
        // the map's own carrying capacity in the literal sense: how many plants fit
        // on it, which is a fact about the terrain rather than a tuning constant.
        _plants.clear();
        const float halfWidthGrid = static_cast<float>(_width) * 0.5f;
        const float halfDepthGrid = static_cast<float>(_depth) * 0.5f;
        for (core::usize i = 0u; i < props.size(); ++i)
        {
            const core::i32 px = static_cast<core::i32>(props[i].x.toFloat() + halfWidthGrid);
            const core::i32 pz = static_cast<core::i32>(props[i].z.toFloat() + halfDepthGrid);
            Plant plant;
            plant.cellX = px < 0 ? 0u : static_cast<core::u32>(px);
            plant.cellZ = pz < 0 ? 0u : static_cast<core::u32>(pz);
            _plants.push_back(plant);
        }
        if (!_plants.empty())
        {
            _web.species[0].params.capacity = math::Fixed32::fromInt(static_cast<core::i32>(_plants.size()));
            _web.species[0].population = math::Fixed32::fromInt(static_cast<core::i32>(_plants.size()));
        }
        _grazed = 0u;
        _regrown = 0u;
        _floraDirty = false;

        const core::u32 herbivores = realisedCount(_web.species[1].population, _web.species[1].params.refuge);
        const core::u32 predators = realisedCount(_web.species[2].population, _web.species[2].params.refuge);
        for (core::u32 i = 0u; i < herbivores + predators; ++i)
            spawnCreature(stock, i < herbivores ? 0u : 1u, stock.next());
    }

    /**
     * @brief How many bodies a head count is worth on screen.
     *
     * @warning The obvious ratio is wrong, and it was measured wrong before it was
     *          written right. One body per ten head reads as a sensible abstraction
     *          until you look at what the web actually settles on: this world's
     *          herbivores live around seven and its predators around four, so the
     *          division floored to zero and the map stayed empty while the HUD
     *          cheerfully reported a working ecosystem. The bug was not in the
     *          spawning, the flocking or the drawing — all three were correct and
     *          all three ran on an empty list.
     *
     *          So: one body per two head, and a floor of three for any species the
     *          model still considers alive. A species above its refuge exists, and
     *          a world where existing is invisible is not showing the model.
     */
    [[nodiscard]] static core::u32 realisedCount(math::Fixed32 population, math::Fixed32 refuge) noexcept
    {
        if (population <= refuge)
            return 0u;
        core::i32 wanted = population.toInt() / 2;
        if (wanted < 3)
            wanted = 3;
        return static_cast<core::u32>(wanted) > kMaxBodies ? kMaxBodies : static_cast<core::u32>(wanted);
    }

    /// Places one animal on walkable ground, with a genome drawn from the stock.
    void spawnCreature(procgen::Random &random, core::u32 species, core::u32 id)
    {
        if (_width == 0u)
            return;
        Creature creature;
        creature.id = id;
        creature.species = species;

        // Mutation off a species archetype, so a herd is a spread rather than a
        // row of clones — the same draw the heredity module makes between
        // generations, used once at founding.
        ecology::Genome archetype{};
        if (species == 1u)
        {
            archetype.size = math::Fixed32::fromFloat(1.4f);
            archetype.maxSpeed = math::Fixed32::fromFloat(5.5f);
            archetype.strength = math::Fixed32::fromFloat(8.0f);
        }
        else
        {
            archetype.size = math::Fixed32::fromFloat(0.9f);
            archetype.maxSpeed = math::Fixed32::fromFloat(4.0f);
        }
        creature.genome = ecology::mutate(archetype, 8u, 0.18f, _heredity);

        // Somewhere it could actually stand: dropping a herd into the sea and
        // letting the flocking rules sort it out produces a very confident-looking
        // shoal of deer.
        for (core::u32 attempt = 0u; attempt < 32u; ++attempt)
        {
            const core::u32 x = random.below(_width);
            const core::u32 z = random.below(_depth);
            if (_blocked.at(x, z) != 0u)
                continue;
            creature.body.x = math::Fixed32::fromInt(static_cast<core::i32>(x)) -
                              math::Fixed32::fromInt(static_cast<core::i32>(_width / 2u));
            creature.body.z = math::Fixed32::fromInt(static_cast<core::i32>(z)) -
                              math::Fixed32::fromInt(static_cast<core::i32>(_depth / 2u));
            creature.body.vx = random.unit() - math::Fixed32::half();
            creature.body.vz = random.unit() - math::Fixed32::half();
            _creatures.push_back(creature);
            return;
        }
    }

    /**
     * @brief Moves the herd: flock, follow the scent, keep off the water.
     *
     * Three rules, and the order matters. Flocking first, because it is the only
     * one that reads its neighbours and therefore needs a coherent snapshot; then
     * the scent, which is what makes the herd go SOMEWHERE rather than mill about;
     * then the terrain, which has the last word — the same "propose, then dispose"
     * split the road grammar uses.
     */
    void stepCreatures(const TerrainData &world)
    {
        if (_creatures.empty() || world.height.empty())
            return;

        // Per species, so a predator flocks with predators and not with its lunch.
        for (core::u32 species = 0u; species < 2u; ++species)
        {
            _flockScratch.clear();
            for (core::u32 i = 0u; i < _creatures.size(); ++i)
                if (_creatures[i].species == species && _creatures[i].alive)
                    _flockScratch.push_back(_creatures[i].body);
            if (_flockScratch.empty())
                continue;

            ai::BoidParams params;
            // A predator holds a looser formation and a wider watch than a herd
            // animal. One parameter set for both would make the pack move like a
            // shoal, which is exactly what it must not look like.
            params.separationWeight = species == 1u ? 1.1f : 0.9f;
            params.alignmentWeight = species == 1u ? 0.5f : 0.8f;
            params.cohesionWeight = species == 1u ? 0.25f : 0.5f;
            params.neighbourRadius = math::Fixed32::fromInt(species == 1u ? 12 : 7);
            params.separationRadius = math::Fixed32::fromFloat(species == 1u ? 2.5f : 1.6f);
            ai::stepBoids(&_flockScratch[0], static_cast<core::u32>(_flockScratch.size()), params, kFixedStep);

            // Only the velocity is taken back. The flock decides where an animal
            // WANTS to go; where it may actually stand is the terrain's business,
            // and letting the boid integrator write the position straight back would
            // put bodies inside rock before any check downstream could refuse it.
            core::u32 cursor = 0u;
            for (core::u32 i = 0u; i < _creatures.size(); ++i)
                if (_creatures[i].species == species && _creatures[i].alive)
                {
                    _creatures[i].body.vx = _flockScratch[cursor].vx;
                    _creatures[i].body.vz = _flockScratch[cursor].vz;
                    ++cursor;
                }
        }

        const float halfW = static_cast<float>(_width) * 0.5f;
        const float halfD = static_cast<float>(_depth) * 0.5f;

        for (core::u32 i = 0u; i < _creatures.size(); ++i)
        {
            Creature &creature = _creatures[i];
            if (!creature.alive)
                continue;

            const ai::PersonalityTraits traits = ai::personalityOf(creature.id, creature.species);

            // Off the map, or standing in something: walk back toward the centre.
            //
            // This is where most of the "refused step" count came from, and it was
            // not the herd bumping into scenery — it was the handful of animals the
            // flocking rules had pushed over the border. Outside the grid EVERY
            // direction is unwalkable, so an escapee refuses both axes on every tick
            // for the rest of the run: a few bodies generating thousands of refusals
            // and looking, from inside the counter, like a herd in constant collision.
            if (!walkable(creature.body.x, creature.body.z, halfW, halfD))
            {
                const math::Fixed32 towardX = creature.body.x.raw() > 0 ? -math::Fixed32::one() : math::Fixed32::one();
                const math::Fixed32 towardZ = creature.body.z.raw() > 0 ? -math::Fixed32::one() : math::Fixed32::one();
                creature.body.x = creature.body.x + towardX;
                creature.body.z = creature.body.z + towardZ;
                creature.heading = towardX;
                creature.headingZ = towardZ;
                creature.body.vx = towardX;
                creature.body.vz = towardZ;
                {
                    const core::i32 px = static_cast<core::i32>(creature.body.x.toFloat() + halfW);
                    const core::i32 pz = static_cast<core::i32>(creature.body.z.toFloat() + halfD);
                    if (!_blocked.contains(px, pz))
                        ++_strayOutside;
                    else
                        ++_strayBlocked;
                }
                ++_strays;
                continue;
            }

            const core::i32 gx = static_cast<core::i32>(creature.body.x.toFloat() + halfW);
            const core::i32 gz = static_cast<core::i32>(creature.body.z.toFloat() + halfD);
            if (world.height.contains(gx, gz))
            {
                const core::u32 x = static_cast<core::u32>(gx);
                const core::u32 z = static_cast<core::u32>(gz);

                // The scent channel is the herd's only shared memory: a herbivore
                // climbs it toward grazing, a predator climbs the same field because
                // it leads to the herd. One substrate, two readings — which is the
                // claim the stigmergy module makes and the reason it is one class.
                const core::u32 direction = _field.gradientDirection(1u, x, z, true);
                if (direction != ai::StigmergyField::kNoDirection)
                {
                    const math::Fixed32 pull = math::Fixed32::fromFloat(creature.species == 1u ? 0.07f : 0.05f) *
                                               (math::Fixed32::half() + traits.energy);
                    creature.body.vx =
                        creature.body.vx + math::Fixed32::fromInt(procgen::kNeighbor8X[direction]) * pull;
                    creature.body.vz =
                        creature.body.vz + math::Fixed32::fromInt(procgen::kNeighbor8Z[direction]) * pull;
                }

                // A herbivore leaves its own scent where it grazes; that is what a
                // predator ends up following. Nothing tells the predator where the
                // herd is — the map does.
                if (creature.species == 0u)
                {
                    _field.deposit(1u, x, z, math::Fixed32::fromFloat(0.6f));
                    graze(x, z);
                }
            }

            // ── How far that is worth, in a second ──────────────────────────
            //
            // The boid rules give a DIRECTION and a normalised speed, not a
            // displacement. Adding the velocity straight to the position advances a
            // body by up to a full cell per fixed tick — sixty cells a second, which
            // is what a herd of deer moving like tracer fire looks like. The step is
            // the genome's speed in world units per second times the tick's own
            // duration, so the herd moves at four cells a second whatever the tick
            // rate is, and a faster machine does not make the animals faster.
            // An animal WALKS. The boid rules produce a steering vector whose length
            // means nothing in particular — separation and cohesion nearly cancel in
            // a settled herd, so its magnitude collapses toward zero and a body that
            // multiplies by it simply stops. Measured, the herd was moving at a tenth
            // of a cell per second and grinding thousands of refused steps against
            // the scenery. What the vector carries is a HEADING; the speed is the
            // genome's. Normalising separates the two, which is the only reading
            // under which "maxSpeed" is a property of the animal at all.
            const math::Fixed32 lengthSquared =
                creature.body.vx * creature.body.vx + creature.body.vz * creature.body.vz;
            math::Fixed32 headingX = creature.body.vx;
            math::Fixed32 headingZ = creature.body.vz;
            const math::Fixed32 length = procgen::fixedSqrt(lengthSquared);
            if (length.raw() > 256)
            {
                headingX = creature.body.vx / length;
                headingZ = creature.body.vz / length;
                creature.heading = headingX;
                creature.headingZ = headingZ;
            }
            else
            {
                // Standing still is a valid boid state and not a valid animal one:
                // keep the last heading rather than freezing on the spot.
                headingX = creature.heading;
                headingZ = creature.headingZ;
            }

            // Avoidance, before the move rather than after refusing it. Steering
            // around an obstacle and being stopped by one look the same in a single
            // frame and nothing alike over a second: the herd was refusing two steps
            // in three and grinding along every treeline it met. Picking the free
            // neighbour closest to where the animal was already going is the whole
            // rule — an animal walks around a rock, it does not walk into it and
            // reconsider.
            if (!walkableAhead(creature, headingX, headingZ, halfW, halfD))
            {
                math::Fixed32 bestX = headingX;
                math::Fixed32 bestZ = headingZ;
                math::Fixed32 bestDot = math::Fixed32::fromInt(-2);
                for (core::u32 n = 0u; n < 8u; ++n)
                {
                    const math::Fixed32 candidateX = math::Fixed32::fromInt(procgen::kNeighbor8X[n]) *
                                                     (n < 4u ? math::Fixed32::one() : procgen::kInvSqrt2);
                    const math::Fixed32 candidateZ = math::Fixed32::fromInt(procgen::kNeighbor8Z[n]) *
                                                     (n < 4u ? math::Fixed32::one() : procgen::kInvSqrt2);
                    if (!walkableAhead(creature, candidateX, candidateZ, halfW, halfD))
                        continue;
                    // Closest to the current heading: turning is cheap, reversing is
                    // not, and a creature that picks the first free direction in
                    // array order makes every herd drift east.
                    const math::Fixed32 dot = candidateX * headingX + candidateZ * headingZ;
                    if (dot > bestDot)
                    {
                        bestDot = dot;
                        bestX = candidateX;
                        bestZ = candidateZ;
                    }
                }
                headingX = bestX;
                headingZ = bestZ;
                creature.heading = bestX;
                creature.headingZ = bestZ;
            }

            const math::Fixed32 pace = creature.genome.maxSpeed * kFixedStep *
                                       (math::Fixed32::fromFloat(0.7f) + traits.energy * math::Fixed32::half());
            const math::Fixed32 stepX = headingX * pace;
            const math::Fixed32 stepZ = headingZ * pace;

            // The terrain disposes, one axis at a time: refusing BOTH axes on a
            // single blocked corner is what made the movement look random. An animal
            // walking into a wall slides along it — reversing its velocity instead
            // sends it back the way it came, and with a herd of those the whole flock
            // shudders in place.
            const math::Fixed32 tryX = creature.body.x + stepX;
            const math::Fixed32 tryZ = creature.body.z + stepZ;
            // The DIAGONAL is checked too, and that is not pedantry: testing the two
            // axes separately and then moving along both walks the corner between
            // two free cells into the blocked one they share. That corner was the
            // last source of strays — a creature stepping diagonally past a tree
            // ended up standing inside it, which the next tick reads as "not on
            // walkable ground" and hands to the containment rule.
            const bool freeX = walkable(tryX, creature.body.z, halfW, halfD);
            const bool freeZ = walkable(creature.body.x, tryZ, halfW, halfD);
            const bool freeDiagonal = freeX && freeZ && walkable(tryX, tryZ, halfW, halfD);
            if (freeDiagonal)
            {
                creature.body.x = tryX;
                creature.body.z = tryZ;
            }
            else if (freeX)
                creature.body.x = tryX;
            else if (freeZ)
                creature.body.z = tryZ;
            if (!freeX && !freeZ)
            {
                // Cornered: turn around rather than vibrate against the obstacle. The
                // HEADING is reversed, not the boid velocity — zeroing that was the
                // other half of why the herd stalled, since it destroys the very
                // state the flocking rules accumulate.
                creature.heading = -headingX;
                creature.headingZ = -headingZ;
                creature.body.x = creature.body.x - stepX;
                creature.body.z = creature.body.z - stepZ;
                ++_refusals;
            }
        }
    }

    /**
     * @brief Records how far the herd actually moved over the last second.
     *
     * Distance travelled, not the speed it was asked to travel at: a body that is
     * pinned against a rock has a perfectly healthy velocity and goes nowhere, and
     * only one of those two numbers would have caught it.
     */
    void measureSpeed()
    {
        if (_creatures.empty())
        {
            _measuredSpeed = 0.0;
            return;
        }
        if (_speedMarks.size() != _creatures.size())
        {
            _speedMarks.clear();
            for (core::u32 i = 0u; i < _creatures.size(); ++i)
                _speedMarks.push_back(_creatures[i].body);
            return;
        }
        if (_ticks % 60u != 0u)
            return;

        double total = 0.0;
        core::u32 counted = 0u;
        for (core::u32 i = 0u; i < _creatures.size(); ++i)
        {
            if (!_creatures[i].alive)
                continue;
            const double dx = _creatures[i].body.x.toFloat() - _speedMarks[i].x.toFloat();
            const double dz = _creatures[i].body.z.toFloat() - _speedMarks[i].z.toFloat();
            total += std::sqrt(dx * dx + dz * dz);
            ++counted;
        }
        _measuredSpeed = counted == 0u ? 0.0 : total / static_cast<double>(counted);
        for (core::u32 i = 0u; i < _creatures.size(); ++i)
            _speedMarks[i] = _creatures[i].body;
    }

    /**
     * @brief A grazer eats whatever is standing on its cell.
     *
     * One plant at a time and only where the animal actually is — no radius, no
     * probability. A herd therefore leaves a visible trail of cropped ground
     * behind it, which is the point: the pressure on the producer level has a
     * SHAPE on the map, and where the herd has been is readable without any
     * overlay.
     */
    void graze(core::u32 x, core::u32 z)
    {
        // Reach, not exact cell. Measured with an exact match: sixty grazers on a
        // 128x128 map with 360 plants ate NOTHING over a minute — two point sets
        // that sparse almost never coincide, so the producer level sat at 360 while
        // the herd walked over the trees. An animal's mouth has an extent; one cell
        // of reach is the smallest honest version of that.
        for (core::u32 i = 0u; i < _plants.size(); ++i)
        {
            if (!_plants[i].standing)
                continue;
            const core::i32 dx = static_cast<core::i32>(_plants[i].cellX) - static_cast<core::i32>(x);
            const core::i32 dz = static_cast<core::i32>(_plants[i].cellZ) - static_cast<core::i32>(z);
            if (dx > 1 || dx < -1 || dz > 1 || dz < -1)
                continue;
            _plants[i].standing = false;
            // Regrowth is slow relative to a visit, so a valley grazed bare stays
            // bare long enough to matter and the herd has to move on. That is the
            // whole feedback loop, and it is one line.
            _plants[i].regrowth = kRegrowthTicks;
            ++_grazed;
            _floraDirty = true;
            return;
        }
    }

    /// Advances regrowth and republishes the standing count as the producer level.
    void tickVegetation()
    {
        for (core::u32 i = 0u; i < _plants.size(); ++i)
        {
            if (_plants[i].standing || _plants[i].regrowth == 0u)
                continue;
            if (--_plants[i].regrowth == 0u)
            {
                _plants[i].standing = true;
                ++_regrown;
                _floraDirty = true;
            }
        }

        // The producer population is not integrated by the web: it is COUNTED. A
        // Lotka-Volterra producer term running alongside the real vegetation would
        // be a second, disagreeing answer to the same question.
        if (!_plants.empty())
            _web.species[0].population = math::Fixed32::fromInt(static_cast<core::i32>(standingPlants()));
    }

    /// Whether the cell one body-length along @p heading is walkable.
    [[nodiscard]] bool walkableAhead(const Creature &creature, math::Fixed32 headingX, math::Fixed32 headingZ,
                                     float halfW, float halfD) const
    {
        // A full cell ahead, not the fraction the next step covers: looking only as
        // far as one tick moves means the turn happens with the obstacle already
        // underfoot, which is a collision reported as an intention.
        const math::Fixed32 reach = math::Fixed32::one() + creature.genome.size;
        return walkable(creature.body.x + headingX * reach, creature.body.z + headingZ * reach, halfW, halfD);
    }

    /// Whether a world-space point is ground an animal may stand on.
    [[nodiscard]] bool walkable(math::Fixed32 x, math::Fixed32 z, float halfW, float halfD) const
    {
        const core::i32 gx = static_cast<core::i32>(x.toFloat() + halfW);
        const core::i32 gz = static_cast<core::i32>(z.toFloat() + halfD);
        if (!_blocked.contains(gx, gz))
            return false;
        return _blocked.at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)) == 0u;
    }

    /// Brings the number of bodies back in line with what the web says exists.
    void reconcilePopulation()
    {
        procgen::Random stock{_heredity ^ (_ticks * 0x9E3779B9u)};
        for (core::u32 species = 0u; species < 2u; ++species)
        {
            const core::u32 wanted =
                realisedCount(_web.species[species + 1u].population, _web.species[species + 1u].params.refuge);
            core::u32 have = 0u;
            for (core::u32 i = 0u; i < _creatures.size(); ++i)
                if (_creatures[i].species == species && _creatures[i].alive)
                    ++have;

            if (have < wanted)
            {
                spawnCreature(stock, species, _nextId++);
                ++_births;
            }
            else if (have > wanted)
            {
                // Retire the oldest first, so a collapse thins the herd rather
                // than deleting whatever the loop reached last.
                for (core::u32 i = 0u; i < _creatures.size(); ++i)
                    if (_creatures[i].species == species && _creatures[i].alive)
                    {
                        _creatures[i].alive = false;
                        ++_deaths;
                        break;
                    }
            }
        }
    }

    /// One simulation step: the agents move and deposit, then the field decays.
    void step(const TerrainData &world)
    {
        if (_width == 0u)
            return;
        ++_ticks;

        for (core::u32 i = 0u; i < _agentX.size(); ++i)
        {
            bool explored = false;
            const core::u32 direction = ai::chooseAntMove(_field, _ants, _agentX[i], _agentZ[i], _stream, explored);
            if (direction != ai::StigmergyField::kNoDirection)
            {
                const core::i32 nx = static_cast<core::i32>(_agentX[i]) + procgen::kNeighbor8X[direction];
                const core::i32 nz = static_cast<core::i32>(_agentZ[i]) + procgen::kNeighbor8Z[direction];
                if (nx >= 0 && nz >= 0 && static_cast<core::u32>(nx) < _width && static_cast<core::u32>(nz) < _depth)
                {
                    _agentX[i] = static_cast<core::u32>(nx);
                    _agentZ[i] = static_cast<core::u32>(nz);
                }
            }
            // An agent that wandered far enough goes home. Without it the colony
            // diffuses outward forever and the trail never closes into a route,
            // which is the difference between a pheromone field and a stain.
            const core::i32 dx = static_cast<core::i32>(_agentX[i]) - static_cast<core::i32>(_nestX);
            const core::i32 dz = static_cast<core::i32>(_agentZ[i]) - static_cast<core::i32>(_nestZ);
            if (dx * dx + dz * dz > static_cast<core::i32>(kForageRange * kForageRange))
            {
                _agentX[i] = _nestX;
                _agentZ[i] = _nestZ;
                ++_returns;
            }
            const core::u32 cell = _agentZ[i] * _width + _agentX[i];
            _field.depositTrail(0u, &cell, 1u, _ants.depositQuality);
        }
        stepCreatures(world);
        tickVegetation();
        measureSpeed();
        _field.step(_stigmergy);

        // Demography on its own clock, once a second rather than sixty times.
        // A Lotka-Volterra step is a GENERATION, not a frame: stepped at the tick
        // rate the web ran a lifetime between two log lines and the curve read as
        // noise. The parity recipe folds forty-eight steps for the same reason —
        // that is a run, not a second.
        if (_ticks % kWebPeriod == 0u)
        {
            _web.step(1u);
            reconcilePopulation();
        }
    }

    /**
     * @brief Draws the trail as a translucent film over the ground, plus the agents.
     *
     * Immediate mode and rebuilt every frame, deliberately: this is the one thing
     * on screen that is actually changing, so a display list would show a field
     * frozen at the moment the world was generated — which looks exactly like a
     * field that is working and is not.
     */
    void draw(const TerrainData &world) const
    {
        if (_width == 0u || world.height.empty())
            return;
        const float halfW = static_cast<float>(_width) * 0.5f;
        const float halfD = static_cast<float>(_depth) * 0.5f;
        const float ceiling = 24.0f;

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);
        glBegin(GL_QUADS);
        for (core::u32 z = 0u; z < _depth; ++z)
            for (core::u32 x = 0u; x < _width; ++x)
            {
                const float strength = _field.value(0u, x, z).toFloat();
                if (strength <= 0.25f)
                    continue;
                float t = strength / ceiling;
                if (t > 1.0f)
                    t = 1.0f;
                const float y = world.height.at(x, z).toFloat() + 0.12f;
                const float x0 = static_cast<float>(x) - halfW;
                const float z0 = static_cast<float>(z) - halfD;
                glColor4f(0.98f, 0.55f + 0.35f * t, 0.12f, 0.10f + 0.55f * t);
                glVertex3f(x0, y, z0);
                glVertex3f(x0 + 1.0f, y, z0);
                glVertex3f(x0 + 1.0f, y, z0 + 1.0f);
                glVertex3f(x0, y, z0 + 1.0f);
            }
        glEnd();
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        drawCreatures(world);

        glBegin(GL_QUADS);
        glColor3f(1.0f, 0.95f, 0.80f);
        for (core::u32 i = 0u; i < _agentX.size(); ++i)
        {
            const float y = world.height.at(_agentX[i], _agentZ[i]).toFloat() + 0.45f;
            const float x0 = static_cast<float>(_agentX[i]) - halfW + 0.30f;
            const float z0 = static_cast<float>(_agentZ[i]) - halfD + 0.30f;
            glVertex3f(x0, y, z0);
            glVertex3f(x0 + 0.4f, y, z0);
            glVertex3f(x0 + 0.4f, y, z0 + 0.4f);
            glVertex3f(x0, y, z0 + 0.4f);
        }
        glEnd();
    }

    /**
     * @brief Draws the herd: one body per animal, sized and tinted by its genome.
     *
     * Size comes from the genome and the tint from the derived temperament, which
     * is the whole reason those two things are separate in the module: a herd of
     * identical silhouettes says the genetics ran; a herd where the big ones are
     * visibly big says what the genetics DID. An anomaly — a body standing well
     * above its species mean — is drawn pale, so the emergent boss the heredity
     * module exists to produce is something you spot on the map.
     */
    void drawCreatures(const TerrainData &world) const
    {
        if (_creatures.empty() || world.height.empty())
            return;

        const float halfW = static_cast<float>(_width) * 0.5f;
        const float halfD = static_cast<float>(_depth) * 0.5f;

        // The species mean, recomputed here rather than cached: it moves as the
        // herd breeds, and a stale mean would mark the wrong animals.
        _statScratch.clear();
        for (core::u32 i = 0u; i < _creatures.size(); ++i)
            if (_creatures[i].alive && _creatures[i].species == 0u)
                _statScratch.push_back(_creatures[i].genome);
        const ecology::PopulationStats stats = ecology::strengthStats(_statScratch.empty() ? nullptr : &_statScratch[0],
                                                                      static_cast<core::u32>(_statScratch.size()));
        ecology::HeredityParams heredity;

        glBegin(GL_QUADS);
        for (core::u32 i = 0u; i < _creatures.size(); ++i)
        {
            const Creature &creature = _creatures[i];
            if (!creature.alive)
                continue;

            const core::i32 gx = static_cast<core::i32>(creature.body.x.toFloat() + halfW);
            const core::i32 gz = static_cast<core::i32>(creature.body.z.toFloat() + halfD);
            if (!world.height.contains(gx, gz))
                continue;
            const float ground = world.height.at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)).toFloat();

            const ai::PersonalityTraits traits = ai::personalityOf(creature.id, creature.species);
            const float half = 0.22f * creature.genome.size.toFloat();

            Rgb colour = creature.species == 1u ? Rgb{0.72f, 0.20f, 0.18f} : Rgb{0.80f, 0.66f, 0.36f};
            // Temperament is visible, faintly: an aggressive animal runs hot.
            colour.r += 0.18f * traits.aggression.toFloat();
            colour.b += 0.12f * traits.nervousness.toFloat();
            if (creature.species == 0u && stats.count > 4u && ecology::isAnomaly(creature.genome, stats, heredity))
                colour = {0.98f, 0.94f, 0.86f};

            const float cx = creature.body.x.toFloat();
            const float cz = creature.body.z.toFloat();
            const float cy = ground + half + 0.1f;
            const float x0 = cx - half;
            const float x1 = cx + half;
            const float y0 = cy - half;
            const float y1 = cy + half;
            const float z0 = cz - half;
            const float z1 = cz + half;

            glColor3f(colour.r, colour.g, colour.b);
            glVertex3f(x0, y1, z0);
            glVertex3f(x1, y1, z0);
            glVertex3f(x1, y1, z1);
            glVertex3f(x0, y1, z1);
            glColor3f(colour.r * 0.7f, colour.g * 0.7f, colour.b * 0.7f);
            glVertex3f(x0, y0, z1);
            glVertex3f(x0, y1, z1);
            glVertex3f(x1, y1, z1);
            glVertex3f(x1, y0, z1);
            glVertex3f(x1, y0, z0);
            glVertex3f(x1, y1, z0);
            glVertex3f(x0, y1, z0);
            glVertex3f(x0, y0, z0);
            glColor3f(colour.r * 0.55f, colour.g * 0.55f, colour.b * 0.55f);
            glVertex3f(x1, y0, z1);
            glVertex3f(x1, y1, z1);
            glVertex3f(x1, y1, z0);
            glVertex3f(x1, y0, z0);
            glVertex3f(x0, y0, z0);
            glVertex3f(x0, y1, z0);
            glVertex3f(x0, y1, z1);
            glVertex3f(x0, y0, z1);
        }
        glEnd();
    }

    [[nodiscard]] core::u32 aliveCount(core::u32 species) const noexcept
    {
        core::u32 alive = 0u;
        for (core::u32 i = 0u; i < _creatures.size(); ++i)
            if (_creatures[i].alive && _creatures[i].species == species)
                ++alive;
        return alive;
    }

    /**
     * @brief Mean ground speed of the herd, in cells per second.
     *
     * The number that says whether the movement is an animal's or a projectile's.
     * It is measured rather than asserted because the first version of this layer
     * moved bodies by their raw boid velocity per tick — a plausible-looking line
     * of code that means sixty cells a second, and the only way to see it was to
     * ask how fast they were actually going.
     */
    [[nodiscard]] double meanSpeed() const { return _measuredSpeed; }

    [[nodiscard]] core::u32 births() const noexcept { return _births; }
    [[nodiscard]] core::u32 deaths() const noexcept { return _deaths; }
    [[nodiscard]] core::u32 refusals() const noexcept { return _refusals; }
    [[nodiscard]] core::u32 strays() const noexcept { return _strays; }
    [[nodiscard]] core::u32 standingPlants() const noexcept
    {
        core::u32 standing = 0u;
        for (core::u32 i = 0u; i < _plants.size(); ++i)
            if (_plants[i].standing)
                ++standing;
        return standing;
    }

    [[nodiscard]] core::u32 plantCount() const noexcept { return static_cast<core::u32>(_plants.size()); }
    [[nodiscard]] core::u32 grazed() const noexcept { return _grazed; }
    [[nodiscard]] core::u32 regrown() const noexcept { return _regrown; }
    [[nodiscard]] bool plantStanding(core::usize index) const
    {
        return index < _plants.size() ? _plants[index].standing : true;
    }
    /// True when the standing set changed since the last @ref clearFloraDirty.
    [[nodiscard]] bool floraDirty() const noexcept { return _floraDirty; }
    void clearFloraDirty() noexcept { _floraDirty = false; }

    [[nodiscard]] core::u32 strayOutside() const noexcept { return _strayOutside; }
    [[nodiscard]] core::u32 strayBlocked() const noexcept { return _strayBlocked; }

    [[nodiscard]] core::u32 trailCells() const
    {
        core::u32 cells = 0u;
        for (core::u32 z = 0u; z < _depth; ++z)
            for (core::u32 x = 0u; x < _width; ++x)
                if (_field.value(0u, x, z).toFloat() > 0.25f)
                    ++cells;
        return cells;
    }

    [[nodiscard]] core::u32 agents() const noexcept { return static_cast<core::u32>(_agentX.size()); }
    [[nodiscard]] core::u32 returns() const noexcept { return _returns; }
    [[nodiscard]] core::u32 ticks() const noexcept { return _ticks; }
    [[nodiscard]] const ecology::TrophicWeb &web() const noexcept { return _web; }

private:
    static constexpr core::u32 kAgents = 48u;
    static constexpr core::u32 kForageRange = 26u;
    /// Bodies the map will hold, whatever the model says.
    static constexpr core::u32 kMaxBodies = 120u;
    /// Fixed ticks between two demographic steps.
    static constexpr core::u32 kWebPeriod = 60u;
    /// Ticks a cropped plant takes to come back: twenty seconds at 60 Hz.
    static constexpr core::u32 kRegrowthTicks = 1200u;
    /// Duration of one fixed tick, as the engine's 60 Hz loop defines it.
    static inline const math::Fixed32 kFixedStep = math::Fixed32::fromFloat(1.0f / 60.0f);

    ai::StigmergyField _field;
    ai::StigmergyParams _stigmergy{};
    ai::AntParams _ants{};
    ecology::TrophicWeb _web;
    lpl::pmr::vector<Creature> _creatures;
    lpl::pmr::vector<Plant> _plants;
    /// Scratch reused every step: the flocking pass needs one species at a time.
    mutable lpl::pmr::vector<ai::Boid> _flockScratch;
    mutable lpl::pmr::vector<ecology::Genome> _statScratch;
    procgen::Grid<core::u8> _blocked;
    lpl::pmr::vector<core::u32> _agentX;
    lpl::pmr::vector<core::u32> _agentZ;
    core::u32 _width{0u};
    core::u32 _depth{0u};
    core::u32 _nestX{0u};
    core::u32 _nestZ{0u};
    core::u32 _stream{1u};
    core::u32 _returns{0u};
    core::u32 _ticks{0u};
    core::u32 _heredity{1u};
    core::u32 _nextId{1u};
    core::u32 _births{0u};
    core::u32 _deaths{0u};
    core::u32 _refusals{0u};
    core::u32 _strays{0u};
    core::u32 _strayOutside{0u};
    core::u32 _strayBlocked{0u};
    core::u32 _grazed{0u};
    core::u32 _regrown{0u};
    bool _floraDirty{false};
    lpl::pmr::vector<ai::Boid> _speedMarks;
    double _measuredSpeed{0.0};
};

void drawHud(const TerrainData &world, const Options &options, int width, int height, double frameMilliseconds,
             core::u32 bodies, core::u32 settled, core::u32 groundHits, core::u32 systemRuns, core::u32 triangles,
             const LivingLayer &living)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, width, height, 0.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    const float scale = 1.0f;
    const float lineHeight = static_cast<float>(image::kFontHeight) * scale + 2.0f;
    float y = 8.0f;
    char line[256];

    // Panel behind the text, sized from the lines actually about to be drawn:
    // seven status lines, one per present biome, and three of key hints. Guessing a
    // height instead leaves the hints spilling onto the map, unreadable.
    core::u32 presentBiomes = 0u;
    for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
        if (world.biomeCounts[i] != 0u)
            ++presentBiomes;
    const float panelHeight = lineHeight * static_cast<float>(11u + presentBiomes + 3u) + 24.0f;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.02f, 0.02f, 0.03f, 0.78f);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(452.0f, 0.0f);
    glVertex2f(452.0f, panelHeight);
    glVertex2f(0.0f, panelHeight);
    glEnd();
    glDisable(GL_BLEND);

    const auto put = [&](const char *text, float r, float g, float b) {
        glColor3f(r, g, b);
        drawText(text, 10.0f, y, scale);
        y += lineHeight;
    };

    std::snprintf(line, sizeof(line), "seed %u  %ux%u  build %.0f ms  frame %.1f ms  %u k tris", options.seed,
                  options.size, options.size, world.buildMilliseconds, frameMilliseconds, triangles / 1000u);
    put(line, 1.0f, 0.68f, 0.16f);

    if (options.shading == Shading::Climate)
        std::snprintf(line, sizeof(line), "view %s   shading %s:%s   vegetation %s", viewName(options.view),
                      shadingName(options.shading), climateAxisName(options.climateAxis),
                      options.vegetation ? "on" : "off");
    else
        std::snprintf(line, sizeof(line), "view %s   shading %s   vegetation %s", viewName(options.view),
                      shadingName(options.shading), options.vegetation ? "on" : "off");
    put(line, 0.90f, 0.90f, 0.88f);

    std::snprintf(line, sizeof(line), "noise %s  warp %s  terraces %s", noiseName(options.noise),
                  options.warp ? "on" : "off", options.terraces ? "on" : "off");
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "erosion %s  rivers %s (%u cells)  wind %s", options.erosion ? "on" : "off",
                  options.rivers ? "on" : "off", world.riverCells, windName(options.windDirection));
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "lakes %u cells   trunk drains %u of %u", world.raisedCells,
                  world.maxAccumulation, world.height.cellCount());
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "roads %s (%u cells)", options.roads ? "on" : "off", world.roadCells);
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "regions %u (%s)   settlement %s", world.regions.regionCount,
                  metricName(options.metric), options.settlement ? "on" : "off");
    put(line, 0.72f, 0.74f, 0.76f);

    if (options.caveKind % 4u == 3u)
        // Entrances and reachability, not a floor count: a sealed system and an
        // open one have the same floor area, and only one of them is a cave.
        std::snprintf(line, sizeof(line), "underground layered  %u layers  %u entrances  %u/%u reachable",
                      world.caveLayers, world.caveEntrances, world.caveReachable, world.caveHollow);
    else
        std::snprintf(line, sizeof(line), "underground %s  %u floor  %s", caveName(options.caveKind),
                      world.dungeonFloor, world.dungeonConnected ? "connected" : "SPLIT");
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "town %s  %zu plots  roadside %u modules  liminal %u open",
                  options.grammarBuildings ? "grammar" : "prisms", world.plots.size(), world.roadsideModules,
                  world.liminal.openCells);
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "bodies %u  resting on terrain %u  ticks %u  flora %u", bodies, groundHits,
                  systemRuns, settled);
    put(line, 0.62f, 0.78f, 0.62f);

    if (options.living)
    {
        const ecology::TrophicWeb &web = living.web();
        std::snprintf(line, sizeof(line), "colony %u agents  %u trail cells  %u returns  t=%u", living.agents(),
                      living.trailCells(), living.returns(), living.ticks());
        put(line, 0.95f, 0.72f, 0.30f);

        // Head counts rather than a signature: what matters when looking is whether
        // the web settles or oscillates apart, and that is a curve, not a hash.
        std::snprintf(line, sizeof(line), "herd  %u grazers  %u hunters  +%u/-%u  %u refused steps",
                      living.aliveCount(0u), living.aliveCount(1u), living.births(), living.deaths(),
                      living.refusals());
        put(line, 0.85f, 0.72f, 0.45f);

        std::snprintf(line, sizeof(line), "web  plants %u/%u standing (-%u +%u)  herbivores %d  predators %d",
                      living.standingPlants(), living.plantCount(), living.grazed(), living.regrown(),
                      web.species.size() > 1u ? web.species[1].population.toInt() : 0,
                      web.species.size() > 2u ? web.species[2].population.toInt() : 0);
        put(line, 0.95f, 0.72f, 0.30f);
    }
    else
    {
        put("living layer off", 0.55f, 0.55f, 0.55f);
        put("", 0.5f, 0.5f, 0.5f);
        put("", 0.5f, 0.5f, 0.5f);
    }

    y += 4.0f;
    // Biome tally with a swatch each, which doubles as the legend.
    core::u32 shown = 0u;
    for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
    {
        if (world.biomeCounts[i] == 0u)
            continue;
        const Rgb colour = biomeColour(static_cast<procgen::BiomeId>(i));
        glColor3f(colour.r, colour.g, colour.b);
        glBegin(GL_QUADS);
        glVertex2f(10.0f, y + 2.0f);
        glVertex2f(24.0f, y + 2.0f);
        glVertex2f(24.0f, y + 14.0f);
        glVertex2f(10.0f, y + 14.0f);
        glEnd();

        const core::u32 perMille = (world.biomeCounts[i] * 1000u) / world.height.cellCount();
        std::snprintf(line, sizeof(line), "%-11s %4u/1000", procgen::biomeName(static_cast<procgen::BiomeId>(i)),
                      perMille);
        glColor3f(0.88f, 0.88f, 0.86f);
        drawText(line, 30.0f, y, scale);
        y += lineHeight;
        ++shown;
        if (shown >= 12u)
            break;
    }

    y += 6.0f;
    glColor3f(0.55f, 0.45f, 0.30f);
    drawText("N seed  S shading  1/2/3 noise  E erode  R rivers  W warp", 10.0f, y, scale);
    y += lineHeight;
    drawText("T terrace  G town  M metric  K wind  C cave  V view  B flora", 10.0f, y, scale);
    y += lineHeight;
    drawText("X climate axis  U grammar  L living  J chunks  H roads", 10.0f, y, scale);
    y += lineHeight;
    drawText("F wire  O water  [ ] size  drag orbit  wheel zoom  Q quit", 10.0f, y, scale);

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ─────────────────────────────────────────────────────────────────────────────
// Presentation: the X11 window and its GL context
//
// This belongs to the host, not to the game. The World draws through it and knows
// nothing about X11 — which is the same split the engine's platform seam makes,
// just resolved by hand here because this tool predates the renderer.
// ─────────────────────────────────────────────────────────────────────────────

class Viewport {
public:
    [[nodiscard]] bool open()
    {
        _display = XOpenDisplay(nullptr);
        if (_display == nullptr)
        {
            std::fprintf(stderr, "mapview: cannot open a display (is DISPLAY set?)\n");
            return false;
        }

        int attributes[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};
        _visual = glXChooseVisual(_display, DefaultScreen(_display), attributes);
        if (_visual == nullptr)
        {
            std::fprintf(stderr, "mapview: no double-buffered RGBA visual with a depth buffer\n");
            return false;
        }

        Window root = DefaultRootWindow(_display);
        XSetWindowAttributes windowAttributes{};
        windowAttributes.colormap = XCreateColormap(_display, root, _visual->visual, AllocNone);
        windowAttributes.event_mask =
            ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask;

        _window =
            XCreateWindow(_display, root, 0, 0, static_cast<unsigned>(_width), static_cast<unsigned>(_height), 0,
                          _visual->depth, InputOutput, _visual->visual, CWColormap | CWEventMask, &windowAttributes);
        XMapWindow(_display, _window);
        XStoreName(_display, _window, "lpl-mapview — solo client");

        _context = glXCreateContext(_display, _visual, nullptr, GL_TRUE);
        glXMakeCurrent(_display, _window, _context);

        glEnable(GL_DEPTH_TEST);
        // No back-face culling. A heightfield is a single sheet, not a closed solid:
        // orbiting below the horizon shows its underside, and culling it there leaves
        // a hole in the world. Getting the winding wrong culls the whole surface,
        // which looks exactly like a lighting failure and is not one.
        glDisable(GL_CULL_FACE);
        glClearColor(0.05f, 0.06f, 0.08f, 1.0f);
        return true;
    }

    void close()
    {
        if (_display == nullptr)
            return;
        glXMakeCurrent(_display, None, nullptr);
        if (_context != nullptr)
            glXDestroyContext(_display, _context);
        if (_window != 0)
            XDestroyWindow(_display, _window);
        XCloseDisplay(_display);
        _display = nullptr;
    }

    [[nodiscard]] Display *display() const noexcept { return _display; }
    [[nodiscard]] int width() const noexcept { return _width; }
    [[nodiscard]] int height() const noexcept { return _height; }
    void resize(int width, int height) noexcept
    {
        _width = width;
        _height = height;
    }
    void swap() { glXSwapBuffers(_display, _window); }

private:
    Display *_display{nullptr};
    XVisualInfo *_visual{nullptr};
    Window _window{0};
    GLXContext _context{nullptr};
    int _width{1280};
    int _height{800};
};

// ─────────────────────────────────────────────────────────────────────────────
// The one system that is the game's business
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @class TerrainCollisionSystem
 * @brief Keeps loose bodies on top of the ground, and lets them slide downhill.
 *
 * The engine's built-in physics knows about gravity, damping and bodies hitting each
 * other. It does not know about *this* world's ground, because a heightfield is
 * content, not a host service — so the collision against it belongs to the World.
 * That is the split the World seam exists to make: generic systems come from the
 * Config, the ones that only make sense for this game come from `onInit`.
 *
 * Registered in @c PostPhysics, deliberately. Doing this work in @c Physics would
 * put a second system in the phase that already owns integration, and the engine's
 * own physics would then be stepping the same buffers alongside it. Correcting
 * positions *after* the step is both the right order physically and the only order
 * that keeps one writer per phase.
 */
class TerrainCollisionSystem final : public ecs::ISystem {
public:
    TerrainCollisionSystem(ecs::Registry &registry, const procgen::Heightfield &terrain, core::f32 cellSize) noexcept
        : _registry(registry), _terrain(terrain), _cellSize(cellSize)
    {
    }

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override { return _descriptor; }

    void execute(core::f32 /*dt*/) override
    {
        ++_executions;
        // Reset per tick: what matters is how many bodies are in contact NOW, not how
        // many contacts have ever happened.
        _resting = 0u;
        if (_terrain.empty())
        {
            if (_executions == 1u)
                core::Log::warn("TerrainCollision: the terrain is empty, nothing to stand on");
            return;
        }

        const core::f32 halfWidth = static_cast<core::f32>(_terrain.width()) * _cellSize * 0.5f;
        const core::f32 halfDepth = static_cast<core::f32>(_terrain.depth()) * _cellSize * 0.5f;

        for (const auto &partition : _registry.partitions())
        {
            if (!partition)
                continue;
            for (const auto &chunk : partition->chunks())
            {
                if (!chunk)
                    continue;
                auto *positions = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
                auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
                auto *aabb = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::AABB));
                auto *mass = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
                // A chunk without Velocity is scenery: it was never going to move,
                // so there is nothing to correct.
                if (positions == nullptr || velocities == nullptr)
                    continue;

                const core::u32 count = chunk->count();
                for (core::u32 i = 0u; i < count; ++i)
                {
                    // Zero mass means immovable. Correcting one to the ground would be
                    // harmless, but the downhill slide below would hand it a velocity and
                    // walk a tree off its own footing.
                    if (mass != nullptr && mass[i].raw() == 0)
                        continue;
                    const core::f32 half = aabb != nullptr ? aabb[i].y.toFloat() * 0.5f : 0.25f;

                    const core::f32 worldX = positions[i].x.toFloat() + halfWidth;
                    const core::f32 worldZ = positions[i].z.toFloat() + halfDepth;
                    const core::i32 cellX = static_cast<core::i32>(worldX / _cellSize);
                    const core::i32 cellZ = static_cast<core::i32>(worldZ / _cellSize);

                    // Keep bodies over the map. A body nudged past the edge by a
                    // collision would otherwise sail off and hang in empty space,
                    // which looks exactly like the collision having failed.
                    const core::f32 limitX = halfWidth - _cellSize;
                    const core::f32 limitZ = halfDepth - _cellSize;
                    if (positions[i].x.toFloat() < -limitX)
                    {
                        positions[i].x = math::Fixed32::fromFloat(-limitX);
                        velocities[i].x = math::Fixed32::zero();
                    }
                    else if (positions[i].x.toFloat() > limitX)
                    {
                        positions[i].x = math::Fixed32::fromFloat(limitX);
                        velocities[i].x = math::Fixed32::zero();
                    }
                    if (positions[i].z.toFloat() < -limitZ)
                    {
                        positions[i].z = math::Fixed32::fromFloat(-limitZ);
                        velocities[i].z = math::Fixed32::zero();
                    }
                    else if (positions[i].z.toFloat() > limitZ)
                    {
                        positions[i].z = math::Fixed32::fromFloat(limitZ);
                        velocities[i].z = math::Fixed32::zero();
                    }

                    const core::f32 ground = _terrain.clamped(cellX, cellZ).toFloat() + half;
                    if (positions[i].y.toFloat() >= ground)
                        continue;

                    // Land: sit on the surface and stop falling.
                    ++_resting;
                    positions[i].y = math::Fixed32::fromFloat(ground);
                    if (velocities[i].y.toFloat() < 0.0f)
                        velocities[i].y = math::Fixed32::zero();

                    // Slide along the downhill gradient, and lose speed doing it.
                    // Without the slide a boulder simply stops where it landed and
                    // the terrain might as well be a floor; with it, the map's
                    // drainage pattern becomes visible in where things collect.
                    const core::f32 left = _terrain.clamped(cellX - 1, cellZ).toFloat();
                    const core::f32 right = _terrain.clamped(cellX + 1, cellZ).toFloat();
                    const core::f32 back = _terrain.clamped(cellX, cellZ - 1).toFloat();
                    const core::f32 front = _terrain.clamped(cellX, cellZ + 1).toFloat();

                    const core::f32 slide = 0.55f;
                    const core::f32 friction = 0.86f;
                    velocities[i].x =
                        math::Fixed32::fromFloat(velocities[i].x.toFloat() * friction + (left - right) * slide);
                    velocities[i].z =
                        math::Fixed32::fromFloat(velocities[i].z.toFloat() * friction + (back - front) * slide);
                }
            }
        }
    }

private:
    using FVec3 = math::Vec3<math::Fixed32>;

    static constexpr ecs::ComponentAccess kAccesses[] = {
        {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
        {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
        {ecs::ComponentId::AABB,     ecs::AccessMode::ReadOnly },
    };

public:
    [[nodiscard]] core::u32 executions() const noexcept { return _executions; }
    /**
     * @brief Bodies touching the ground on the most recent tick.
     *
     * The honest measure of "at rest". Testing the vertical velocity instead reports
     * nothing ever settling: a body held up by position correction still has one
     * tick of gravity applied to it every tick, so its velocity oscillates around
     * zero forever even though it has not moved. What is stable is the contact.
     */
    [[nodiscard]] core::u32 resting() const noexcept { return _resting; }

private:
    core::u32 _executions{0u};
    core::u32 _resting{0u};
    ecs::SystemDescriptor _descriptor{"TerrainCollision", ecs::SchedulePhase::PostPhysics, kAccesses};
    ecs::Registry &_registry;
    const procgen::Heightfield &_terrain;
    core::f32 _cellSize;
};

// ─────────────────────────────────────────────────────────────────────────────
// The game
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @class MapviewWorld
 * @brief A solo world: procgen builds the terrain, the engine runs it, GL shows it.
 */
class MapviewWorld final : public engine::World {
public:
    explicit MapviewWorld(Viewport &viewport, const Options &options) noexcept : _viewport(viewport), _options(options)
    {
    }

    [[nodiscard]] core::Expected<void> onInit(engine::WorldContext &context) override
    {
        regenerate();
        // The game's own system, added on top of whatever the Config already put on
        // this scheduler. The engine's physics is registered before onInit runs.
        auto collision = lpl::pmr::make_unique<TerrainCollisionSystem>(registry(), _terrain.height, 1.0f);
        _collision = collision.get();
        auto registered =
            scheduler().registerSystem(static_cast<lpl::pmr::unique_ptr<ecs::ISystem>>(std::move(collision)));
        if (!registered)
        {
            _collision = nullptr;
            core::Log::error("mapview: the scheduler REFUSED the terrain collision system");
        }
        else
        {
            core::Log::info("mapview: terrain collision system registered in PostPhysics");
        }

        core::Log::info("mapview: world hosted by the engine, physics on the scheduler");
        (void) context;
        return {};
    }

    void onFixedStep(core::f32 dt) override
    {
        ++_steps;
        pumpEvents();
        const double beforeTick = nowMilliseconds();
        // The base implementation is exactly this, and calling it is the point: the
        // scheduler advances the engine's physics and this world's own system in the
        // order the DAG decided.
        engine::World::onFixedStep(dt);
        // On the fixed clock, deliberately: a colony stepped once per FRAME would
        // converge faster on a fast machine, which is the same class of bug the
        // engine's fixed timestep exists to prevent for the physics.
        if (_options.living)
            _living.step(_terrain);
        _tickMilliseconds = nowMilliseconds() - beforeTick;
    }

    void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override
    {
        const double started = nowMilliseconds();

        glViewport(0, 0, _viewport.width(), _viewport.height());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        applyPerspective(_viewport.width(), _viewport.height());

        glLoadIdentity();
        glTranslatef(0.0f, _camera.height, -_camera.distance);
        glRotatef(_camera.pitch * 57.29578f, 1.0f, 0.0f, 0.0f);
        glRotatef(_camera.yaw * 57.29578f, 0.0f, 1.0f, 0.0f);

        switch (_options.view)
        {
        case View::Surface:
            if (_options.wireframe)
            {
                drawMesh(_surfaceMesh, true);
                drawMesh(_townMesh, true);
            }
            else
            {
                _surfaceList.call();
                _townList.call();
                if (_options.vegetation)
                    _floraList.call();
                _roadsideList.call();
            }
            drawEntities();
            if (_options.living)
                _living.draw(_terrain);
            if (_options.chunkOverlay)
                drawStreamingPlan();
            if (_options.water && !_terrain.height.empty())
            {
                // The sea level travelled with the terrain when it was lifted, so the
                // shoreline in the mesh and the shoreline in the water still agree.
                drawWaterPlane(_terrain, _terrain.seaLevel);
            }
            break;

        case View::Cutaway:
            // Underground first and opaque, then the ground over it with blending on
            // and depth writes off. Drawing the translucent surface first would let it
            // occlude the very thing it is meant to reveal.
            if (_options.wireframe)
                drawMesh(_undergroundMesh, true);
            else
                _undergroundList.call();
            drawEntities();
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_FALSE);
            _surfaceGhostList.call();
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
            if (!_options.wireframe)
                _townList.call();
            break;

        case View::Underground:
            if (_options.wireframe)
                drawMesh(_undergroundMesh, true);
            else
                _undergroundList.call();
            break;

        case View::Liminal:
        case View::Count:
            if (_options.wireframe)
                drawMesh(_liminalMesh, true);
            else
                _liminalList.call();
            break;
        }

        // The vegetation on screen follows the vegetation in the model, but not at
        // the model's rate: recompiling a display list of several hundred plants on
        // every bite would cost more than the whole frame. Once a second is faster
        // than a herd can visibly strip a hillside.
        if (_options.living && _living.floraDirty() && _frames % 60u == 0u)
        {
            _floraMesh = buildFloraMesh();
            _floraList.compile(_floraMesh, &drawMesh, false);
            _living.clearFloraDirty();
        }

        drawHud(_terrain, _options, _viewport.width(), _viewport.height(), _frameMilliseconds, _boulderCount,
                static_cast<core::u32>(_propIds.size()), _collision != nullptr ? _collision->resting() : 0u,
                _collision != nullptr ? _collision->executions() : 0u,
                _surfaceList.triangles() + _undergroundList.triangles() + _townList.triangles() +
                    _floraList.triangles() + _roadsideList.triangles(),
                _living);
        _viewport.swap();
        _frameMilliseconds = nowMilliseconds() - started;

        // Periodic sanity line, so the physics can be checked from a log rather than
        // from a screenshot.
        if (++_frames % 240u == 0u)
        {
            const double elapsed = (nowMilliseconds() - _startedAt) / 1000.0;
            // Trees must not have moved. Their positions are hashed once and compared:
            // an obstacle that drifts is a solver or a system writing where it should not.
            core::u32 floraHash = 0x811C9DC5u;
            for (const auto &partition : registry().partitions())
            {
                if (!partition || !partition->archetype().has(ecs::ComponentId::Mass))
                    continue;
                for (const auto &chunk : partition->chunks())
                {
                    if (!chunk)
                        continue;
                    const auto *p = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
                    const auto *m = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
                    if (p == nullptr || m == nullptr)
                        continue;
                    for (core::u32 i = 0u; i < chunk->count(); ++i)
                        if (m[i].raw() == 0)
                            floraHash = (floraHash ^ static_cast<core::u32>(p[i].y.raw())) * 0x01000193u;
                }
            }
            std::printf("  obstacle fold %08X%s\n", floraHash,
                        _obstacleFold == 0u || _obstacleFold == floraHash ? "" : "  <-- AN OBSTACLE MOVED");
            _obstacleFold = floraHash;
            // The herd on the log line too, not only in the HUD: a population that
            // collapsed to nothing and a layer that never spawned look identical on
            // screen, and only one of them is the model working.
            if (_options.living)
                std::printf("  herd %u grazers / %u hunters | %.2f cells/s | %u refused | trail %u cells | "
                            "%u strays | plants %u/%u | web %d/%d/%d\n",
                            _living.aliveCount(0u), _living.aliveCount(1u), _living.meanSpeed(), _living.refusals(),
                            _living.trailCells(), _living.strays(), _living.standingPlants(), _living.plantCount(),
                            _living.web().species.size() > 0u ? _living.web().species[0].population.toInt() : 0,
                            _living.web().species.size() > 1u ? _living.web().species[1].population.toInt() : 0,
                            _living.web().species.size() > 2u ? _living.web().species[2].population.toInt() : 0);
            std::printf("%.0f fps | frame %.1f ms | tick %.2f ms | view %s | resting %u/%u\n",
                        elapsed > 0.0 ? static_cast<double>(_frames) / elapsed : 0.0, _frameMilliseconds,
                        _tickMilliseconds, viewName(_options.view), _collision != nullptr ? _collision->resting() : 0u,
                        _boulderCount);
            std::fflush(stdout);
        }
        (void) context;
    }

    void onShutdown() override { core::Log::info("mapview: closed"); }

    [[nodiscard]] const char *name() const noexcept override { return "Mapview"; }

    void bindEngine(engine::Engine &engine) noexcept { _engine = &engine; }

private:
    using FVec3 = math::Vec3<math::Fixed32>;

    /// Rebuilds the terrain and everything derived from it, entities included.
    void regenerate()
    {
        // Retire the previous world's scenery. The registry has no bulk clear by
        // design: whoever created entities is the one that can retire them, and the ids
        // came back from materializeProps for exactly this.
        for (core::usize i = 0u; i < _propIds.size(); ++i)
            (void) registry().destroyEntity(_propIds[i]);
        _propIds.clear();

        _terrain = generateTerrain(_options, &registry(), &_propIds);
        _surfaceMesh = buildSurfaceMesh(_terrain, _options);

        // The layered system replaces the flat plan entirely rather than being drawn
        // beside it: they are two answers to the same question, and showing both at
        // once would say nothing about either.
        _undergroundMesh = _options.caveKind % 4u == 3u ? buildCaveSystemMesh(_terrain, kCaveDepth, kCaveLayerSpacing) :
                                                          buildDungeonMesh(_terrain, kCaveDepth);

        // The grammar's own volume when there is one, the box heuristic otherwise.
        // What is on screen is then what the generator produced, not the viewer's
        // guess at what it might have meant.
        if (_options.settlement && _options.grammarBuildings && !_terrain.townVolume.empty())
        {
            const Rgb palette[4] = {
                {0.0f,  0.0f,  0.0f },
                {0.66f, 0.56f, 0.44f}, // walls
                {0.44f, 0.40f, 0.36f}, // base course
                {0.42f, 0.24f, 0.20f}
            }; // roof
            _townMesh = buildVoxelMesh(_terrain.townVolume, _terrain, -0.4f, palette, 4u);
        }
        else
        {
            _townMesh = _options.settlement ? buildTownMesh(_terrain, _terrain.plots) : std::vector<Vertex>{};
        }

        if (_options.roads && _options.grammarBuildings && !_terrain.roadsideVolume.empty())
        {
            const Rgb palette[4] = {
                {0.0f,  0.0f,  0.0f },
                {0.35f, 0.28f, 0.20f}, // fence
                {0.30f, 0.26f, 0.22f},
                {0.95f, 0.80f, 0.35f}
            }; // lamp
            _roadsideMesh = buildVoxelMesh(_terrain.roadsideVolume, _terrain, 0.0f, palette, 4u);
        }
        else
        {
            _roadsideMesh.clear();
        }

        // The liminal sector is its own generator on its own seed, built once per
        // regeneration so switching to the view costs nothing.
        {
            procgen::LiminalParams liminal;
            liminal.width = _options.size;
            liminal.depth = _options.size;
            liminal.seed = _options.seed ^ 0x11B1A0u;
            _terrain.liminal = procgen::generateLiminal(liminal);
            _liminalMesh = buildLiminalMesh(_terrain.liminal);
        }

        // The herd needs to know where the trees ARE, which is a question only the
        // registry can answer — the scatter rules say how many, not where.
        _propPositions.clear();
        for (const auto &partition : registry().partitions())
        {
            if (!partition)
                continue;
            for (const auto &chunk : partition->chunks())
            {
                if (!chunk)
                    continue;
                const auto *position = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
                const auto *mass = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
                const bool hasMass = partition->archetype().has(ecs::ComponentId::Mass);
                if (position == nullptr)
                    continue;
                for (core::u32 i = 0u; i < chunk->count(); ++i)
                    if (!hasMass || mass == nullptr || mass[i].raw() == 0)
                        _propPositions.push_back(position[i]);
            }
        }
        _living.reset(_terrain, _options.seed, _propPositions);
        _floraMesh = buildFloraMesh();
        compileLists();

        // The bodies are created once and moved afterwards. Destroying and recreating
        // them on every regeneration would churn the registry's pools for no reason,
        // and there is no bulk clear precisely because nothing should need one.
        if (_boulderCount == 0u)
            createBoulders();
        placeBoulders();
        std::printf("seed %u %ux%u noise=%s warp=%d erode=%d rivers=%d(%u) wind=%s metric=%s cave=%s "
                    "build=%.1fms tris=%zu flora=%zu boulders=%u plots=%zu townTris=%zu caveTris=%zu\n",
                    _options.seed, _options.size, _options.size, noiseName(_options.noise), int(_options.warp),
                    int(_options.erosion), int(_options.rivers), _terrain.riverCells, windName(_options.windDirection),
                    metricName(_options.metric), caveName(_options.caveKind), _terrain.buildMilliseconds,
                    _surfaceMesh.size() / 3u, _propIds.size(), _boulderCount, _terrain.plots.size(),
                    _townMesh.size() / 3u, _undergroundMesh.size() / 3u);
        std::fflush(stdout);
    }

    /// Creates the dynamic bodies, once, so the simulation has something to do.
    void createBoulders()
    {
        const ecs::ComponentId ids[] = {ecs::ComponentId::Position, ecs::ComponentId::Velocity, ecs::ComponentId::Mass,
                                        ecs::ComponentId::AABB};
        const ecs::Archetype archetype{ids};
        _boulderIds.clear();
        for (core::u32 i = 0u; i < kBoulders; ++i)
            if (auto created = registry().createEntity(archetype))
                _boulderIds.push_back(*created);
    }

    /**
     * @brief Drops every body back above the current terrain.
     *
     * Addressed by IDENTITY, like the scatter that shares this archetype with it.
     * The row-order version of this loop wrote the first 240 rows of whichever
     * partition exposed a Velocity buffer — and a collidable prop's archetype is
     * exactly a body's, so after a regeneration those rows were part trees. The
     * bodies that lost their row kept a stale position in the sky and a prop's zero
     * mass, which is a boulder that has stopped being subject to gravity: the
     * "stones frozen in mid-air after a rebuild" this viewer showed.
     */
    void placeBoulders()
    {
        _boulderCount = 0u;
        if (_terrain.height.empty())
            return;

        procgen::Random random = procgen::deriveStream(_options.seed, 0xB0DEu);
        const core::f32 halfWidth = static_cast<core::f32>(_terrain.height.width()) * 0.5f;
        const core::f32 halfDepth = static_cast<core::f32>(_terrain.height.depth()) * 0.5f;
        const core::f32 ceiling = _terrain.high.toFloat() + 3.0f;

        for (core::usize n = 0u; n < _boulderIds.size(); ++n)
        {
            const auto ref = registry().resolve(_boulderIds[n]);
            if (!ref)
                continue;

            const core::f32 x = static_cast<core::f32>(random.below(_terrain.height.width())) - halfWidth;
            const core::f32 z = static_cast<core::f32>(random.below(_terrain.height.depth())) - halfDepth;
            const core::f32 y = ceiling + static_cast<core::f32>(random.below(24u)) * 0.35f;
            const core::f32 size = 0.7f + static_cast<core::f32>(random.below(60u)) * 0.02f;

            const FVec3 at{math::Fixed32::fromFloat(x), math::Fixed32::fromFloat(y), math::Fixed32::fromFloat(z)};
            const FVec3 still{};
            const FVec3 box{math::Fixed32::fromFloat(size), math::Fixed32::fromFloat(size),
                            math::Fixed32::fromFloat(size)};
            // Mass is not decoration: the integrator applies gravity only to bodies
            // whose mass exceeds an epsilon. A body left at zero hangs exactly where
            // it was put, which looks like broken collision and is no gravity at all.
            const math::Fixed32 weight = math::Fixed32::fromFloat(size * size * size * 4.0f);

            for (const auto &partition : registry().partitions())
            {
                if (!partition || !partition->archetype().has(ecs::ComponentId::Velocity))
                    continue;
                const auto &chunks = partition->chunks();
                if (ref->chunkIndex >= chunks.size() || !chunks[ref->chunkIndex])
                    continue;
                const auto &chunk = chunks[ref->chunkIndex];
                const core::u32 i = ref->localIndex;
                if (i >= chunk->count())
                    continue;

                auto *position = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
                auto *positionRead =
                    static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Position)));
                auto *velocity = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
                auto *velocityRead =
                    static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Velocity)));
                auto *extent = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB));
                auto *extentRead =
                    static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::AABB)));
                auto *mass = static_cast<math::Fixed32 *>(chunk->writeComponent(ecs::ComponentId::Mass));
                auto *massRead =
                    static_cast<math::Fixed32 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Mass)));
                if (position == nullptr)
                    break;

                // Both buffers: the first tick reads the front one, which would
                // otherwise still hold uninitialised memory.
                position[i] = at;
                if (positionRead != nullptr)
                    positionRead[i] = at;
                if (velocity != nullptr)
                    velocity[i] = still;
                if (velocityRead != nullptr)
                    velocityRead[i] = still;
                if (extent != nullptr)
                    extent[i] = box;
                if (extentRead != nullptr)
                    extentRead[i] = box;
                if (mass != nullptr)
                    mass[i] = weight;
                if (massRead != nullptr)
                    massRead[i] = weight;
                ++_boulderCount;
                break;
            }
        }
    }

    /**
     * @brief Draws the moving bodies, in one batch.
     *
     * A chunk that has a Velocity component holds bodies; one that does not holds
     * scenery. That distinction is not a convention invented here — it is what the
     * archetype already says, and it is the same test the physics uses to decide what
     * to integrate. Bodies move, so they are re-sent every frame; scenery does not, so
     * it is compiled once (see @ref buildFloraMesh).
     *
     * One glBegin for everything. Wrapping each body in its own begin/end pair costs a
     * driver round trip per body, and that is not a micro-optimisation: drawing seven
     * hundred plants that way, two pairs each, took the frame from nine milliseconds to
     * over a hundred.
     */
    void drawEntities()
    {
        glBegin(GL_QUADS);
        for (const auto &partition : registry().partitions())
        {
            if (!partition)
                continue;
            for (const auto &chunk : partition->chunks())
            {
                if (!chunk)
                    continue;
                if (!partition->archetype().has(ecs::ComponentId::Velocity))
                    continue; // pure decoration: already in the flora list
                const auto *position = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
                const auto *extent = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::AABB));
                const auto *mass = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
                if (position == nullptr)
                    continue;

                const core::u32 count = chunk->count();
                for (core::u32 i = 0u; i < count; ++i)
                {
                    // An obstacle shares this archetype but has zero mass, and it is not
                    // a body: it is a tree that boulders bounce off. Mass is what tells
                    // them apart, and it is the same test the solver makes.
                    if (mass != nullptr && mass[i].raw() == 0)
                        continue;
                    const core::f32 half = extent != nullptr ? extent[i].x.toFloat() * 0.5f : 0.3f;
                    drawCube(position[i].x.toFloat(), position[i].y.toFloat(), position[i].z.toFloat(), half);
                }
            }
        }
        glEnd();
    }

    /**
     * @brief Turns the static scenery into one mesh, so it can be compiled like the terrain.
     *
     * Grazed plants are skipped. The walk order is the same one that filled
     * @c _propPositions, which is what keeps the ecology's index and the registry's
     * row talking about the same plant — and the reason both are built from a single
     * traversal rather than two that merely look alike.
     */
    [[nodiscard]] std::vector<Vertex> buildFloraMesh() const
    {
        std::vector<Vertex> mesh;
        core::usize propIndex = 0u;
        for (const auto &partition : registry().partitions())
        {
            if (!partition)
                continue;
            for (const auto &chunk : partition->chunks())
            {
                if (!chunk)
                    continue;
                const auto *position = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
                const auto *extent = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::AABB));
                const auto *mass = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
                const bool hasMass = partition->archetype().has(ecs::ComponentId::Mass);
                if (position == nullptr)
                    continue;

                const core::u32 count = chunk->count();
                for (core::u32 i = 0u; i < count; ++i)
                {
                    // Everything that does not move is scenery, whether it carries a mass
                    // component or not: decoration has no mass at all, an obstacle has a
                    // mass of exactly zero, and a body has a real one.
                    if (hasMass && mass != nullptr && mass[i].raw() != 0)
                        continue;
                    const core::usize thisPlant = propIndex++;
                    if (_options.living && !_living.plantStanding(thisPlant))
                        continue;
                    const core::f32 half = extent != nullptr ? extent[i].x.toFloat() * 0.5f : 0.3f;
                    appendPlant(mesh, position[i].x.toFloat(), position[i].y.toFloat(), position[i].z.toFloat(), half);
                }
            }
        }
        return mesh;
    }

    /**
     * @brief One plant: a trunk and a canopy, tinted by where it stands.
     *
     * Crude on purpose. What matters is that it has vertical extent and reads as
     * vegetation from a distance; a scatter drawn as flat squares tells you the
     * placement is right and nothing about whether the world looks like a world.
     */
    void appendPlant(std::vector<Vertex> &mesh, core::f32 cx, core::f32 cy, core::f32 cz, core::f32 half) const
    {
        const core::f32 trunkHalf = half * 0.22f;
        const core::f32 trunkTop = cy + half * 1.1f;
        const core::f32 base = cy - half;

        // Tint from the biome it stands in, so a conifer stand and a jungle differ.
        Rgb canopy{0.16f, 0.42f, 0.20f};
        if (!_terrain.biomes.empty())
        {
            const core::f32 halfWidth = static_cast<core::f32>(_terrain.height.width()) * 0.5f;
            const core::f32 halfDepth = static_cast<core::f32>(_terrain.height.depth()) * 0.5f;
            const core::i32 gx = static_cast<core::i32>(cx + halfWidth);
            const core::i32 gz = static_cast<core::i32>(cz + halfDepth);
            if (_terrain.biomes.contains(gx, gz))
            {
                switch (_terrain.biomes.at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)))
                {
                case procgen::BiomeId::Taiga: canopy = {0.13f, 0.33f, 0.25f}; break;
                case procgen::BiomeId::Rainforest: canopy = {0.09f, 0.42f, 0.17f}; break;
                case procgen::BiomeId::Savanna: canopy = {0.50f, 0.47f, 0.21f}; break;
                case procgen::BiomeId::Desert: canopy = {0.30f, 0.50f, 0.28f}; break;
                case procgen::BiomeId::Marsh: canopy = {0.29f, 0.46f, 0.27f}; break;
                default: break;
                }
            }
        }

        const Rgb bark{0.33f, 0.23f, 0.15f};
        const auto quad = [&mesh](float ax, float ay, float az, float bx, float by, float bz, float cx2, float cy2,
                                  float cz2, float dx, float dy, float dz, float nx, float ny, float nz,
                                  const Rgb &colour) {
            const Vertex a{ax, ay, az, nx, ny, nz, colour};
            const Vertex b{bx, by, bz, nx, ny, nz, colour};
            const Vertex c{cx2, cy2, cz2, nx, ny, nz, colour};
            const Vertex d{dx, dy, dz, nx, ny, nz, colour};
            mesh.push_back(a);
            mesh.push_back(b);
            mesh.push_back(c);
            mesh.push_back(a);
            mesh.push_back(c);
            mesh.push_back(d);
        };
        const auto tri = [&mesh](float ax, float ay, float az, float bx, float by, float bz, float cx2, float cy2,
                                 float cz2, float nx, float ny, float nz, const Rgb &colour) {
            mesh.push_back(Vertex{ax, ay, az, nx, ny, nz, colour});
            mesh.push_back(Vertex{bx, by, bz, nx, ny, nz, colour});
            mesh.push_back(Vertex{cx2, cy2, cz2, nx, ny, nz, colour});
        };

        // Trunk: two crossed faces are enough silhouette at this scale.
        quad(cx - trunkHalf, base, cz, cx - trunkHalf, trunkTop, cz, cx + trunkHalf, trunkTop, cz, cx + trunkHalf, base,
             cz, 0.0f, 0.0f, 1.0f, bark);
        quad(cx, base, cz - trunkHalf, cx, trunkTop, cz - trunkHalf, cx, trunkTop, cz + trunkHalf, cx, base,
             cz + trunkHalf, 1.0f, 0.0f, 0.0f, bark);

        // Canopy: a four-sided pyramid, each face with its own normal so it catches
        // the light unevenly and reads as volume rather than as a flat blob.
        const core::f32 spread = half * 1.15f;
        const core::f32 apex = trunkTop + half * 2.1f;
        tri(cx, apex, cz, cx - spread, trunkTop, cz - spread, cx + spread, trunkTop, cz - spread, 0.0f, 0.45f, -0.89f,
            canopy);
        tri(cx, apex, cz, cx + spread, trunkTop, cz - spread, cx + spread, trunkTop, cz + spread, 0.89f, 0.45f, 0.0f,
            canopy);
        tri(cx, apex, cz, cx + spread, trunkTop, cz + spread, cx - spread, trunkTop, cz + spread, 0.0f, 0.45f, 0.89f,
            canopy);
        tri(cx, apex, cz, cx - spread, trunkTop, cz + spread, cx - spread, trunkTop, cz - spread, -0.89f, 0.45f, 0.0f,
            canopy);
    }

    /// One cube, six faces, each flat-shaded so the form reads. Emits into GL_QUADS.
    static void drawCube(core::f32 cx, core::f32 cy, core::f32 cz, core::f32 half)
    {
        const core::f32 x0 = cx - half;
        const core::f32 x1 = cx + half;
        const core::f32 y0 = cy - half;
        const core::f32 y1 = cy + half;
        const core::f32 z0 = cz - half;
        const core::f32 z1 = cz + half;

        glColor3f(0.93f, 0.62f, 0.18f);
        glVertex3f(x0, y1, z0);
        glVertex3f(x1, y1, z0);
        glVertex3f(x1, y1, z1);
        glVertex3f(x0, y1, z1);
        glColor3f(0.55f, 0.35f, 0.10f);
        glVertex3f(x0, y0, z0);
        glVertex3f(x0, y0, z1);
        glVertex3f(x1, y0, z1);
        glVertex3f(x1, y0, z0);
        glColor3f(0.80f, 0.50f, 0.14f);
        glVertex3f(x0, y0, z1);
        glVertex3f(x0, y1, z1);
        glVertex3f(x1, y1, z1);
        glVertex3f(x1, y0, z1);
        glColor3f(0.68f, 0.43f, 0.12f);
        glVertex3f(x1, y0, z0);
        glVertex3f(x1, y1, z0);
        glVertex3f(x0, y1, z0);
        glVertex3f(x0, y0, z0);
        glColor3f(0.74f, 0.47f, 0.13f);
        glVertex3f(x1, y0, z1);
        glVertex3f(x1, y1, z1);
        glVertex3f(x1, y1, z0);
        glVertex3f(x1, y0, z0);
        glColor3f(0.62f, 0.39f, 0.11f);
        glVertex3f(x0, y0, z0);
        glVertex3f(x0, y1, z0);
        glVertex3f(x0, y1, z1);
        glVertex3f(x0, y0, z1);
    }

    /**
     * @brief Draws the streaming plan the camera would produce, as chunk outlines.
     *
     * The policy has two properties a fold cannot show and a picture can: the
     * release radius is WIDER than the generate radius, so the two rings are
     * visibly different sizes; and the budgeted chunks lie ahead of the heading
     * rather than being scattered. Both are visible at a glance, and both were
     * only assertions in a test until now.
     */
    void drawStreamingPlan() const
    {
        if (_terrain.height.empty())
            return;

        procgen::StreamingParams params;
        params.generateRadius = 3u;
        params.maxGeneratePerTick = 6u;

        // The camera's ground position and the direction it is looking, which is
        // what a real streamer would be fed.
        procgen::GenerationSource source;
        source.x = math::Fixed32::fromFloat(0.0f);
        source.z = math::Fixed32::fromFloat(0.0f);
        source.headingX = math::Fixed32::fromFloat(-std::sin(_camera.yaw));
        source.headingZ = math::Fixed32::fromFloat(-std::cos(_camera.yaw));

        const procgen::StreamingPlan plan = procgen::planStreaming(&source, 1u, nullptr, 0u, params);
        const float chunk = 16.0f;
        const float y = _terrain.high.toFloat() + 2.0f;

        glDisable(GL_DEPTH_TEST);
        glBegin(GL_LINES);
        for (core::u32 i = 0u; i < plan.toGenerate.size(); ++i)
        {
            // The first chunks the budget admits: what would actually be built this
            // tick, in amber. Everything else in the wanted region stays outlined.
            glColor3f(1.0f, 0.68f, 0.16f);
            const float x0 = static_cast<float>(plan.toGenerate[i].coord.x) * chunk;
            const float z0 = static_cast<float>(plan.toGenerate[i].coord.z) * chunk;
            const float x1 = x0 + chunk;
            const float z1 = z0 + chunk;
            glVertex3f(x0, y, z0);
            glVertex3f(x1, y, z0);
            glVertex3f(x1, y, z0);
            glVertex3f(x1, y, z1);
            glVertex3f(x1, y, z1);
            glVertex3f(x0, y, z1);
            glVertex3f(x0, y, z1);
            glVertex3f(x0, y, z0);
        }
        glEnd();
        glEnable(GL_DEPTH_TEST);
    }

    /// Recompiles the display lists for everything that does not move.
    void compileLists()
    {
        _surfaceList.compile(_surfaceMesh, &drawMesh, false);
        _surfaceGhostList.compile(_surfaceMesh, &drawMeshGhosted, false);
        _undergroundList.compile(_undergroundMesh, &drawMesh, false);
        _townList.compile(_townMesh, &drawMesh, false);
        _floraList.compile(_floraMesh, &drawMesh, false);
        _roadsideList.compile(_roadsideMesh, &drawMesh, false);
        _liminalList.compile(_liminalMesh, &drawMesh, false);

        // A display list that failed to build draws nothing and says nothing — a blank
        // view with no error anywhere. Checking once per rebuild is enough to turn that
        // into a message.
        const GLenum failure = glGetError();
        if (failure != GL_NO_ERROR)
            core::Log::warn("mapview: the driver refused a display list, the view will be incomplete");
    }

    void pumpEvents()
    {
        Display *display = _viewport.display();
        if (display == nullptr)
            return;

        bool rebuild = false;
        bool remesh = false;

        while (XPending(display) > 0)
        {
            XEvent event;
            XNextEvent(display, &event);

            if (event.type == ConfigureNotify)
            {
                _viewport.resize(event.xconfigure.width, event.xconfigure.height);
            }
            else if (event.type == ButtonPress)
            {
                if (event.xbutton.button == Button1)
                {
                    _dragging = true;
                    _lastX = event.xbutton.x;
                    _lastY = event.xbutton.y;
                }
                else if (event.xbutton.button == Button4)
                    _camera.distance *= 0.9f;
                else if (event.xbutton.button == Button5)
                    _camera.distance *= 1.1f;
            }
            else if (event.type == ButtonRelease && event.xbutton.button == Button1)
            {
                _dragging = false;
            }
            else if (event.type == MotionNotify && _dragging)
            {
                _camera.yaw += static_cast<float>(event.xmotion.x - _lastX) * 0.008f;
                _camera.pitch += static_cast<float>(event.xmotion.y - _lastY) * 0.008f;
                if (_camera.pitch < -1.50f)
                    _camera.pitch = -1.50f;
                if (_camera.pitch > 1.55f)
                    _camera.pitch = 1.55f;
                _lastX = event.xmotion.x;
                _lastY = event.xmotion.y;
            }
            else if (event.type == KeyPress)
            {
                const KeySym key = XLookupKeysym(&event.xkey, 0);
                switch (key)
                {
                case XK_q:
                case XK_Escape:
                    if (_engine != nullptr)
                        _engine->requestShutdown();
                    break;
                case XK_n:
                    _options.seed = _options.seed * 1664525u + 1013904223u;
                    rebuild = true;
                    break;
                case XK_s:
                    _options.shading = static_cast<Shading>((static_cast<int>(_options.shading) + 1) %
                                                            static_cast<int>(Shading::Count));
                    remesh = true;
                    break;
                case XK_1:
                    _options.noise = procgen::NoiseKind::Fbm;
                    rebuild = true;
                    break;
                case XK_2:
                    _options.noise = procgen::NoiseKind::Ridged;
                    rebuild = true;
                    break;
                case XK_3:
                    _options.noise = procgen::NoiseKind::Billow;
                    rebuild = true;
                    break;
                case XK_e:
                    _options.erosion = !_options.erosion;
                    rebuild = true;
                    break;
                case XK_r:
                    _options.rivers = !_options.rivers;
                    rebuild = true;
                    break;
                case XK_w:
                    _options.warp = !_options.warp;
                    rebuild = true;
                    break;
                case XK_t:
                    _options.terraces = !_options.terraces;
                    rebuild = true;
                    break;
                case XK_g:
                    _options.settlement = !_options.settlement;
                    rebuild = true;
                    break;
                case XK_h:
                    _options.roads = !_options.roads;
                    rebuild = true;
                    break;
                case XK_m:
                    _options.metric = static_cast<procgen::DistanceMetric>((static_cast<int>(_options.metric) + 1) % 3);
                    rebuild = true;
                    break;
                case XK_k:
                    _options.windDirection = (_options.windDirection + 1u) % 4u;
                    rebuild = true;
                    break;
                case XK_c:
                    _options.caveKind = (_options.caveKind + 1u) % 4u;
                    // Look at what just changed. Cycling the underground while the
                    // camera is on the surface changes nothing you can see, which
                    // reads as a key that does not work.
                    if (_options.view == View::Surface)
                        _options.view = View::Cutaway;
                    rebuild = true;
                    break;
                case XK_x:
                    _options.climateAxis = (_options.climateAxis + 1u) % procgen::kClimateAxisCount;
                    _options.shading = Shading::Climate;
                    remesh = true;
                    break;
                case XK_u:
                    _options.grammarBuildings = !_options.grammarBuildings;
                    rebuild = true;
                    break;
                case XK_l: _options.living = !_options.living; break;
                case XK_j: _options.chunkOverlay = !_options.chunkOverlay; break;
                case XK_v:
                    _options.view =
                        static_cast<View>((static_cast<int>(_options.view) + 1) % static_cast<int>(View::Count));
                    break;
                case XK_f: _options.wireframe = !_options.wireframe; break;
                case XK_b: _options.vegetation = !_options.vegetation; break;
                case XK_o: _options.water = !_options.water; break;
                case XK_space: rebuild = true; break;
                case XK_bracketright:
                    if (_options.size < 320u)
                    {
                        _options.size += 32u;
                        rebuild = true;
                    }
                    break;
                case XK_bracketleft:
                    if (_options.size > 48u)
                    {
                        _options.size -= 32u;
                        rebuild = true;
                    }
                    break;
                default: break;
                }
            }
        }

        if (rebuild)
            regenerate();
        else if (remesh)
        {
            _surfaceMesh = buildSurfaceMesh(_terrain, _options);
            compileLists();
        }
    }

    /// How many loose bodies the world carries.
    static constexpr core::u32 kBoulders = 240u;

    Viewport &_viewport;
    Options _options;
    TerrainData _terrain;
    /// How far below the surface the caves are carved.
    static constexpr float kCaveDepth = 7.0f;
    /// Vertical spacing between two floors of the layered system.
    static constexpr float kCaveLayerSpacing = 5.0f;

    std::vector<Vertex> _surfaceMesh;
    std::vector<Vertex> _undergroundMesh;
    std::vector<Vertex> _townMesh;
    std::vector<Vertex> _floraMesh;
    std::vector<Vertex> _roadsideMesh;
    std::vector<Vertex> _liminalMesh;
    MeshList _surfaceList;
    MeshList _surfaceGhostList;
    MeshList _undergroundList;
    MeshList _townList;
    MeshList _floraList;
    MeshList _roadsideList;
    MeshList _liminalList;
    LivingLayer _living;
    Camera _camera;
    engine::Engine *_engine{nullptr};
    TerrainCollisionSystem *_collision{nullptr};
    lpl::pmr::vector<ecs::EntityId> _propIds;
    lpl::pmr::vector<ecs::EntityId> _boulderIds;
    lpl::pmr::vector<FVec3> _propPositions;
    core::u32 _boulderCount{0u};
    core::u32 _frames{0u};
    core::u32 _steps{0u};
    double _tickMilliseconds{0.0};
    core::u32 _obstacleFold{0u};
    double _startedAt{nowMilliseconds()};
    double _frameMilliseconds{0.0};
    bool _dragging{false};
    int _lastX{0};
    int _lastY{0};
};

} // namespace

int main()
{
    Viewport viewport;
    if (!viewport.open())
        return 1;

    // A solo client. Physics on, because the world has loose bodies in it; rendering
    // off, because the engine's renderer is not what is drawing here; networking off,
    // because there is nobody to agree with.
    engine::Config config = engine::Config::Builder{}
                                .tickRate(60u)
                                .enablePhysics(true)
                                .enableRendering(false)
                                .enableNetworking(false)
                                .headless(false)
                                .build();

    Options options;
    auto world = lpl::pmr::make_unique<MapviewWorld>(viewport, options);
    MapviewWorld &worldRef = *world;

    engine::Engine engine{config, static_cast<lpl::pmr::unique_ptr<engine::World>>(std::move(world))};
    worldRef.bindEngine(engine);

    if (auto started = engine.init(); !started)
    {
        std::fprintf(stderr, "mapview: engine init failed\n");
        viewport.close();
        return 1;
    }

    std::printf("lpl-mapview: engine hosting a generated world. Keys are listed in the window.\n");
    engine.run();
    engine.shutdown();

    viewport.close();
    return 0;
}
