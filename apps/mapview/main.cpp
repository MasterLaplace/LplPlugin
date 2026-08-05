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
#include <lpl/ai/AntColony.hpp>
#include <lpl/ai/Swarm.hpp>
#include <lpl/ecology/Genome.hpp>
#include <lpl/ecology/Herd.hpp>
#include <lpl/ecology/Populations.hpp>
#include <lpl/ecology/Vegetation.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Entity.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/System.hpp>
#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/engine/Engine.hpp>
#include <lpl/engine/GridTerrain.hpp>
#include <lpl/engine/systems/HeightfieldCollisionSystem.hpp>
#include <lpl/engine/ITerrainQuery.hpp>
#include <lpl/engine/LivingLayer.hpp>
#include <lpl/engine/systems/CreatureSystems.hpp>
#include <lpl/engine/World.hpp>
#include <lpl/image/Font8x16.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/CaveSystem.hpp>
#include <lpl/procgen/Climate.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/procgen/Extrusion.hpp>
#include <lpl/math/FixedMath.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/Hydrology.hpp>
#include <lpl/procgen/Liminal.hpp>
#include <lpl/procgen/Settlement.hpp>
#include <lpl/procgen/ShapeGrammar.hpp>
#include <lpl/procgen/Streaming.hpp>
#include <lpl/procgen/ValueNoise.hpp>
#include <lpl/procgen/Voronoi.hpp>
#include <lpl/procgen/MapMesh.hpp>
#include <lpl/procgen/MapShading.hpp>
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

/// How the surface is coloured — procgen::MapShading, which the editor shares.
using Shading = procgen::MapShading;

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

/// Everything the generator produced, plus what only a viewer adds.
///
/// It IS a procgen::WorldAtlas, extended — not a parallel copy of one. This struct
/// used to be that copy: its own names for counters that live in
/// procgen::BuiltWorldStats, its own flattened duplicates of two
/// procgen::DrainageNetwork fields, and its own libm logarithm where the module
/// already had math::fixedLog2. Deriving from the atlas is what makes the
/// shading functions shareable with the editor, since they can now take the type
/// both tools hold.
///
/// What stays here is what the builder did not produce: the liminal sector is its
/// own generator on its own seed, and the wall-clock timing is a fact about this
/// process rather than about the world.
struct TerrainData : procgen::WorldAtlas {
    procgen::LiminalSpace liminal;
    core::u32 liminalSectors{0u};
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

/**
 * @brief The viewer's world, AS A RECIPE.
 *
 * Two hundred and twenty lines of hand-written @c WorldBuilder calls used to live
 * here, and that was the last piece of engine knowledge stranded in this file. It
 * could not leave until the recipe could NAME what the viewer was asking for:
 * Voronoi provinces, terracing, three of the four underground generators, the
 * building grammar, the roadside L-system, the lift that puts the ground above the
 * physics floor, and — the one that blocked it longest — a sixth scatter rule,
 * against a ceiling of four.
 *
 * What that costs is nothing and what it buys is everything the format is for: this
 * world can now be saved to a `.lplscene`, baked into a cartridge, replayed in ring 0
 * and asked for by Caine, because it is the same pipeline the parity gate runs rather
 * than a second one that resembles it.
 *
 * The pass ORDER is no longer decided here either. It is @c procgen::applyRecipe's,
 * which is the point: two callers ordering erosion and rivers differently produce two
 * different worlds from the same description, and nothing would say which was meant.
 */
[[nodiscard]] procgen::WorldRecipe recipeFor(const Options &options)
{
    procgen::WorldRecipe recipe;

    recipe.seed = options.seed;
    recipe.width = options.size;
    recipe.depth = options.size;

    recipe.terrain.seed = options.seed;
    recipe.terrain.kind = options.noise;
    recipe.terrain.warpStrength = options.warp ? 8.0f : 0.0f;
    // The viewer's own framing: it looks at a map from above, so it takes the noise as
    // it comes rather than rescaling it into a fixed band.
    recipe.normalizeTerrain = false;

    recipe.terraceSteps = options.terraces ? 8u : 0u;
    recipe.erodeTerrain = options.erosion;
    recipe.carveRivers = options.rivers;

    // Half a unit is where the built-in physics stops a body, and it knows nothing
    // about a heightfield — so a world dipping below that line has two floors. One
    // and a half leaves room for the deepest river bed the erosion cuts.
    recipe.groundClearance = 1.5f;

    recipe.climate.windDirection = options.windDirection;

    recipe.partitionRegions = true;
    recipe.provinces.cellSize = options.size / 6u == 0u ? 6u : options.size / 6u;
    recipe.provinces.metric = options.metric;
    recipe.provinces.warpStrength = options.warp ? 6.0f : 0.0f;

    recipe.placeSettlement = options.settlement;
    recipe.settlement.districtSize = 14u;

    recipe.growRoads = options.roads;
    recipe.roads.iterations = 6u;
    recipe.roads.stepLength = 3u;

    // Underground, generated alongside so the view can be toggled without a rebuild.
    // The layered system is a stack of plans joined by shafts, at least one of which
    // pierces the surface; the three flat kinds are buried voids with no way in, which
    // is exactly what looking at them makes obvious and what a floor count never did.
    switch (options.caveKind % 4u)
    {
    case 1u:
        recipe.caveKind = procgen::CaveKind::Bsp;
        recipe.rooms.width = options.size;
        recipe.rooms.depth = options.size;
        recipe.rooms.seed = options.seed;
        break;
    case 2u:
        recipe.caveKind = procgen::CaveKind::Dla;
        recipe.aggregation.width = options.size;
        recipe.aggregation.depth = options.size;
        recipe.aggregation.seed = options.seed;
        recipe.aggregation.particles = options.size * 8u;
        break;
    case 3u:
        recipe.caveKind = procgen::CaveKind::Layered;
        recipe.caveSystem.width = options.size;
        recipe.caveSystem.depth = options.size;
        recipe.caveSystem.seed = options.seed;
        recipe.caveSystem.layers = 3u;
        recipe.caveSystem.entrances = 3u;
        break;
    default:
        recipe.caveKind = procgen::CaveKind::Cellular;
        recipe.caves.width = options.size;
        recipe.caves.depth = options.size;
        recipe.caves.seed = options.seed;
        break;
    }

    // The town's third dimension. `extrudeTown` gives prisms and the viewer used to
    // draw its own boxes on top of the footprints; the grammar gives a base course,
    // storeys and a roof, which is the difference between a bar chart and something
    // that reads as architecture.
    //
    // Three storeys, not five: the grid is one world unit per cell, so a plot is three
    // or four units across and an eight-level volume reads as a tower block dropped on
    // a village.
    recipe.raiseBuildings = options.settlement && options.grammarBuildings;
    recipe.buildings.minFloors = 1u;
    recipe.buildings.maxFloors = 3u;
    recipe.buildings.roofHeight = 1u;

    // The linear form of the same grammar: two fence posts to one lamp, then a gap.
    // The point is that one parser serves both.
    if (options.roads && options.grammarBuildings)
    {
        const char pattern[] = "{[A,P]:2,[BL,P]:1}*,[G,P]";
        for (core::u32 i = 0u; i < sizeof(pattern); ++i)
            recipe.roadsidePattern[i] = pattern[i];
        recipe.roadsideLevels = 3u;
    }

    // Vegetation: one rule per biome, so a forest is conifers and a savanna is scrub.
    // Six of them — the case that could not be written as a recipe at all until the
    // ceiling moved, and the reason this whole function had to stay hand-written.
    if (options.vegetation)
    {
        struct Species {
            procgen::BiomeId biome;
            core::f32 density;
            core::f32 halfExtent;
            core::f32 maxSlope;
            bool collidable;
        };
        static constexpr Species kSpecies[] = {
            {procgen::BiomeId::Taiga, 0.16f, 0.42f, 1.6f, true},
            {procgen::BiomeId::Forest, 0.18f, 0.5f, 1.6f, true},
            {procgen::BiomeId::Rainforest, 0.22f, 0.55f, 1.8f, true},
            {procgen::BiomeId::Savanna, 0.05f, 0.3f, 0.0f, false},
            {procgen::BiomeId::Desert, 0.03f, 0.22f, 0.0f, false},
            {procgen::BiomeId::Marsh, 0.2f, 0.25f, 0.0f, false},
        };
        static_assert(sizeof(kSpecies) / sizeof(kSpecies[0]) <= procgen::kMaxScatterRules,
                      "one scatter rule per plant, and the recipe has to hold them all");

        for (const Species &species : kSpecies)
        {
            procgen::ScatterRule &rule = recipe.scatter[recipe.scatterCount++];
            rule.biome = species.biome;
            rule.density = species.density;
            rule.halfExtent = species.halfExtent;
            if (species.maxSlope > 0.0f)
                rule.maxSlope = species.maxSlope;
            rule.collidable = species.collidable;
        }
    }

    return recipe;
}

TerrainData generateTerrain(const Options &options, ecs::Registry *registry,
                            lpl::pmr::vector<ecs::EntityId> *outPropIds)
{
    const double started = nowMilliseconds();

    // 2.2 rather than the rule's default 2.4 because this is THE walkability mask of
    // this viewer, the one the living layer walks on — and the snapshot's own header
    // says why there must be exactly one: three notions of "blocked" is how an animal
    // ends up standing in a lake the scent flows around.
    //
    // Sea level is left at the recipe's own, NOT at a number written here: buildAtlas
    // shifts both the rule and the reported sea level by whatever lift the ground
    // clearance actually applied, which is only knowable after erosion has run.
    procgen::WorldRecipe recipe = recipeFor(options);
    TerrainData world;
    static_cast<procgen::WorldAtlas &>(world) = procgen::buildAtlas(
        recipe, registry, outPropIds, procgen::WalkabilityRule{recipe.biomes.seaLevel, 2.2f});
    world.buildMilliseconds = nowMilliseconds() - started;
    return world;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour
// ─────────────────────────────────────────────────────────────────────────────

// The palette, the ramps and the seven modes live in procgen::MapShading now —
// one definition, two tools. They were about to be written a second time for the
// editor, which is the pattern this repository has paid for eight times.
using Rgb = procgen::Rgb;


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

// The six meshers used to live here — six hundred lines of them. They are
// procgen::MapMesh now, for the reason that matters more than tidiness: code in an
// app has NO TEST TARGET. A mesher that winds a face inside out, drops a boundary
// quad or shears a building into a staircase could only be caught by a human
// looking at a picture, and all three of those happened. In a module a vertex
// count is arithmetic and a fold is a fold — see test-map-mesh.
//
// What stays in this file is the plant mesher, which is the one piece of geometry
// that reads the REGISTRY rather than a generated grid.
using Vertex = procgen::MapVertex;

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

// MapTerrain used to live here: engine::ITerrainQuery over this viewer's bounded
// grid. It is engine::GridTerrain now, because the editor needs exactly the same
// answers and a second copy of them would have drifted — which is what the
// creature loop in this very file did before it was folded back.
using MapTerrain = engine::GridTerrain;

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
     * @brief The layer's own state, and what it delegates.
     *
     * A plant is @c ecology::PlantCell and an animal is an ENTITY: both used to be
     * structs declared right here, alongside a copy of flocking, scent following,
     * grazing and locomotion. The copy had drifted BOTH ways — it had learned
     * avoidance and stray containment the engine's systems lacked, and it had never
     * learned that a scent channel means something, so a grazer climbed the very
     * channel grazers deposit and followed itself instead of fleeing the wolf.
     *
     * Two implementations of one idea do not stay equal; they take turns being
     * right. What is left in this class is the ant colony, which the engine's layer
     * has no notion of, and the diagnostics, which are genuinely a viewer's.
     */
    /// Rebuilds the layer against a freshly generated world.
    void reset(const TerrainData &world, core::u32 seed, const lpl::pmr::vector<math::Vec3<math::Fixed32>> &props)
    {
        _width = world.height.width();
        _depth = world.height.depth();
        if (_width == 0u || _depth == 0u)
            return;

        // Obstacles: below the water line, or too steep to walk. Marking them is
        // what makes the field flow around the map instead of through it — the
        // diffusion step refuses to cross a blocked cell in either direction.
        // ONE mask, three consumers: the field diffuses around it, the herd refuses
        // to walk into it, and the spawn refuses to start inside it. Three separate
        // notions of "blocked" is how a creature ends up standing in a wall that the
        // pheromone flows around.
        //
        // The terrain half comes from the atlas, which computed it from the same
        // sea level and slope limit while the builder was still alive. It was
        // recomputed here for as long as the two structs were separate, which is
        // two notions of "blocked" that merely happened to agree.
        _blocked = world.blocked;

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

        // ── The living layer, as the engine's ───────────────────────────────
        //
        // ONE field, six named channels. This file used to open its own
        // two-channel field and read channel 1 uphill for every animal — so a
        // grazer climbed the channel grazers deposit and was attracted to itself
        // instead of fleeing the wolf. The ants keep their own channel, and it is
        // the one ai::ScentChannel already names for them: Pheromone.
        engine::LivingLayerParams living;
        living.maxBodies = kMaxBodies;
        living.speciesCount = 2u;
        living.scentSpan = _width < _depth ? _width : _depth;
        living.scentLayers = 6u;
        living.webPeriod = kWebPeriod;

        ecology::LivingRecipe recipe = ecology::parityLivingRecipe();
        recipe.regrowthTicks = kRegrowthTicks;
        recipe.stigmergy = _stigmergy;

        _terrain.reset(&_blocked, _width, _depth, kRegrowthTicks);
        _living.configure(living, recipe, seed);
        _living.bind(_registry);
        _living.openScent(living.scentSpan, living.scentLayers);
        _living.scent().centreOn(0, 0);
        _living.scent().field().setObstacles(_blocked);

        core::u32 blockedCells = 0u;
        for (core::u32 i = 0u; i < _blocked.cellCount(); ++i)
            if (_blocked[i] != 0u)
                ++blockedCells;
        std::printf("  living: %u of %u cells blocked (%.0f%%)\n", blockedCells, _blocked.cellCount(),
                    100.0 * static_cast<double>(blockedCells) / static_cast<double>(_blocked.cellCount()));

        // The nest goes where the town is, when there is one: a colony foraging out
        // of a settlement reads as something happening in the world rather than as a
        // demo running beside it. WHERE is this viewer's call; the colony itself is
        // ai::AntColony, because holding the agents and the rule that sends them home
        // is the one part of ant colony optimisation the module was missing.
        core::u32 nestX = _width / 2u;
        core::u32 nestZ = _depth / 2u;
        if (!world.settlement.empty())
            for (core::u32 z = 0u; z < _depth; ++z)
                for (core::u32 x = 0u; x < _width; ++x)
                    if (world.settlement.at(x, z) == procgen::SettlementCell::Plaza)
                    {
                        nestX = x;
                        nestZ = z;
                        z = _depth;
                        break;
                    }

        ai::AntColonyParams colony;
        colony.agents = kAgents;
        colony.forageRange = kForageRange;
        colony.seed = seed;
        colony.ants = _ants;
        _colony.reset(_living.scent().field(), _width, _depth, colony, nestX, nestZ);

        // The vegetation, taken from what the scatter actually placed. Capacity is
        // the map's own carrying capacity in the literal sense: how many plants fit
        // on it, which is a fact about the terrain rather than a tuning constant.
        //
        // World cells, not grid cells: ecology::PlantCell is documented as signed
        // world coordinates, because a streamed world has no corner to count from.
        const float halfWidthGrid = static_cast<float>(_width) * 0.5f;
        const float halfDepthGrid = static_cast<float>(_depth) * 0.5f;
        for (core::usize i = 0u; i < props.size(); ++i)
        {
            (void) halfWidthGrid;
            (void) halfDepthGrid;
            _terrain.addPlant(static_cast<core::i32>(props[i].x.toFloat()), static_cast<core::i32>(props[i].z.toFloat()));
        }
        // The producer level is COUNTED, not integrated: it is the standing
        // vegetation this seed actually grew.
        _living.seedWeb(_terrain.standingPlants());
        _ticks = 0u;
        _births = 0u;
        _deaths = 0u;
        _measuredSpeed = 0.0;
        _speedIds.clear();
        _speedAt.clear();

        // The six systems, in the order engine::systems::CreatureStage states.
        // Rebuilt per world because the scent window is reopened. This viewer used
        // to construct and step them itself, which was the SECOND place the order
        // was written down — and a wrong order here is silent: flocking before
        // steering overwrites the scent impulse and the pack just stops flanking.
        _creatures.build(_registry, _living, _terrain);

        // The bodies. Head counts come from the web rather than from a constant:
        // what is on screen is then the population the model says exists, and a
        // collapse is something you watch happen rather than read in a number.
        _living.reconcile(0u, [this](math::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
            return proposeSpawn(r, a, x, z);
        });
    }

    /// Where the pheromone window sits, for the creature systems.
    [[nodiscard]] engine::systems::CreatureFieldView fieldView() const noexcept { return _living.fieldView(); }

    /**
     * @brief Proposes somewhere an animal could actually stand.
     *
     * Dropping a herd into the sea and letting the flocking rules sort it out
     * produces a very confident-looking shoal of deer.
     */
    [[nodiscard]] bool proposeSpawn(math::Random &random, core::u32 attempt, math::Fixed32 &outX,
                                    math::Fixed32 &outZ) const
    {
        (void) attempt;
        if (_width == 0u)
            return false;
        const core::u32 x = random.below(_width);
        const core::u32 z = random.below(_depth);
        if (_blocked.at(x, z) != 0u)
            return false;
        outX = math::Fixed32::fromInt(static_cast<core::i32>(x) - static_cast<core::i32>(_width / 2u));
        outZ = math::Fixed32::fromInt(static_cast<core::i32>(z) - static_cast<core::i32>(_depth / 2u));
        return true;
    }

    /**
     * @brief Mean ground speed of the herd, in cells per second.
     *
     * Distance travelled, not the speed it was asked to travel at: a body pinned
     * against a rock has a perfectly healthy velocity and goes nowhere, and only one
     * of those two numbers would have caught it. It is measured rather than asserted
     * because the first version of this layer moved bodies by their raw boid velocity
     * per tick — a plausible-looking line that means sixty cells a second.
     */
    void measureSpeed()
    {
        const ecology::Herd &herd = _living.herd();
        if (herd.empty())
        {
            _measuredSpeed = 0.0;
            return;
        }
        if (_speedIds.size() != herd.size())
        {
            _speedIds.clear();
            _speedAt.clear();
            for (core::u32 i = 0u; i < herd.size(); ++i)
            {
                _speedIds.push_back(herd.at(i));
                _speedAt.push_back(positionOf(herd.at(i)));
            }
            return;
        }
        if (_ticks % 60u != 0u)
            return;

        double total = 0.0;
        core::u32 counted = 0u;
        for (core::u32 i = 0u; i < _speedIds.size(); ++i)
        {
            const math::Vec3<math::Fixed32> at = positionOf(_speedIds[i]);
            const double dx = at.x.toFloat() - _speedAt[i].x.toFloat();
            const double dz = at.z.toFloat() - _speedAt[i].z.toFloat();
            total += std::sqrt(dx * dx + dz * dz);
            ++counted;
            _speedAt[i] = at;
        }
        _measuredSpeed = counted != 0u ? total / static_cast<double>(counted) : 0.0;
    }

    /// Where one body is, read from the registry rather than from a second list.
    [[nodiscard]] math::Vec3<math::Fixed32> positionOf(ecs::EntityId id) const
    {
        // ecs::Registry::chunkOf, which walks the partitions AND checks that the chunk
        // really holds this entity. That walk was written out by hand here and in
        // ecology::Herd; a third copy was about to appear in a test.
        core::u32 row = 0u;
        ecs::Chunk *chunk = _registry.chunkOf(id, row);
        if (chunk == nullptr || !chunk->archetype().has(ecs::ComponentId::Creature))
            return {};
        // The WRITE side, like every creature system: nothing here swaps buffers, so
        // the front one holds whatever a component was born with.
        const auto *positions =
            static_cast<const math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Position));
        return positions != nullptr ? positions[row] : math::Vec3<math::Fixed32>{};
    }

    /// Brings the bodies in line with the census; engine::LivingLayer keeps the ratio.
    void reconcilePopulation()
    {
        const core::u32 before = _living.herd().size();
        _living.reconcile(_ticks, [this](math::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
            return proposeSpawn(r, a, x, z);
        });
        const core::u32 after = _living.herd().size();
        if (after > before)
            _births += after - before;
        else
            _deaths += before - after;
    }

    /// One simulation step: the agents move and deposit, then the field decays.
    void step(const TerrainData &world)
    {
        if (_width == 0u)
            return;
        ++_ticks;

        _colony.step(_living.scent().field());

        // The animals, as systems, in the order the scheduler derives: mark, forget,
        // steer, flock, graze, walk. Nothing about an animal is a loop in this file
        // any more — which is the point, because the loop that used to be here had
        // drifted from the engine's in both directions.
        (void) world;
        _creatures.step(0.016f);

        _living.setProducerPopulation(_terrain.tickVegetation());
        measureSpeed();

        // Demography on its own clock, once a second rather than sixty times.
        // A Lotka-Volterra step is a GENERATION, not a frame: stepped at the tick
        // rate the web ran a lifetime between two log lines and the curve read as
        // noise. The parity recipe folds forty-eight steps for the same reason —
        // that is a run, not a second.
        if (_ticks % kWebPeriod == 0u)
        {
            _living.stepWeb(1u);
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
                const float strength = _living.scent().field().value(kAntChannel, x, z).toFloat();
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
        for (core::u32 i = 0u; i < _colony.agentCount(); ++i)
        {
            const core::u32 ax = _colony.agentX(i);
            const core::u32 az = _colony.agentZ(i);
            const float y = world.height.at(ax, az).toFloat() + 0.45f;
            const float x0 = static_cast<float>(ax) - halfW + 0.30f;
            const float z0 = static_cast<float>(az) - halfD + 0.30f;
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
        if (world.height.empty())
            return;

        const float halfW = static_cast<float>(_width) * 0.5f;
        const float halfD = static_cast<float>(_depth) * 0.5f;

        // The species mean, recomputed here rather than cached: it moves as the
        // herd breeds, and a stale mean would mark the wrong animals.
        _statScratch.clear();
        forEachBody([this](core::u32 species, core::u32, const ecology::Genome &genome,
                           const math::Vec3<math::Fixed32> &) {
            if (species == 0u)
                _statScratch.push_back(genome);
        });
        const ecology::PopulationStats stats = ecology::strengthStats(_statScratch.empty() ? nullptr : &_statScratch[0],
                                                                     static_cast<core::u32>(_statScratch.size()));
        ecology::HeredityParams heredity;

        glBegin(GL_QUADS);
        forEachBody([&](core::u32 species, core::u32 id, const ecology::Genome &genome,
                        const math::Vec3<math::Fixed32> &at) {
            const core::i32 gx = static_cast<core::i32>(at.x.toFloat() + halfW);
            const core::i32 gz = static_cast<core::i32>(at.z.toFloat() + halfD);
            if (!world.height.contains(gx, gz))
                return;
            const float ground = world.height.at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)).toFloat();

            const ai::PersonalityTraits traits = ai::personalityOf(id, species);
            const float half = 0.22f * genome.size.toFloat();

            Rgb colour = species == 1u ? Rgb{0.72f, 0.20f, 0.18f} : Rgb{0.80f, 0.66f, 0.36f};
            // Temperament is visible, faintly: an aggressive animal runs hot.
            colour.r += 0.18f * traits.aggression.toFloat();
            colour.b += 0.12f * traits.nervousness.toFloat();
            if (species == 0u && stats.count > 4u && ecology::isAnomaly(genome, stats, heredity))
                colour = {0.98f, 0.94f, 0.86f};

            const float cx = at.x.toFloat();
            const float cz = at.z.toFloat();
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
        });
        glEnd();
    }

    /**
     * @brief Walks the bodies the way a system does: by chunk, through the registry.
     *
     * One traversal, three consumers (the stats pass, the draw pass, the census).
     * Writing it three times is how a viewer ends up disagreeing with itself about
     * which animals exist.
     */
    /// Every body, one at a time. engine::systems::forEachCreature does the walk:
    /// this file had its own copy, and a third was about to be written.
    template <typename Visit> void forEachBody(Visit &&visit) const
    {
        engine::systems::forEachCreature(_registry, visit);
    }

    [[nodiscard]] core::u32 aliveCount(core::u32 species) const noexcept
    {
        return _living.herd().countSpecies(species);
    }

    [[nodiscard]] double meanSpeed() const { return _measuredSpeed; }
    [[nodiscard]] core::u32 births() const noexcept { return _births; }
    [[nodiscard]] core::u32 deaths() const noexcept { return _deaths; }
    /// Steps the terrain refused outright on the last tick.
    [[nodiscard]] core::u32 refusals() const noexcept { return _creatures.locomotion() != nullptr ? _creatures.locomotion()->cornered() : 0u; }
    /// Bodies recovered from somewhere they could not stand.
    [[nodiscard]] core::u32 strays() const noexcept { return _creatures.locomotion() != nullptr ? _creatures.locomotion()->strays() : 0u; }
    /// Bodies that steered AROUND an obstacle rather than into it.
    [[nodiscard]] core::u32 avoided() const noexcept { return _creatures.locomotion() != nullptr ? _creatures.locomotion()->avoided() : 0u; }
    [[nodiscard]] core::u32 standingPlants() const noexcept { return _terrain.standingPlants(); }
    [[nodiscard]] core::u32 plantCount() const noexcept { return _terrain.plantCount(); }
    [[nodiscard]] core::u32 grazed() const noexcept { return _terrain.grazed(); }
    [[nodiscard]] core::u32 regrown() const noexcept { return _terrain.regrown(); }
    [[nodiscard]] bool plantStanding(core::usize index) const { return _terrain.plantStanding(index); }
    [[nodiscard]] bool floraDirty() const noexcept { return _terrain.floraDirty(); }
    void clearFloraDirty() noexcept { _terrain.clearFloraDirty(); }
    [[nodiscard]] core::u32 trailCells() const
    {
        core::u32 cells = 0u;
        for (core::u32 z = 0u; z < _depth; ++z)
            for (core::u32 x = 0u; x < _width; ++x)
                if (_living.scent().field().value(kAntChannel, x, z).toFloat() > 0.25f)
                    ++cells;
        return cells;
    }
    [[nodiscard]] core::u32 agents() const noexcept { return _colony.agentCount(); }
    [[nodiscard]] core::u32 returns() const noexcept { return _colony.returns(); }
    [[nodiscard]] core::u32 ticks() const noexcept { return _ticks; }
    [[nodiscard]] const ecology::TrophicWeb &web() const noexcept { return _living.web(); }

private:
    /// The ants' channel, named for them by ai::ScentChannel itself.
    static constexpr core::u32 kAntChannel = static_cast<core::u32>(ai::ScentChannel::Pheromone);

    static constexpr core::u32 kAgents = 48u;
    static constexpr core::u32 kForageRange = 26u;
    /// Bodies the map will hold, whatever the model says.
    static constexpr core::u32 kMaxBodies = 120u;
    /// Fixed ticks between two demographic steps.
    static constexpr core::u32 kWebPeriod = 60u;
    /// Ticks a cropped plant takes to come back: twenty seconds at 60 Hz.
    static constexpr core::u32 kRegrowthTicks = 1200u;

    /// The ants' own field parameters. The herd's arrive in the LivingRecipe.
    ai::StigmergyParams _stigmergy{};
    ai::AntParams _ants{};

    // ── The living world, which is now the ENGINE's ──────────────────────────
    //
    // A registry, the engine's living layer, this map as a terrain, and the six
    // creature systems. What is left in this file is the ant colony (which the
    // engine's layer has no notion of) and the diagnostics — the two things that
    // are genuinely a viewer's.
    mutable ecs::Registry _registry;
    engine::LivingLayer _living;
    MapTerrain _terrain;
    /// The six stages of an animal's tick, ordered by engine::systems::CreatureStage.
    engine::systems::CreaturePipeline _creatures;

    mutable lpl::pmr::vector<ecology::Genome> _statScratch;
    procgen::Grid<core::u8> _blocked;
    /// The foragers, in ai/ where the mechanisms they use already lived.
    ai::AntColony _colony;
    core::u32 _width{0u};
    core::u32 _depth{0u};
    core::u32 _ticks{0u};
    core::u32 _births{0u};
    core::u32 _deaths{0u};
    /// Where each body was one second ago, for the measured ground speed.
    lpl::pmr::vector<ecs::EntityId> _speedIds;
    lpl::pmr::vector<math::Vec3<math::Fixed32>> _speedAt;
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
                      procgen::mapShadingName(options.shading), procgen::climateAxisName(options.climateAxis),
                      options.vegetation ? "on" : "off");
    else
        std::snprintf(line, sizeof(line), "view %s   shading %s   vegetation %s", viewName(options.view),
                      procgen::mapShadingName(options.shading), options.vegetation ? "on" : "off");
    put(line, 0.90f, 0.90f, 0.88f);

    std::snprintf(line, sizeof(line), "noise %s  warp %s  terraces %s", noiseName(options.noise),
                  options.warp ? "on" : "off", options.terraces ? "on" : "off");
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "erosion %s  rivers %s (%u cells)  wind %s", options.erosion ? "on" : "off",
                  options.rivers ? "on" : "off", world.stats.riverCells, windName(options.windDirection));
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "lakes %u cells   trunk drains %u of %u", world.drainage.raisedCells,
                  world.drainage.maxAccumulation, world.height.cellCount());
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "roads %s (%u cells)", options.roads ? "on" : "off", world.stats.roadCells);
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "regions %u (%s)   settlement %s", world.regions.regionCount,
                  metricName(options.metric), options.settlement ? "on" : "off");
    put(line, 0.72f, 0.74f, 0.76f);

    if (options.caveKind % 4u == 3u)
        // Entrances and reachability, not a floor count: a sealed system and an
        // open one have the same floor area, and only one of them is a cave.
        std::snprintf(line, sizeof(line), "underground layered  %u layers  %u entrances  %u/%u reachable",
                      world.stats.caveLayers, world.stats.caveEntrances, world.stats.caveReachable, world.stats.caveHollow);
    else
        std::snprintf(line, sizeof(line), "underground %s  %u floor  %s", caveName(options.caveKind),
                      world.stats.dungeonFloor, world.stats.dungeonConnected ? "connected" : "SPLIT");
    put(line, 0.72f, 0.74f, 0.76f);

    std::snprintf(line, sizeof(line), "town %s  %zu plots  roadside %u modules  liminal %u open",
                  options.grammarBuildings ? "grammar" : "prisms", world.plots.size(), world.stats.roadsideModules,
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
        const Rgb colour = procgen::biomeColour(static_cast<procgen::BiomeId>(i));
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
        auto collision =
            lpl::pmr::make_unique<engine::systems::HeightfieldCollisionSystem>(registry(), _terrain.height, 1.0f);
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

    /**
     * @brief One base height per cell: the lowest ground each building footprint covers.
     *
     * A building is a plan, and a plan has one floor level. Cells outside every
     * footprint keep their own ground, which is what the roadside decoration
     * wants — a fence does follow the slope.
     */
    [[nodiscard]] const std::vector<float> &buildPlotDatum()
    {
        _plotDatum.assign(static_cast<std::size_t>(_terrain.height.width()) * _terrain.height.depth(), 0.0f);
        for (core::u32 z = 0u; z < _terrain.height.depth(); ++z)
            for (core::u32 x = 0u; x < _terrain.height.width(); ++x)
                _plotDatum[_terrain.height.index(x, z)] = _terrain.height.at(x, z).toFloat();

        for (core::usize p = 0u; p < _terrain.plots.size(); ++p)
        {
            const procgen::BuildingPlot &plot = _terrain.plots[p];
            float lowest = 1.0e9f;
            for (core::u32 z = plot.z; z < plot.z + plot.depth; ++z)
                for (core::u32 x = plot.x; x < plot.x + plot.width; ++x)
                    if (_terrain.height.contains(static_cast<core::i32>(x), static_cast<core::i32>(z)))
                    {
                        const float h = _terrain.height.at(x, z).toFloat();
                        if (h < lowest)
                            lowest = h;
                    }
            if (lowest > 1.0e8f)
                continue;
            for (core::u32 z = plot.z; z < plot.z + plot.depth; ++z)
                for (core::u32 x = plot.x; x < plot.x + plot.width; ++x)
                    if (_terrain.height.contains(static_cast<core::i32>(x), static_cast<core::i32>(z)))
                        _plotDatum[_terrain.height.index(x, z)] = lowest;
        }
        return _plotDatum;
    }

    /// What the surface mesh should show, from the viewer's own options.
    [[nodiscard]] procgen::MapSurfaceStyle surfaceStyle() const noexcept
    {
        procgen::MapSurfaceStyle style;
        style.shading = _options.shading;
        style.climateAxis = _options.climateAxis;
        style.rivers = _options.rivers;
        style.settlement = _options.settlement;
        style.roads = _options.roads;
        return style;
    }

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
        _surfaceMesh = procgen::buildSurfaceMesh(_terrain, surfaceStyle());

        // The layered system replaces the flat plan entirely rather than being drawn
        // beside it: they are two answers to the same question, and showing both at
        // once would say nothing about either.
        _undergroundMesh = _options.caveKind % 4u == 3u ?
                               procgen::buildCaveSystemMesh(_terrain, kCaveDepth, kCaveLayerSpacing) :
                               procgen::buildDungeonMesh(_terrain, kCaveDepth);

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
            // The footprint datum, at last CONNECTED. buildPlotDatum was written,
            // documented as the fix for exactly this, and had no caller: the mesher
            // fell back to the per-column ground every time, so a plot on a slope
            // still sheared into a staircase and, along a ridge, into a long wall
            // standing free of the hillside. The comment claiming otherwise was
            // already there. Moving the mesher into a module is what surfaced it.
            const std::vector<float> &datum = buildPlotDatum();
            _townMesh = procgen::buildVoxelMesh(_terrain.townVolume, _terrain, -0.4f, palette, 4u, datum.data(),
                                                datum.size());
        }
        else
        {
            _townMesh = _options.settlement ? procgen::buildTownMesh(_terrain) : procgen::MapMesh{};
        }

        if (_options.roads && _options.grammarBuildings && !_terrain.roadsideVolume.empty())
        {
            const Rgb palette[4] = {
                {0.0f,  0.0f,  0.0f },
                {0.35f, 0.28f, 0.20f}, // fence
                {0.30f, 0.26f, 0.22f},
                {0.95f, 0.80f, 0.35f}
            }; // lamp
            _roadsideMesh = procgen::buildVoxelMesh(_terrain.roadsideVolume, _terrain, 0.0f, palette, 4u);
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
            _liminalMesh = procgen::buildLiminalMesh(_terrain.liminal);
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
                    int(_options.erosion), int(_options.rivers), _terrain.stats.riverCells, windName(_options.windDirection),
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

        math::Random random = math::deriveStream(_options.seed, 0xB0DEu);
        const core::f32 halfWidth = static_cast<core::f32>(_terrain.height.width()) * 0.5f;
        const core::f32 halfDepth = static_cast<core::f32>(_terrain.height.depth()) * 0.5f;
        const core::f32 ceiling = _terrain.highest.toFloat() + 3.0f;

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
                    procgen::appendPlant(mesh, _terrain, position[i].x.toFloat(), position[i].y.toFloat(),
                                         position[i].z.toFloat(), half);
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
        const float y = _terrain.highest.toFloat() + 2.0f;

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
            _surfaceMesh = procgen::buildSurfaceMesh(_terrain, surfaceStyle());
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
    std::vector<float> _plotDatum;
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
    engine::systems::HeightfieldCollisionSystem *_collision{nullptr};
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
