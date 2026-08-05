/**
 * @file main.cpp
 * @brief `lpl-worldforge` — a standalone OpenGL world editor prototype.
 *
 * A deliberately throwaway, self-contained editor: GLFW window, a legacy
 * immediate-mode OpenGL viewport, and Dear ImGui panels — everything lives in
 * this one file, wired to none of the engine's Vulkan render module. It exists
 * so world editing works *today* without finishing the Vulkan pipeline. The only
 * things reused are the renderer-agnostic pieces: @c editor::EditorSession drives
 * every panel (inspector / hierarchy / procgen / sim) with zero per-component UI
 * code, the procgen commands build worlds, and @c physics::CpuPhysicsBackend
 * steps the authoritative Fixed32 state. The viewport draws one cube per entity
 * (Position + AABB read straight from the ECS) under an orbital camera.
 *
 * When the engine's Vulkan renderer is finished, the real viewport moves there;
 * this stays as the quick tool.
 *
 * One engine dependency did arrive, deliberately: @c engine::DemonHost, for the
 * Caine panel. The loop that generates, looks and corrects lives there, and a
 * second copy of it inside a panel would be the duplication this project keeps
 * paying for — a panel is a view of a loop, not a place to keep one. It brings no
 * Vulkan with it: the viewport below is still legacy OpenGL.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl2.h>
#include <imgui.h>

#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/EditorSession.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/agent/Observation.hpp>
#include <lpl/agent/Planner.hpp>
#include <lpl/agent/Transcript.hpp>
#include <lpl/engine/DemonHost.hpp>
#include <lpl/engine/InferenceBudget.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/ecology/Herd.hpp>
#include <lpl/ecology/LivingRecipe.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/engine/GridTerrain.hpp>
#include <lpl/engine/LivingLayer.hpp>
#include <lpl/engine/systems/CreatureSystems.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/procgen/Liminal.hpp>
#include <lpl/procgen/MapMesh.hpp>
#include <lpl/procgen/MapShading.hpp>
#include <lpl/procgen/QualityGate.hpp>
#include <lpl/procgen/Streaming.hpp>
#include <lpl/procgen/WorldAtlas.hpp>

namespace {

using FVec3 = lpl::math::Vec3<lpl::math::Fixed32>;

// Biome names in lpl::procgen::BiomeId order, which is also the spelling
// lpl::procgen::biomeIdByName accepts — the combo writes the document's own
// vocabulary rather than an enum index a reordering would silently reinterpret.
constexpr const char *kBiomeNames[] = {"ocean",  "beach",   "snow",      "tundra", "taiga",      "rock",
                                       "desert", "savanna", "grassland", "forest", "rainforest", "marsh"};

/**
 * @struct AtlasView
 * @brief The generated world as an instrument, beside the world as cubes.
 *
 * The editor's viewport draws one cube per entity, which says where things are and
 * nothing about the world they are in: no relief, no biomes, no rivers, no
 * underground, nothing living. The map viewer showed all of that and this could not,
 * for a concrete reason — `generate_world` runs @c procgen::bakeWorld, which returns
 * signatures and counts and throws every grid away.
 *
 * So this holds a @c procgen::WorldAtlas, re-derived from the SAME procedural block
 * the Generate button issued. Generation is deterministic, so a recipe run twice is
 * the same world down to the bit: a diagnostic re-derivation, not a second source of
 * truth. Nothing here is authored — none of it is saved, and the creatures below live
 * in their own registry precisely so `save_scene` does not emit a preview.
 *
 * Every layer is drawn with the shared definitions rather than a second set:
 * @c procgen::MapShading colours it, @c engine::GridTerrain answers the ground, and
 * @c engine::systems::CreaturePipeline runs the six systems in the one order they are
 * declared in. Writing a second copy of any of those is what this whole slice exists
 * to refuse.
 */
struct AtlasView {
    lpl::procgen::WorldAtlas atlas;
    lpl::procgen::LiminalSpace liminal;
    bool has = false;
    bool hasLiminal = false;

    // What the map shows.
    int shading = 0; ///< Index into lpl::procgen::MapShading.
    int climateAxis = 0;
    int scale = 8; ///< Pixels per cell.
    bool paintRivers = true, paintSettlement = true, paintRoads = true;
    bool overlayBlocked = false, overlayCaves = false, overlayPlots = false;
    bool overlayChunks = false, showLiminal = false;
    int chunkSize = 16, chunkRadius = 3;

    // The living preview: its own registry, because these bodies are not part of the
    // document being edited and must never be saved with it.
    lpl::ecs::Registry registry;
    lpl::engine::LivingLayer living;
    lpl::engine::GridTerrain terrain;
    lpl::engine::systems::CreaturePipeline creatures;
    lpl::procgen::Grid<lpl::core::u8> obstacles;
    bool livingReady = false;
    bool livingRunning = false;
    lpl::core::u32 ticks = 0u;
};

// ── Editor UI + world state (owned by main, referenced by the panel helpers) ──
struct EditorUi {
    lpl::editor::EditorSession session;
    lpl::physics::CpuPhysicsBackend backend{session.registry()};

    // Sim control.
    bool playing = false;
    bool stepOnce = false;
    const float dt = 1.0f / 60.0f;

    // Orbital camera.
    float camYaw = 35.0f, camPitch = 25.0f, camDist = 30.0f;
    float camTarget[3] = {0.0f, 0.0f, 0.0f};

    // Procgen panel params — sections of one recipe (human units).
    int hfSeed = 1337, hfCols = 24, hfRows = 24, hfOctaves = 4;
    float hfAmplitude = 12.0f, hfNoiseScale = 0.15f, hfSpacing = 0.5f;
    bool doErode = true, doRivers = true, doClimate = true, doCaves = true, doTown = true;
    int thermalIterations = 8, hydraulicIterations = 12;
    int scBiome = 8; // grassland, in kBiomeNames order
    float scDensity = 0.06f, scHalfExtent = 0.2f;
    bool scCollidable = false;
    float caveFill = 0.45f;
    int caveSteps = 5, gateMinPath = 4;

    std::string lastReport = "(no command yet)";
    std::string lastGate = "(not checked)";

    // Scene file I/O.
    char scenePath[256] = "world.lplscene";
    std::string sceneStatus = "(no file yet)";

    // ── Caine: the intelligence attached to this session ──────────────────────
    //
    // The planner is deterministic on purpose. Local inference is a separate
    // process behind a file/socket boundary, and until it is wired the panel is
    // still worth having: agent::CorrectionPlanner follows the critics' own
    // suggestions, which is what a competent operator would do, so the loop can be
    // watched working without a model in the picture.
    lpl::agent::CorrectionPlanner planner;
    lpl::engine::DemonHost demon{session.registry(), session.journal(), planner};
    char intent[256] = "a world with rivers and a village";
    // Furnishing panel: what the level's own measurements say to place.
    int furnishEncounters = 4, furnishRewards = 3, furnishSpacing = 3;
    std::vector<lpl::procgen::Placement> furnished;
    std::string furnishStatus = "(not furnished yet)";
    int demonTurns = 8;
    bool hasThought = false;
    lpl::engine::ThinkResult lastThink{};

    /// The world as an instrument. See AtlasView for why it is a re-derivation.
    AtlasView view;
};

/**
 * @brief The `procedural` block the procgen panel's controls describe.
 *
 * Built once and used twice: handed to the session as a `generate_world` command (so
 * the act is journalled and undoable) and parsed into a recipe for the atlas. Two
 * strings would eventually differ by one field and the picture would stop being of
 * the world the button built.
 */
std::string proceduralJson(const EditorUi &ui)
{
    char cmd[768];
    std::snprintf(cmd, sizeof(cmd),
                  R"({"cmd":"generate_world","seed":%d,"width":%d,"depth":%d,"cellSize":%f,)"
                  R"("terrain":{"seed":%d,"frequency":%f,"amplitude":%f,"octaves":%d},)"
                  R"("erosion":{"enabled":%s,"thermalIterations":%d,"hydraulicIterations":%d},)"
                  R"("rivers":{"enabled":%s},"climate":{"enabled":%s},)"
                  R"("caves":{"enabled":%s,"width":%d,"depth":%d,"fillProbability":%f,"steps":%d},)"
                  R"("settlement":{"enabled":%s},)"
                  R"("gate":{"enabled":%s,"minPathLength":%d},)"
                  R"("scatter":[{"biome":"%s","density":%f,"halfExtent":%f,"collidable":%s}]})",
                  ui.hfSeed, ui.hfCols, ui.hfRows, ui.hfSpacing, ui.hfSeed, ui.hfNoiseScale, ui.hfAmplitude,
                  ui.hfOctaves, ui.doErode ? "true" : "false", ui.thermalIterations, ui.hydraulicIterations,
                  ui.doRivers ? "true" : "false", ui.doClimate ? "true" : "false", ui.doCaves ? "true" : "false",
                  ui.hfCols, ui.hfRows, ui.caveFill, ui.caveSteps, ui.doTown ? "true" : "false",
                  ui.doCaves ? "true" : "false", ui.gateMinPath, kBiomeNames[ui.scBiome], ui.scDensity,
                  ui.scHalfExtent, ui.scCollidable ? "true" : "false");
    return std::string{cmd};
}

/**
 * @brief Re-derives the atlas from the block the Generate button just issued.
 *
 * Registry is null on purpose: the entities already exist in the session's registry,
 * created by the journalled command. Materialising a second copy here would double
 * every prop in the hierarchy.
 */
void refreshAtlas(EditorUi &ui)
{
    lpl::procgen::WorldRecipe recipe{};
    if (!lpl::editor::parseProceduralBlock(proceduralJson(ui), recipe))
    {
        ui.view.has = false;
        return;
    }
    ui.view.atlas = lpl::procgen::buildAtlas(recipe, nullptr, nullptr);
    ui.view.has = !ui.view.atlas.height.empty();
    ui.view.livingReady = false;
    ui.view.livingRunning = false;
    ui.view.ticks = 0u;
}

/**
 * @brief Stands an ecology up on the atlas: obstacles, plants, herd, six systems.
 *
 * The obstacle mask is the atlas's own — ONE notion of "blocked", extended with the
 * footprints the settlement raised. A second slope threshold here would be the third
 * notion, which is how an animal ends up standing in a lake the scent flows around.
 */
void startLiving(EditorUi &ui)
{
    AtlasView &view = ui.view;
    if (!view.has)
        return;

    const lpl::core::u32 width = view.atlas.height.width();
    const lpl::core::u32 depth = view.atlas.height.depth();
    const lpl::core::u32 seed = static_cast<lpl::core::u32>(ui.hfSeed);

    view.obstacles = view.atlas.blocked;
    if (!view.atlas.settlement.empty())
        for (lpl::core::u32 z = 0u; z < depth; ++z)
            for (lpl::core::u32 x = 0u; x < width; ++x)
                if (view.atlas.settlement.at(x, z) == lpl::procgen::SettlementCell::Plot)
                    view.obstacles.at(x, z) = 1u;

    lpl::engine::LivingLayerParams params;
    params.maxBodies = 96u;
    params.speciesCount = 2u;
    params.scentSpan = width < depth ? width : depth;
    params.scentLayers = 6u;
    params.webPeriod = 60u;

    lpl::ecology::LivingRecipe recipe = lpl::ecology::parityLivingRecipe();
    recipe.regrowthTicks = 600u;

    view.terrain.reset(&view.obstacles, width, depth, recipe.regrowthTicks);
    // The plants come from procgen's own thinning rule, which exists for exactly this
    // and keeps a reload growing the same forest. Cell indices become world cells: the
    // grid is centred on the origin, which is the one conversion the systems cannot know.
    const lpl::core::i32 halfW = static_cast<lpl::core::i32>(width / 2u);
    const lpl::core::i32 halfD = static_cast<lpl::core::i32>(depth / 2u);
    (void) lpl::procgen::scatterVegetation(
        view.atlas, seed ^ 0x5EEDu, 6u,
        [](lpl::procgen::BiomeId biome) {
            return biome == lpl::procgen::BiomeId::Forest || biome == lpl::procgen::BiomeId::Taiga ||
                   biome == lpl::procgen::BiomeId::Rainforest || biome == lpl::procgen::BiomeId::Grassland ||
                   biome == lpl::procgen::BiomeId::Marsh;
        },
        [&](lpl::core::i32 x, lpl::core::i32 z) { view.terrain.addPlant(x - halfW, z - halfD); });

    view.living.configure(params, recipe, seed);
    view.living.bind(view.registry);
    view.living.openScent(params.scentSpan, params.scentLayers);
    view.living.scent().centreOn(0, 0);
    view.living.scent().field().setObstacles(view.obstacles);
    view.living.seedWeb(view.terrain.standingPlants());

    // The six systems, in the one order engine::systems::CreatureStage declares.
    view.creatures.build(view.registry, view.living, view.terrain);

    view.living.reconcile(0u, [&](lpl::math::Random &random, lpl::core::u32 /*attempt*/, lpl::math::Fixed32 &outX,
                                  lpl::math::Fixed32 &outZ) {
        // Somewhere an animal can actually stand: dropping a herd into the sea and
        // letting the flocking rules sort it out gives a confident-looking shoal of deer.
        if (width == 0u)
            return false;
        const lpl::core::u32 x = random.below(width);
        const lpl::core::u32 z = random.below(depth);
        if (view.obstacles.at(x, z) != 0u)
            return false;
        outX = lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(x) - halfW);
        outZ = lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(z) - halfD);
        return true;
    });

    view.livingReady = true;
    view.ticks = 0u;
}

/// One tick of the preview ecology: the six systems, then the field, then the web.
void stepLiving(EditorUi &ui)
{
    AtlasView &view = ui.view;
    if (!view.livingReady)
        return;
    ++view.ticks;
    view.creatures.step(1.0f / 60.0f);
    view.living.stepScent();
    view.living.setProducerPopulation(view.terrain.tickVegetation());
    if (view.ticks % view.living.params().webPeriod == 0u)
    {
        view.living.stepWeb();
        view.living.reconcile(view.ticks, [&](lpl::math::Random &random, lpl::core::u32 /*attempt*/,
                                             lpl::math::Fixed32 &outX, lpl::math::Fixed32 &outZ) {
            const lpl::core::u32 width = view.obstacles.width();
            const lpl::core::u32 depth = view.obstacles.depth();
            if (width == 0u)
                return false;
            const lpl::core::u32 x = random.below(width);
            const lpl::core::u32 z = random.below(depth);
            if (view.obstacles.at(x, z) != 0u)
                return false;
            outX = lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(x) - static_cast<lpl::core::i32>(width / 2u));
            outZ = lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(z) - static_cast<lpl::core::i32>(depth / 2u));
            return true;
        });
    }
}

// ── ImGui panels — all backend-agnostic, all driven by EditorSession ──────────

// Enumerates the selected entity's components/fields and edits them generically.
void drawInspector(EditorUi &ui)
{
    ImGui::Begin("Inspector");
    if (!ui.session.hasSelection())
    {
        ImGui::TextUnformatted("No entity selected.");
        ImGui::End();
        return;
    }
    const lpl::core::u32 e = ui.session.selection();
    ImGui::Text("Entity #%u", e);
    ImGui::Separator();

    const lpl::editor::EntityLocation loc = ui.session.locate(e);
    for (const lpl::ecs::ComponentSchema &schema : lpl::ecs::allSchemas())
    {
        if (!loc.valid() || !loc.chunk->archetype().has(schema.id))
            continue;
        const std::string header{schema.name};
        if (!ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
            continue;
        for (const lpl::ecs::FieldDesc &f : schema.fields)
        {
            const lpl::core::u32 lanes = lpl::editor::EditorSession::laneCount(f.type);
            float vals[4] = {0, 0, 0, 0};
            for (lpl::core::u32 l = 0; l < lanes; ++l)
            {
                double v = 0.0;
                (void) ui.session.getField(e, schema.id, f.name, l, v);
                vals[l] = static_cast<float>(v);
            }
            const std::string label{f.name};
            const std::string id = label + "##" + header;
            bool changed = false;
            if (lanes == 1)
                changed = ImGui::DragFloat(id.c_str(), &vals[0], 0.05f);
            else if (lanes == 3)
                changed = ImGui::DragFloat3(id.c_str(), vals, 0.05f);
            else
                changed = ImGui::DragFloat4(id.c_str(), vals, 0.05f);
            if (changed)
                for (lpl::core::u32 l = 0; l < lanes; ++l)
                    (void) ui.session.setField(e, schema.id, f.name, l, static_cast<double>(vals[l]));
        }
    }
    ImGui::End();
}

// Entity list; click to select.
void drawHierarchy(EditorUi &ui)
{
    ImGui::Begin("Hierarchy");
    const lpl::core::u32 count = ui.session.entityCount();
    ImGui::Text("%u entities", count);
    ImGui::Separator();
    ImGui::BeginChild("entity-list");
    const lpl::core::u32 shown = count < 4096u ? count : 4096u;
    for (lpl::core::u32 i = 0; i < shown; ++i)
    {
        char label[32];
        std::snprintf(label, sizeof(label), "Entity #%u", i);
        if (ImGui::Selectable(label, ui.session.hasSelection() && ui.session.selection() == i))
            ui.session.select(i);
    }
    if (shown < count)
        ImGui::Text("... %u more", count - shown);
    ImGui::EndChild();
    ImGui::End();
}

// Procgen commands + playability gate, all through EditorSession::command.
void drawProcgen(EditorUi &ui)
{
    ImGui::Begin("Procedural Generation");

    // One button, one world. The panels below are sections of a single recipe
    // rather than independent commands, because the passes are not independent:
    // moisture needs the drainage the rivers left, biomes need the moisture. What
    // a panel switches off is a pass, not a separate generator.
    if (ImGui::CollapsingHeader("Terrain", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputInt("seed##hf", &ui.hfSeed);
        ImGui::SliderInt("width##hf", &ui.hfCols, 1, 128);
        ImGui::SliderInt("depth##hf", &ui.hfRows, 1, 128);
        ImGui::SliderFloat("amplitude##hf", &ui.hfAmplitude, 0.0f, 32.0f);
        ImGui::SliderFloat("frequency##hf", &ui.hfNoiseScale, 0.01f, 1.0f);
        ImGui::SliderInt("octaves##hf", &ui.hfOctaves, 1, 8);
        ImGui::SliderFloat("cell size##hf", &ui.hfSpacing, 0.1f, 2.0f);
    }

    if (ImGui::CollapsingHeader("Morphogenesis", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("erosion", &ui.doErode);
        ImGui::SameLine();
        ImGui::Checkbox("rivers", &ui.doRivers);
        ImGui::SameLine();
        ImGui::Checkbox("climate", &ui.doClimate);
        ImGui::SliderInt("thermal iterations", &ui.thermalIterations, 0, 64);
        ImGui::SliderInt("hydraulic iterations", &ui.hydraulicIterations, 0, 64);
    }

    if (ImGui::CollapsingHeader("Scatter"))
    {
        ImGui::Combo("biome##sc", &ui.scBiome, kBiomeNames, IM_ARRAYSIZE(kBiomeNames));
        ImGui::SliderFloat("density##sc", &ui.scDensity, 0.0f, 0.5f);
        ImGui::SliderFloat("half extent##sc", &ui.scHalfExtent, 0.05f, 2.0f);
        ImGui::Checkbox("collidable##sc", &ui.scCollidable);
    }

    if (ImGui::CollapsingHeader("Underground + gate"))
    {
        ImGui::Checkbox("caves", &ui.doCaves);
        ImGui::SameLine();
        ImGui::Checkbox("settlement", &ui.doTown);
        ImGui::SliderFloat("rock fill", &ui.caveFill, 0.3f, 0.6f);
        ImGui::SliderInt("automaton steps", &ui.caveSteps, 0, 12);
        ImGui::SliderInt("min path length", &ui.gateMinPath, 0, 64);
        ImGui::TextWrapped("%s", ui.lastGate.c_str());
    }

    if (ImGui::Button("Generate world"))
    {
        // ONE procedural block, two consumers: the journalled command that builds the
        // world, and the atlas that shows it. See proceduralJson.
        const std::string cmd = proceduralJson(ui);
        const auto r = ui.session.command(cmd);
        ui.lastReport = r.has_value() ? r.value() : "generate failed";
        ui.lastGate = ui.lastReport;
        refreshAtlas(ui);
    }

    ImGui::Separator();
    ImGui::TextWrapped("Last: %s", ui.lastReport.c_str());
    ImGui::End();
}

// Save / load the world as a `.lplscene` document on disk.
void drawSceneIO(EditorUi &ui)
{
    ImGui::Begin("Scene");
    ImGui::InputText("path", ui.scenePath, sizeof(ui.scenePath));
    if (ImGui::Button("Save"))
    {
        const std::string doc = ui.session.save();
        std::ofstream out(ui.scenePath, std::ios::binary | std::ios::trunc);
        if (out.is_open())
        {
            out.write(doc.data(), static_cast<std::streamsize>(doc.size()));
            ui.sceneStatus = "saved " + std::to_string(doc.size()) + " bytes";
        }
        else
        {
            ui.sceneStatus = "save failed (cannot open file)";
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load (replace)"))
    {
        std::ifstream in(ui.scenePath, std::ios::binary);
        if (in.is_open())
        {
            std::stringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();
            ui.session.clear();
            const auto n = ui.session.load(text);
            ui.sceneStatus = n.has_value() ? "loaded " + std::to_string(n.value()) + " entities" : "load: parse error";
        }
        else
        {
            ui.sceneStatus = "load failed (cannot open file)";
        }
    }
    ImGui::TextWrapped("%s", ui.sceneStatus.c_str());
    ImGui::End();
}

// Play / pause / step / reset over the authoritative Fixed32 physics + camera.
void drawSim(EditorUi &ui)
{
    ImGui::Begin("Simulation");
    if (ImGui::Button(ui.playing ? "Pause" : "Play"))
        ui.playing = !ui.playing;
    ImGui::SameLine();
    if (ImGui::Button("Step"))
        ui.stepOnce = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear world"))
    {
        ui.session.clear();
        ui.playing = false;
    }
    ImGui::Text("state: %s", ui.playing ? "running" : "paused");
    ImGui::Separator();
    ImGui::TextUnformatted("Camera");
    ImGui::SliderFloat("yaw", &ui.camYaw, -180.0f, 180.0f);
    ImGui::SliderFloat("pitch", &ui.camPitch, -89.0f, 89.0f);
    ImGui::SliderFloat("distance", &ui.camDist, 2.0f, 120.0f);
    ImGui::DragFloat3("target", ui.camTarget, 0.1f);
    ImGui::Separator();
    ImGui::TextWrapped("Viewport: drag = orbit, scroll = zoom, click = select. "
                       "Selected: arrows move XZ, Q/E move up/down.");
    ImGui::End();
}

// ── OpenGL immediate-mode viewport ────────────────────────────────────────────

// A unit cube with per-face normals (colour comes from the caller via
// GL_COLOR_MATERIAL, shading from GL_LIGHT0).
/**
 * @brief The Caine panel: what was asked, what was done, what is still wrong.
 *
 * The only place the loop can be WATCHED. Everything it shows is already
 * computed — the critics' findings, the ReAct transcript, the journal indices —
 * and none of it was visible anywhere: a fold proves a loop ran, it does not show
 * what it decided.
 */
void drawCaine(EditorUi &ui)
{
    ImGui::Begin("Caine");

    ImGui::TextWrapped("The intelligence directs; the deterministic engine executes.");
    ImGui::Separator();

    ImGui::InputText("intent", ui.intent, sizeof(ui.intent));
    ImGui::SliderInt("turn budget", &ui.demonTurns, 1, 32);
    // Turns, never milliseconds: a wall clock on a replayable path is a desync
    // waiting for a slow frame.
    if (ImGui::Button("Think"))
    {
        ui.demon.consider(lpl::agent::Intent{ui.intent, 0u});
        ui.lastThink = ui.demon.think(lpl::engine::InferenceBudget::ofTurns(static_cast<lpl::core::u32>(ui.demonTurns)));
        ui.hasThought = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Undo last act"))
        (void) ui.session.undo();

    if (ui.hasThought)
    {
        const char *outcome = "concluded";
        switch (ui.lastThink.outcome)
        {
        case lpl::engine::ThinkOutcome::Concluded: outcome = "concluded"; break;
        case lpl::engine::ThinkOutcome::BudgetExhausted: outcome = "budget exhausted"; break;
        case lpl::engine::ThinkOutcome::Stuck: outcome = "stuck (same call twice)"; break;
        case lpl::engine::ThinkOutcome::NoLegalMove: outcome = "no legal move"; break;
        }
        ImGui::Text("%s in %u turns: %u acts, %u refusals", outcome, ui.lastThink.turns, ui.lastThink.acts,
                    ui.lastThink.refusals);
        ImGui::Text("defects %u -> %u  %s", ui.lastThink.defectsBefore, ui.lastThink.defectsAfter,
                    ui.lastThink.sound() ? "(sound)" : "");
    }

    // ── What the critics say NOW, not what they said when it stopped ──────────
    const lpl::agent::Observations findings = ui.demon.observe();
    if (ImGui::CollapsingHeader("Defects", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (findings.findings.empty())
            ImGui::TextUnformatted("Nothing to report.");
        for (const lpl::agent::Finding &f : findings.findings)
        {
            const bool defect = f.severity == lpl::agent::Severity::Defect;
            ImGui::TextColored(defect ? ImVec4{0.95f, 0.45f, 0.25f, 1.0f} : ImVec4{0.75f, 0.75f, 0.75f, 1.0f},
                               "[%s] %s", f.code.c_str(), f.message.c_str());
            if (!f.suggestedTool.empty())
            {
                ImGui::SameLine();
                ImGui::TextDisabled("-> %s", f.suggestedTool.c_str());
            }
        }
        if (findings.truncated)
            ImGui::TextDisabled("... %u more", findings.total - static_cast<lpl::core::u32>(findings.findings.size()));
    }

    if (ImGui::CollapsingHeader("Trace"))
    {
        ImGui::BeginChild("react-trace", ImVec2{0.0f, 160.0f}, true);
        for (const lpl::agent::Turn &turn : ui.demon.transcript().turns())
        {
            ImGui::Text("%u. %s %s", turn.index, turn.ok ? "ok " : "REF", turn.tool.c_str());
            if (!turn.thought.empty())
                ImGui::TextDisabled("    %s", turn.thought.c_str());
            if (!turn.observation.empty())
                ImGui::TextWrapped("    %s", turn.observation.c_str());
            // The journal index, because that is what makes an act undoable rather
            // than merely logged.
            if (turn.journalEntry != lpl::agent::kNotJournalled)
                ImGui::TextDisabled("    journal #%u", turn.journalEntry);
        }
        ImGui::EndChild();
    }

    ImGui::End();
}

/**
 * @brief What a level's own measurements say to put where.
 *
 * procgen::analyseHotPath computed both answers and nothing consumed either. The
 * spine is where a player certainly walks, so an encounter there is met rather
 * than missed; the deepest dead ends are where something is worth hiding, for the
 * same reason they were architectural excrescences.
 */
void drawFurnishing(EditorUi &ui)
{
    ImGui::Begin("Furnishing");
    ImGui::SliderInt("encounters", &ui.furnishEncounters, 1, 12);
    ImGui::SliderInt("rewards", &ui.furnishRewards, 1, 12);
    ImGui::SliderInt("spacing", &ui.furnishSpacing, 1, 12);

    if (ImGui::Button("Furnish the cave"))
    {
        // The cave the procgen panel's parameters describe, generated here so the
        // panel reads the SAME level the Generate button built rather than a second
        // one that happens to share a seed.
        lpl::procgen::CaveParams cave;
        cave.width = static_cast<lpl::core::u32>(ui.hfCols);
        cave.depth = static_cast<lpl::core::u32>(ui.hfRows);
        cave.seed = static_cast<lpl::core::u32>(ui.hfSeed);
        cave.fillProbability = ui.caveFill;
        cave.steps = static_cast<lpl::core::u32>(ui.caveSteps);
        const lpl::procgen::DungeonMap level = lpl::procgen::generateCellularCave(cave);

        lpl::core::u32 sx = 0u, sz = 0u, gx = 0u, gz = 0u;
        ui.furnished.clear();
        if (!lpl::procgen::findFarthestPair(level, sx, sz, gx, gz))
        {
            ui.furnishStatus = "the cave has no two ends to run between";
        }
        else
        {
            const lpl::procgen::HotPathAnalysis hot =
                lpl::procgen::analyseHotPath(level, sx, sz, gx, gz, /*detourLimit=*/6u);
            lpl::procgen::PlacementParams wanted;
            wanted.encounters = static_cast<lpl::core::u32>(ui.furnishEncounters);
            wanted.rewards = static_cast<lpl::core::u32>(ui.furnishRewards);
            wanted.minSpacing = static_cast<lpl::core::u32>(ui.furnishSpacing);
            lpl::procgen::Placement spots[32]{};
            const lpl::core::u32 placed =
                lpl::procgen::placeAlongHotPath(level, hot, sx, sz, wanted, spots, 32u);
            for (lpl::core::u32 i = 0u; i < placed; ++i)
                ui.furnished.push_back(spots[i]);

            char line[160];
            std::snprintf(line, sizeof(line), "spine %u cells, deepest detour %u, %u spots placed", hot.pathCells,
                          hot.deepestDetour, placed);
            ui.furnishStatus = line;
        }
    }

    ImGui::TextWrapped("%s", ui.furnishStatus.c_str());
    ImGui::Separator();
    // Fewer spots than asked is a fact about the level, not an error: a cave with
    // one dead end has one hiding place.
    for (const lpl::procgen::Placement &spot : ui.furnished)
        ImGui::Text("%-9s (%3u,%3u)  detour %2u  progress %3u",
                    spot.role == lpl::procgen::PlacementRole::Encounter ? "encounter" : "reward", spot.x, spot.z,
                    spot.detour, spot.progress);
    ImGui::End();
}

/**
 * @brief The Atlas panel: the five layers the editor could not show.
 *
 * Drawn as filled rectangles on an ImGui draw list rather than as a texture: there is
 * no texture upload path in this throwaway viewport, a map is at most 128x128 cells,
 * and a rectangle per cell is both cheap and exactly what a flat map is.
 *
 * Layer order is not arbitrary. The base shading goes down first, then what is a
 * property of the ground (rivers, streets, highways — painted INTO the surface by
 * procgen::buildSurfaceMesh's own style for the same reason), then what sits on it
 * (blocked cells, cave floor, plots), then what moves (creatures), then what is only a
 * plan (chunk boundaries). Reversing any pair hides the thing the layer exists to show.
 */
void drawAtlas(EditorUi &ui)
{
    ImGui::Begin("Atlas");
    AtlasView &view = ui.view;

    if (ImGui::Button("Re-derive from the recipe"))
        refreshAtlas(ui);
    ImGui::SameLine();
    ImGui::TextDisabled(view.has ? "%u x %u cells" : "(generate a world first)", view.atlas.width, view.atlas.depth);

    if (!view.has)
    {
        ImGui::TextWrapped("The Generate button builds the world; this shows it. Both read the same "
                           "procedural block, and generation is deterministic, so the picture is of "
                           "the world that was built rather than of one that resembles it.");
        ImGui::End();
        return;
    }

    // ── What colours a cell ───────────────────────────────────────────────────
    const char *modes[static_cast<int>(lpl::procgen::MapShading::Count)];
    for (int i = 0; i < static_cast<int>(lpl::procgen::MapShading::Count); ++i)
        modes[i] = lpl::procgen::mapShadingName(static_cast<lpl::procgen::MapShading>(i));
    ImGui::Combo("shading", &view.shading, modes, static_cast<int>(lpl::procgen::MapShading::Count));
    if (view.shading == static_cast<int>(lpl::procgen::MapShading::Climate))
    {
        ImGui::SliderInt("axis", &view.climateAxis, 0, static_cast<int>(lpl::procgen::kClimateAxisCount) - 1);
        ImGui::SameLine();
        ImGui::TextDisabled("%s", lpl::procgen::climateAxisName(static_cast<lpl::core::u32>(view.climateAxis)));
    }
    ImGui::SliderInt("zoom", &view.scale, 2, 16);

    if (ImGui::CollapsingHeader("Layers", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("rivers", &view.paintRivers);
        ImGui::SameLine();
        ImGui::Checkbox("streets", &view.paintSettlement);
        ImGui::SameLine();
        ImGui::Checkbox("highways", &view.paintRoads);
        ImGui::Checkbox("blocked", &view.overlayBlocked);
        ImGui::SameLine();
        ImGui::Checkbox("caves", &view.overlayCaves);
        ImGui::SameLine();
        ImGui::Checkbox("plots", &view.overlayPlots);
        ImGui::Checkbox("streaming plan", &view.overlayChunks);
        if (view.overlayChunks)
        {
            ImGui::SliderInt("chunk size", &view.chunkSize, 4, 64);
            ImGui::SliderInt("radius", &view.chunkRadius, 1, 6);
        }
    }

    if (ImGui::CollapsingHeader("Liminal sector"))
    {
        // A different generator on its own seed, not a view of this world: it is here
        // because a liminal space is the one thing in the module whose whole point is
        // how it feels to stand in, and a fold cannot report that.
        if (ImGui::Button("Generate liminal"))
        {
            lpl::procgen::LiminalParams params;
            params.width = view.atlas.width;
            params.depth = view.atlas.depth;
            params.seed = static_cast<lpl::core::u32>(ui.hfSeed) ^ 0x11B1A0u;
            view.liminal = lpl::procgen::generateLiminal(params);
            view.hasLiminal = !view.liminal.map.empty();
            view.showLiminal = view.hasLiminal;
        }
        if (view.hasLiminal)
        {
            ImGui::SameLine();
            ImGui::Checkbox("show instead", &view.showLiminal);
            ImGui::Text("%u open cells, %s, %u pillars", view.liminal.openCells,
                        view.liminal.connected ? "connected" : "SPLIT", view.liminal.pillars);
        }
    }

    if (ImGui::CollapsingHeader("Living layer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button(view.livingReady ? "Reseed ecology" : "Stand an ecology up"))
            startLiving(ui);
        if (view.livingReady)
        {
            ImGui::SameLine();
            ImGui::Checkbox("running", &view.livingRunning);
            ImGui::SameLine();
            if (ImGui::Button("Tick once"))
                stepLiving(ui);
            ImGui::Text("tick %u  %u bodies  %u/%u plants standing  %u grazed", view.ticks,
                        view.living.herd().size(), view.terrain.standingPlants(), view.terrain.plantCount(),
                        view.terrain.grazed());
            if (const lpl::engine::systems::LocomotionSystem *walk = view.creatures.locomotion(); walk != nullptr)
                ImGui::TextDisabled("%u cornered, %u avoided, %u strays", walk->cornered(), walk->avoided(),
                                    walk->strays());
        }
    }

    ImGui::Separator();

    // ── The canvas ────────────────────────────────────────────────────────────
    const float cell = static_cast<float>(view.scale);
    const bool liminalView = view.showLiminal && view.hasLiminal;
    const lpl::core::u32 width = liminalView ? view.liminal.map.width() : view.atlas.width;
    const lpl::core::u32 depth = liminalView ? view.liminal.map.depth() : view.atlas.depth;

    ImDrawList *draw = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const auto rect = [&](lpl::core::u32 x, lpl::core::u32 z, const lpl::procgen::Rgb &colour, float alpha) {
        const ImVec2 a{origin.x + static_cast<float>(x) * cell, origin.y + static_cast<float>(z) * cell};
        const ImVec2 b{a.x + cell, a.y + cell};
        draw->AddRectFilled(a, b, ImGui::ColorConvertFloat4ToU32(ImVec4{colour.r, colour.g, colour.b, alpha}));
    };

    for (lpl::core::u32 z = 0u; z < depth; ++z)
        for (lpl::core::u32 x = 0u; x < width; ++x)
        {
            if (liminalView)
            {
                // The zone palette, from the same place the 3D mesher takes it.
                const lpl::procgen::Rgb colour =
                    lpl::procgen::isWalkable(view.liminal.map.at(x, z))
                        ? lpl::procgen::liminalZoneColour(view.liminal.zones.at(x, z))
                        : lpl::procgen::Rgb{0.10f, 0.10f, 0.12f};
                rect(x, z, colour, 1.0f);
                continue;
            }

            lpl::procgen::Rgb colour = lpl::procgen::surfaceColour(
                view.atlas, static_cast<lpl::procgen::MapShading>(view.shading),
                static_cast<lpl::core::u32>(view.climateAxis), x, z);
            if (view.paintRivers && !view.atlas.rivers.empty() && view.atlas.rivers.at(x, z) != 0u)
                colour = {0.16f, 0.42f, 0.72f};
            if (view.paintSettlement && !view.atlas.settlement.empty())
                switch (view.atlas.settlement.at(x, z))
                {
                case lpl::procgen::SettlementCell::Road: colour = {0.35f, 0.32f, 0.30f}; break;
                case lpl::procgen::SettlementCell::Plaza: colour = {0.55f, 0.50f, 0.44f}; break;
                case lpl::procgen::SettlementCell::Plot: colour = {0.82f, 0.55f, 0.22f}; break;
                default: break;
                }
            if (view.paintRoads && !view.atlas.roads.empty() && view.atlas.roads.at(x, z) != 0u)
                colour = {0.24f, 0.22f, 0.20f};
            rect(x, z, colour, 1.0f);

            // On top of the ground rather than in it: these are questions ABOUT the
            // cell, so they are drawn as a wash and the ground stays readable under
            // them. "This pass did not run" and "this pass produced zero" must not
            // look alike.
            if (view.overlayBlocked && !view.atlas.blocked.empty() && view.atlas.blocked.at(x, z) != 0u)
                rect(x, z, {0.85f, 0.15f, 0.20f}, 0.35f);
            if (view.overlayCaves && !view.atlas.dungeon.empty() &&
                lpl::procgen::isWalkable(view.atlas.dungeon.at(x, z)))
                rect(x, z, {0.95f, 0.62f, 0.14f}, 0.40f);
        }

    if (!liminalView && view.overlayPlots)
        for (lpl::core::usize p = 0u; p < view.atlas.plots.size(); ++p)
        {
            const lpl::procgen::BuildingPlot &plot = view.atlas.plots[p];
            const ImVec2 a{origin.x + static_cast<float>(plot.x) * cell, origin.y + static_cast<float>(plot.z) * cell};
            const ImVec2 b{a.x + static_cast<float>(plot.width) * cell, a.y + static_cast<float>(plot.depth) * cell};
            draw->AddRect(a, b, IM_COL32(255, 190, 90, 220));
        }

    // The creatures, read from the registry — there is no second list of them.
    if (!liminalView && view.livingReady)
    {
        const lpl::core::i32 halfW = static_cast<lpl::core::i32>(width / 2u);
        const lpl::core::i32 halfD = static_cast<lpl::core::i32>(depth / 2u);
        const lpl::ecology::Herd &herd = view.living.herd();
        for (lpl::core::u32 i = 0u; i < herd.size(); ++i)
        {
            lpl::core::u32 row = 0u;
            lpl::ecs::Chunk *chunk = view.registry.chunkOf(herd.at(i), row);
            if (chunk == nullptr || !chunk->archetype().has(lpl::ecs::ComponentId::Creature))
                continue;
            // The write side, like every creature system: nothing here swaps buffers.
            const auto *position = static_cast<const FVec3 *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            const auto *creature = static_cast<const lpl::core::u32 *>(
                chunk->writeComponent(lpl::ecs::ComponentId::Creature));
            if (position == nullptr || creature == nullptr)
                continue;
            const float cx = origin.x + (position[row].x.toFloat() + static_cast<float>(halfW)) * cell;
            const float cz = origin.y + (position[row].z.toFloat() + static_cast<float>(halfD)) * cell;
            const bool hunter = creature[row * 2u] != 0u;
            draw->AddCircleFilled(ImVec2{cx, cz}, cell * 0.4f,
                                  hunter ? IM_COL32(220, 70, 60, 255) : IM_COL32(230, 200, 120, 255));
        }
    }

    // The streaming plan, last: it is a plan about the world and not a thing in it, so
    // it is outlines over everything else. Amber is what the budget would admit THIS
    // tick; the rest of the wanted region stays unmarked, which is the whole point of
    // there being a budget at all.
    if (!liminalView && view.overlayChunks)
    {
        lpl::procgen::StreamingParams params;
        params.generateRadius = static_cast<lpl::core::u32>(view.chunkRadius);
        params.maxGeneratePerTick = 6u;
        lpl::procgen::GenerationSource source;
        source.x = lpl::math::Fixed32::fromFloat(ui.camTarget[0]);
        source.z = lpl::math::Fixed32::fromFloat(ui.camTarget[2]);
        const lpl::procgen::StreamingPlan plan = lpl::procgen::planStreaming(&source, 1u, nullptr, 0u, params);
        const float chunk = static_cast<float>(view.chunkSize) * cell;
        const float halfW = static_cast<float>(width) * 0.5f * cell;
        const float halfD = static_cast<float>(depth) * 0.5f * cell;
        for (lpl::core::u32 i = 0u; i < plan.toGenerate.size(); ++i)
        {
            const ImVec2 a{origin.x + halfW + static_cast<float>(plan.toGenerate[i].coord.x) * chunk,
                           origin.y + halfD + static_cast<float>(plan.toGenerate[i].coord.z) * chunk};
            const ImVec2 b{a.x + chunk, a.y + chunk};
            draw->AddRect(a, b, IM_COL32(255, 174, 41, 200));
        }
        ImGui::Dummy(ImVec2{0.0f, 0.0f});
        ImGui::TextDisabled("%u chunks wanted, %u would be built this tick", plan.wanted,
                            static_cast<lpl::core::u32>(plan.toGenerate.size()));
    }

    // Reserve the canvas so the panel scrolls rather than overlapping what follows.
    ImGui::Dummy(ImVec2{static_cast<float>(width) * cell, static_cast<float>(depth) * cell});
    ImGui::End();
}

void drawUnitCube()
{
    glBegin(GL_QUADS);
    glNormal3f(0.0f, 1.0f, 0.0f); // +Y
    glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);
    glNormal3f(0.0f, -1.0f, 0.0f); // -Y
    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, -0.5f, 0.5f);
    glNormal3f(1.0f, 0.0f, 0.0f); // +X
    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);
    glNormal3f(-1.0f, 0.0f, 0.0f); // -X
    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);
    glNormal3f(0.0f, 0.0f, 1.0f); // +Z
    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glNormal3f(0.0f, 0.0f, -1.0f); // -Z
    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);
    glEnd();
}

// Height → terrain colour ramp (deep → grass → rock → snow), t in [0,1].
void terrainColor(float t, float rgb[3])
{
    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;
    // Four stops with linear blends.
    const float stops[5][3] = {
        {0.16f, 0.30f, 0.48f}, // low / water line
        {0.26f, 0.52f, 0.30f}, // grass
        {0.55f, 0.50f, 0.36f}, // rock
        {0.72f, 0.66f, 0.52f}, // high rock
        {0.96f, 0.97f, 1.00f}, // snow
    };
    const float scaled = t * 4.0f;
    const int i = static_cast<int>(scaled);
    const int lo = i < 4 ? i : 3;
    const float f = scaled - static_cast<float>(lo);
    for (int k = 0; k < 3; ++k)
        rgb[k] = stops[lo][k] + (stops[lo + 1][k] - stops[lo][k]) * f;
}

void drawGrid(int half, float step)
{
    glDisable(GL_LIGHTING);
    glColor3f(0.25f, 0.25f, 0.28f);
    glBegin(GL_LINES);
    for (int i = -half; i <= half; ++i)
    {
        const float t = static_cast<float>(i) * step;
        const float e = static_cast<float>(half) * step;
        glVertex3f(t, 0.0f, -e);
        glVertex3f(t, 0.0f, e);
        glVertex3f(-e, 0.0f, t);
        glVertex3f(e, 0.0f, t);
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

// Sets the viewport + orbital projection/modelview. Shared by the lit render
// pass and the colour-picking pass so both see identical geometry on screen.
void applyCamera(const EditorUi &ui, int fbWidth, int fbHeight)
{
    glViewport(0, 0, fbWidth, fbHeight);
    const float aspect = fbHeight > 0 ? static_cast<float>(fbWidth) / static_cast<float>(fbHeight) : 1.0f;
    const float nearP = 0.1f, farP = 800.0f;
    const float top = nearP * std::tan(60.0f * 0.5f * 3.14159265f / 180.0f);
    const float right = top * aspect;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-right, right, -top, top, nearP, farP);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -ui.camDist);
    glRotatef(ui.camPitch, 1.0f, 0.0f, 0.0f);
    glRotatef(ui.camYaw, 0.0f, 1.0f, 0.0f);
    glTranslatef(-ui.camTarget[0], -ui.camTarget[1], -ui.camTarget[2]);
}

// Reads Position/AABB of entity @p flat (iteration order) into out params.
bool entityTransform(const lpl::ecs::Registry &registry, lpl::core::u32 target, float pos[3], float half[3])
{
    lpl::core::u32 flat = 0;
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            const lpl::core::u32 n = chunk->count();
            const auto *p = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
            const auto *a = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::AABB));
            if (p == nullptr)
            {
                flat += n;
                continue;
            }
            if (target >= flat && target < flat + n)
            {
                const lpl::core::u32 i = target - flat;
                pos[0] = p[i].x.toFloat();
                pos[1] = p[i].y.toFloat();
                pos[2] = p[i].z.toFloat();
                half[0] = a != nullptr ? a[i].x.toFloat() : 0.4f;
                half[1] = a != nullptr ? a[i].y.toFloat() : 0.4f;
                half[2] = a != nullptr ? a[i].z.toFloat() : 0.4f;
                return true;
            }
            flat += n;
        }
    }
    return false;
}

// Draws one cube per entity. In @p pickMode each cube is flat-shaded with a
// colour encoding its (flat index + 1) for GPU colour-picking; otherwise it is
// lit and coloured by height, with the selection highlighted.
void drawEntities(const EditorUi &ui, bool pickMode)
{
    const lpl::ecs::Registry &registry = ui.session.registry();
    const bool hasSel = ui.session.hasSelection();
    const lpl::core::u32 sel = ui.session.selection();
    const float amp = ui.hfAmplitude > 0.1f ? ui.hfAmplitude : 4.0f;
    lpl::core::u32 flat = 0;
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            const lpl::core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
            const auto *aabb = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::AABB));
            if (pos == nullptr)
            {
                flat += n;
                continue;
            }
            for (lpl::core::u32 i = 0; i < n; ++i, ++flat)
            {
                const float x = pos[i].x.toFloat();
                const float y = pos[i].y.toFloat();
                const float z = pos[i].z.toFloat();
                const float sx = aabb != nullptr ? aabb[i].x.toFloat() * 2.0f : 0.8f;
                const float sy = aabb != nullptr ? aabb[i].y.toFloat() * 2.0f : 0.8f;
                const float sz = aabb != nullptr ? aabb[i].z.toFloat() * 2.0f : 0.8f;

                glPushMatrix();
                glTranslatef(x, y, z);
                glScalef(sx, sy, sz);
                if (pickMode)
                {
                    const lpl::core::u32 id = flat + 1u; // 0 = background
                    glColor3ub(static_cast<GLubyte>(id & 0xFFu), static_cast<GLubyte>((id >> 8) & 0xFFu),
                               static_cast<GLubyte>((id >> 16) & 0xFFu));
                    drawUnitCube();
                }
                else
                {
                    float rgb[3];
                    terrainColor((y + amp) / (2.0f * amp), rgb);
                    glColor3f(rgb[0], rgb[1], rgb[2]);
                    drawUnitCube();
                    if (hasSel && flat == sel)
                    {
                        glDisable(GL_LIGHTING);
                        glDisable(GL_DEPTH_TEST);
                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                        glColor3f(1.0f, 0.65f, 0.0f);
                        drawUnitCube();
                        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                        glEnable(GL_DEPTH_TEST);
                        glEnable(GL_LIGHTING);
                    }
                }
                glPopMatrix();
            }
        }
    }
}

// Three world-axis handles (X red, Y green, Z blue) at the selected entity.
void drawGizmo(const EditorUi &ui)
{
    if (!ui.session.hasSelection())
        return;
    float pos[3], half[3];
    if (!entityTransform(ui.session.registry(), ui.session.selection(), pos, half))
        return;
    const float len = 2.5f;
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.5f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.2f, 0.2f);
    glVertex3f(pos[0], pos[1], pos[2]);
    glVertex3f(pos[0] + len, pos[1], pos[2]);
    glColor3f(0.2f, 1.0f, 0.2f);
    glVertex3f(pos[0], pos[1], pos[2]);
    glVertex3f(pos[0], pos[1] + len, pos[2]);
    glColor3f(0.3f, 0.5f, 1.0f);
    glVertex3f(pos[0], pos[1], pos[2]);
    glVertex3f(pos[0], pos[1], pos[2] + len);
    glEnd();
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
}

// Colour-picking: render entity ids to the back buffer, read the pixel under
// the cursor, decode the flat index. Returns -1 if the background was hit.
long pickEntity(const EditorUi &ui, int fbWidth, int fbHeight, double mouseX, double mouseY)
{
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glDisable(GL_DITHER);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    applyCamera(ui, fbWidth, fbHeight);
    drawEntities(ui, /*pickMode=*/true);

    const int px = static_cast<int>(mouseX);
    const int py = fbHeight - 1 - static_cast<int>(mouseY); // GL origin is bottom-left
    GLubyte rgb[3] = {0, 0, 0};
    if (px >= 0 && px < fbWidth && py >= 0 && py < fbHeight)
        glReadPixels(px, py, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    glEnable(GL_DITHER);

    const lpl::core::u32 id = static_cast<lpl::core::u32>(rgb[0]) | (static_cast<lpl::core::u32>(rgb[1]) << 8) |
                              (static_cast<lpl::core::u32>(rgb[2]) << 16);
    return id == 0u ? -1 : static_cast<long>(id - 1u);
}

void renderViewport(const EditorUi &ui, int fbWidth, int fbHeight)
{
    glClearColor(0.10f, 0.12f, 0.16f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    // Distance fog for depth, matched to the clear colour.
    const GLfloat fogColor[4] = {0.10f, 0.12f, 0.16f, 1.0f};
    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogfv(GL_FOG_COLOR, fogColor);
    glFogf(GL_FOG_START, ui.camDist * 0.6f);
    glFogf(GL_FOG_END, ui.camDist + 90.0f);

    applyCamera(ui, fbWidth, fbHeight);

    // Directional light + colour-as-material shading. Set the light position
    // after the camera transform so it stays fixed in world space.
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_NORMALIZE);
    const GLfloat lightDir[4] = {0.4f, 1.0f, 0.5f, 0.0f}; // directional (w=0)
    const GLfloat ambient[4] = {0.35f, 0.36f, 0.40f, 1.0f};
    const GLfloat diffuse[4] = {0.95f, 0.93f, 0.85f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightDir);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);

    drawGrid(24, 1.0f);
    drawEntities(ui, /*pickMode=*/false);
    glDisable(GL_FOG);
    drawGizmo(ui);
}

// Moves the selected entity's Position by (dx,dy,dz) world units via reflection.
void nudgeSelected(EditorUi &ui, double dx, double dy, double dz)
{
    if (!ui.session.hasSelection())
        return;
    const lpl::core::u32 e = ui.session.selection();
    double v = 0.0;
    if (dx != 0.0 && ui.session.getField(e, lpl::ecs::ComponentId::Position, "value", 0u, v))
        (void) ui.session.setField(e, lpl::ecs::ComponentId::Position, "value", 0u, v + dx);
    if (dy != 0.0 && ui.session.getField(e, lpl::ecs::ComponentId::Position, "value", 1u, v))
        (void) ui.session.setField(e, lpl::ecs::ComponentId::Position, "value", 1u, v + dy);
    if (dz != 0.0 && ui.session.getField(e, lpl::ecs::ComponentId::Position, "value", 2u, v))
        (void) ui.session.setField(e, lpl::ecs::ComponentId::Position, "value", 2u, v + dz);
}

// Left-drag orbits the camera; scroll zooms. Ignored while ImGui wants the mouse.
double g_lastX = 0.0, g_lastY = 0.0;
EditorUi *g_ui = nullptr;

void scrollCallback(GLFWwindow * /*w*/, double /*xoff*/, double yoff)
{
    if (g_ui == nullptr || ImGui::GetIO().WantCaptureMouse)
        return;
    g_ui->camDist -= static_cast<float>(yoff) * 2.0f;
    if (g_ui->camDist < 2.0f)
        g_ui->camDist = 2.0f;
    if (g_ui->camDist > 120.0f)
        g_ui->camDist = 120.0f;
}

} // namespace

int main()
{
    lpl::core::Log::info("=== lpl-worldforge (OpenGL editor prototype) ===");

    if (glfwInit() != GLFW_TRUE)
    {
        lpl::core::Log::error("glfwInit failed");
        return 1;
    }
    // Default hints give a legacy-compatible GL context (immediate mode works).
    GLFWwindow *window = glfwCreateWindow(1280, 800, "lpl-worldforge", nullptr, nullptr);
    if (window == nullptr)
    {
        lpl::core::Log::error("glfwCreateWindow failed");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    EditorUi ui;
    g_ui = &ui;
    (void) ui.backend.init();
    (void) ui.session.command(proceduralJson(ui));
    refreshAtlas(ui);
    glfwSetScrollCallback(window, scrollCallback);

    // Left-button state across frames: distinguishes a click (pick) from a drag
    // (orbit), and remembers if the press started over an ImGui panel.
    bool wasDown = false, movedFar = false, pressOverUi = false;
    double pressX = 0.0, pressY = 0.0;

    while (glfwWindowShouldClose(window) == 0)
    {
        glfwPollEvents();

        const ImGuiIO &io = ImGui::GetIO();
        const bool down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        double mx = 0.0, my = 0.0;
        glfwGetCursorPos(window, &mx, &my);

        bool wantPick = false;
        double pickX = 0.0, pickY = 0.0;
        if (down && !wasDown) // press edge
        {
            pressX = mx;
            pressY = my;
            movedFar = false;
            pressOverUi = io.WantCaptureMouse;
        }
        else if (down && wasDown && !pressOverUi) // held: drag orbits the camera
        {
            if (std::fabs(mx - pressX) > 4.0 || std::fabs(my - pressY) > 4.0)
                movedFar = true;
            ui.camYaw += static_cast<float>(mx - g_lastX) * 0.4f;
            ui.camPitch += static_cast<float>(my - g_lastY) * 0.4f;
            if (ui.camPitch > 89.0f)
                ui.camPitch = 89.0f;
            if (ui.camPitch < -89.0f)
                ui.camPitch = -89.0f;
        }
        else if (!down && wasDown && !pressOverUi && !movedFar) // release without drag = click
        {
            wantPick = true;
            pickX = mx;
            pickY = my;
        }
        wasDown = down;
        g_lastX = mx;
        g_lastY = my;

        // Keyboard nudge of the selected entity (arrows = XZ, Q/E = up/down).
        if (ui.session.hasSelection() && !io.WantCaptureKeyboard)
        {
            const double s = 0.15;
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                nudgeSelected(ui, s, 0, 0);
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
                nudgeSelected(ui, -s, 0, 0);
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
                nudgeSelected(ui, 0, 0, s);
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
                nudgeSelected(ui, 0, 0, -s);
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                nudgeSelected(ui, 0, s, 0);
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                nudgeSelected(ui, 0, -s, 0);
        }

        // Advance the authoritative physics when playing (or single-stepping).
        if (ui.playing || ui.stepOnce)
        {
            (void) ui.backend.step(ui.dt);
            ui.stepOnce = false;
        }

        // The preview ecology runs on its own switch: it is a diagnostic on its own
        // registry, so it must not start or stop with the document's physics.
        if (ui.view.livingRunning)
            stepLiving(ui);

        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);

        // A click first runs a colour-picking pass into the back buffer, then the
        // real render overwrites it — so picking is invisible to the user.
        if (wantPick)
        {
            const long hit = pickEntity(ui, fbW, fbH, pickX, pickY);
            if (hit >= 0)
                ui.session.select(static_cast<lpl::core::u32>(hit));
            else
                ui.session.clearSelection();
        }

        renderViewport(ui, fbW, fbH);

        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawHierarchy(ui);
        drawInspector(ui);
        drawProcgen(ui);
        drawSceneIO(ui);
        drawSim(ui);
        drawCaine(ui);
        drawFurnishing(ui);
        drawAtlas(ui);
        ImGui::Render();
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    lpl::core::Log::info("worldforge exited cleanly");
    return 0;
}
