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
#include <lpl/physics/CpuPhysicsBackend.hpp>

namespace {

using FVec3 = lpl::math::Vec3<lpl::math::Fixed32>;

// Biome names in lpl::procgen::BiomeId order, which is also the spelling
// lpl::procgen::biomeIdByName accepts — the combo writes the document's own
// vocabulary rather than an enum index a reordering would silently reinterpret.
constexpr const char *kBiomeNames[] = {"ocean",  "beach",   "snow",      "tundra", "taiga",      "rock",
                                       "desert", "savanna", "grassland", "forest", "rainforest", "marsh"};

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
};

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
        const auto r = ui.session.command(cmd);
        ui.lastReport = r.has_value() ? r.value() : "generate failed";
        ui.lastGate = ui.lastReport;
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
    (void) ui.session.command(R"({"cmd":"generate_world","seed":1337,"width":24,"depth":24,
                                  "caves":{"width":24,"depth":24,"minRegionSize":12},
                                  "gate":{"minPathLength":4,"minWalkableCells":16}})");
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
