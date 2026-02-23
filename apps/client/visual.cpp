// --- LAPLACE CLIENT (3D) --- //
// File: visual.cpp
// Description: Client OpenGL utilisant le Core engine.
//              Enregistre les systèmes client (BCI, prediction, camera, render, etc.)
//              via le SystemScheduler avec phases PreSwap/PostSwap.
// Auteur: MasterLaplace

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "BciSourceFactory.hpp"
#include "Calibration.hpp"
#include "Core.hpp"
#include "Systems.hpp"

// ─── Camera ───────────────────────────────────────────────────

struct Camera {
    glm::vec3 position = {0.f, 80.f, -120.f};
    glm::vec3 front = {0.f, -0.4f, 1.f};
    glm::vec3 up = {0.f, 1.f, 0.f};
    float speed = 300.f;
    float yaw = 90.f;
    float pitch = -20.f;
};

// ─── Client State ─────────────────────────────────────────────

struct ClientState {
    Core *core = nullptr;
    Camera camera;
    std::unique_ptr<BciSource> bci; ///< Source BCI abstraite (serial/synthetic/lsl/csv)
    Calibration calibration{30.0f}; ///< Machine à états de calibration
    uint32_t myEntityId = 0;
    bool connected = false;
    GLFWwindow *window = nullptr;

    // Neural state (thread-safe)
    std::mutex neuralMutex;
    NeuralState neuralState;
};

static ClientState state;
static std::atomic<bool> running{true};
static std::atomic<bool> keys[512];

// ─── Time Helpers ────────────────────────────────────────────

static double nowSeconds()
{
    using Clock = std::chrono::steady_clock;
    static const auto start = Clock::now();
    auto elapsed = Clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

// ─── Callbacks ────────────────────────────────────────────────

static void keyCallback(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key < 0 || key >= 512)
        return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        keys[key] = true;
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, true);
    }
    else if (action == GLFW_RELEASE)
    {
        keys[key] = false;
    }
}

// ─── Camera Direction ─────────────────────────────────────────

static void updateCameraDirection()
{
    glm::vec3 front;
    front.x = cosf(glm::radians(state.camera.yaw)) * cosf(glm::radians(state.camera.pitch));
    front.y = sinf(glm::radians(state.camera.pitch));
    front.z = sinf(glm::radians(state.camera.yaw)) * cosf(glm::radians(state.camera.pitch));
    state.camera.front = glm::normalize(front);
}

// ─── OpenGL Drawing ───────────────────────────────────────────

static void drawCube(const glm::vec3 &center, const glm::vec3 &halfSize, const glm::vec3 &color)
{
    glColor3f(color.x, color.y, color.z);
    glPushMatrix();
    glTranslatef(center.x, center.y, center.z);

    float hx = halfSize.x, hy = halfSize.y, hz = halfSize.z;

    glBegin(GL_QUADS);
    // Front
    glVertex3f(-hx, -hy, hz);
    glVertex3f(hx, -hy, hz);
    glVertex3f(hx, hy, hz);
    glVertex3f(-hx, hy, hz);
    // Back
    glVertex3f(-hx, -hy, -hz);
    glVertex3f(-hx, hy, -hz);
    glVertex3f(hx, hy, -hz);
    glVertex3f(hx, -hy, -hz);
    // Left
    glVertex3f(-hx, -hy, -hz);
    glVertex3f(-hx, -hy, hz);
    glVertex3f(-hx, hy, hz);
    glVertex3f(-hx, hy, -hz);
    // Right
    glVertex3f(hx, -hy, -hz);
    glVertex3f(hx, hy, -hz);
    glVertex3f(hx, hy, hz);
    glVertex3f(hx, -hy, hz);
    // Top
    glVertex3f(-hx, hy, -hz);
    glVertex3f(-hx, hy, hz);
    glVertex3f(hx, hy, hz);
    glVertex3f(hx, hy, -hz);
    // Bottom
    glVertex3f(-hx, -hy, -hz);
    glVertex3f(hx, -hy, -hz);
    glVertex3f(hx, -hy, hz);
    glVertex3f(-hx, -hy, hz);
    glEnd();

    glPopMatrix();
}

static void drawChunkGrid()
{
    constexpr float CHUNK_SIZE = 1000.f;

    // Ground grid
    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_LINES);
    for (int x = -2000; x <= 2000; x += 500)
    {
        glVertex3f(static_cast<float>(x), -1.f, -2000.f);
        glVertex3f(static_cast<float>(x), -1.f, 2000.f);
    }
    for (int z = -2000; z <= 2000; z += 500)
    {
        glVertex3f(-2000.f, -1.f, static_cast<float>(z));
        glVertex3f(2000.f, -1.f, static_cast<float>(z));
    }
    glEnd();

    // Chunk boundaries
    glColor3f(0.2f, 0.2f, 0.5f);
    glBegin(GL_LINES);
    for (int cx = -8; cx <= 8; ++cx)
    {
        for (int cz = -8; cz <= 8; ++cz)
        {
            float minX = cx * CHUNK_SIZE;
            float maxX = minX + CHUNK_SIZE;
            float minZ = cz * CHUNK_SIZE;
            float maxZ = minZ + CHUNK_SIZE;

            glVertex3f(minX, 0.f, minZ);
            glVertex3f(maxX, 0.f, minZ);
            glVertex3f(maxX, 0.f, minZ);
            glVertex3f(maxX, 0.f, maxZ);
            glVertex3f(maxX, 0.f, maxZ);
            glVertex3f(minX, 0.f, maxZ);
            glVertex3f(minX, 0.f, maxZ);
            glVertex3f(minX, 0.f, minZ);
        }
    }
    glEnd();
}

// ─── Draw Entities ────────────────────────────────────────────

static void drawEntities()
{
    uint32_t readIdx = state.core->world().getReadIdx();
    state.core->world().forEachChunk([&](Partition &p) {
        for (size_t i = 0; i < p.getEntityCount(); ++i)
        {
            auto ent = p.getEntity(i, readIdx);
            glm::vec3 pos(ent.position.x, ent.position.y, ent.position.z);
            glm::vec3 halfSize(ent.size.x * 0.5f, ent.size.y * 0.5f, ent.size.z * 0.5f);

            uint32_t id = p.getEntityId(i);
            glm::vec3 color;
            if (id == state.myEntityId)
                color = {0.1f, 1.f, 0.2f};
            else
                color = {0.4f + (id % 5) * 0.12f, 0.3f + (id % 7) * 0.08f, 0.7f - (id % 3) * 0.15f};

            drawCube(pos, halfSize, color);
        }
    });
}

static size_t countEntities()
{
    size_t total = 0;
    state.core->world().forEachChunk([&](Partition &p) { total += p.getEntityCount(); });
    return total;
}

// ─── HUD ──────────────────────────────────────────────────────

static void drawHUD(GLFWwindow *window)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    if (state.connected)
        glColor3f(0.f, 1.f, 0.f);
    else
        glColor3f(1.f, 0.f, 0.f);

    glBegin(GL_QUADS);
    glVertex2f(10.f, static_cast<float>(height) - 10.f);
    glVertex2f(30.f, static_cast<float>(height) - 10.f);
    glVertex2f(30.f, static_cast<float>(height) - 30.f);
    glVertex2f(10.f, static_cast<float>(height) - 30.f);
    glEnd();

    float barWidth = std::min(200.f, static_cast<float>(countEntities()) * 2.f);
    glColor3f(0.3f, 0.6f, 1.f);
    glBegin(GL_QUADS);
    glVertex2f(40.f, static_cast<float>(height) - 10.f);
    glVertex2f(40.f + barWidth, static_cast<float>(height) - 10.f);
    glVertex2f(40.f + barWidth, static_cast<float>(height) - 25.f);
    glVertex2f(40.f, static_cast<float>(height) - 25.f);
    glEnd();

    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// ─── Camera System ────────────────────────────────────────────

static bool getEntityPosition(uint32_t id, Vec3 &outPos)
{
    int localIdx = -1;
    Partition *chunk = state.core->world().findEntity(id, localIdx);
    if (!chunk || localIdx < 0)
        return false;

    uint32_t readIdx = state.core->world().getReadIdx();
    auto ent = chunk->getEntity(static_cast<size_t>(localIdx), readIdx);
    outPos = ent.position;
    return true;
}

static void updateCameraSystem(float dt)
{
    constexpr float ROT_SPEED = 60.f;

    if (keys[GLFW_KEY_LEFT])
    {
        state.camera.yaw -= ROT_SPEED * dt;
        updateCameraDirection();
    }
    if (keys[GLFW_KEY_RIGHT])
    {
        state.camera.yaw += ROT_SPEED * dt;
        updateCameraDirection();
    }
    if (keys[GLFW_KEY_UP])
    {
        state.camera.pitch += ROT_SPEED * dt;
        if (state.camera.pitch > 89.f)
            state.camera.pitch = 89.f;
        updateCameraDirection();
    }
    if (keys[GLFW_KEY_DOWN])
    {
        state.camera.pitch -= ROT_SPEED * dt;
        if (state.camera.pitch < -89.f)
            state.camera.pitch = -89.f;
        updateCameraDirection();
    }

    if (state.connected && state.myEntityId != 0)
    {
        Vec3 pos;
        if (getEntityPosition(state.myEntityId, pos))
        {
            glm::vec3 target(pos.x, pos.y, pos.z);
            state.camera.position = target - state.camera.front * 60.f + glm::vec3(0.f, 30.f, 0.f);
            return;
        }
    }

    glm::vec3 movement(0.f);
    if (keys[GLFW_KEY_W])
        movement += state.camera.front;
    if (keys[GLFW_KEY_S])
        movement -= state.camera.front;
    if (keys[GLFW_KEY_A])
        movement -= glm::normalize(glm::cross(state.camera.front, state.camera.up));
    if (keys[GLFW_KEY_D])
        movement += glm::normalize(glm::cross(state.camera.front, state.camera.up));

    if (glm::length(movement) > 0.1f)
        state.camera.position += glm::normalize(movement) * state.camera.speed * dt;
}

// ─── Render System ────────────────────────────────────────────

static void renderSystem(GLFWwindow *window)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glm::mat4 proj =
        glm::perspective(glm::radians(45.f), static_cast<float>(width) / static_cast<float>(height), 0.1f, 5000.f);
    glLoadMatrixf(glm::value_ptr(proj));

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glm::vec3 target = state.camera.position + state.camera.front;
    glm::mat4 view = glm::lookAt(state.camera.position, target, state.camera.up);
    glLoadMatrixf(glm::value_ptr(view));

    drawChunkGrid();
    drawEntities();
    drawHUD(window);

    glfwSwapBuffers(window);
    glfwPollEvents();

    if (glfwWindowShouldClose(window))
    {
        running = false;
        state.core->stop();
    }
}

// ─── Systems Setup ────────────────────────────────────────────

static void setupSystems(Core &core)
{
    // ── PreSwap Systems ──────────────────────────────────────

    // 1. Network Receive
    core.registerSystem(Systems::NetworkReceiveSystem(core.network(), core.packetQueue()));

    // 2. Welcome (handle MSG_WELCOME)
    core.registerSystem(Systems::WelcomeSystem(core.packetQueue(), state.myEntityId, state.connected));

    // 3. State Reconciliation (handle MSG_STATE)
    core.registerSystem(Systems::StateReconciliationSystem(core.packetQueue()));

    // 4. BCI System (source abstraite + calibration)
    core.registerSystem({
        "BCISystem", -10,
        [&core](WorldPartition &/*w*/, float /*dt*/) {
            if (!state.bci)
                return;
            std::lock_guard<std::mutex> lock(state.neuralMutex);
            state.bci->update(state.neuralState);
            state.calibration.tick(state.neuralState);

            // Update InputManager with neural data
            if (state.connected && state.myEntityId != 0)
            {
                core.inputManager().setNeural(state.myEntityId,
                    state.neuralState.alphaPower,
                    state.neuralState.betaPower,
                    state.neuralState.concentration,
                    state.neuralState.blinkDetected);
            }
        },
        {},
        SchedulePhase::PreSwap
    });

    // 5. Local Input System (GLFW keys → InputManager)
    core.registerSystem({
        "LocalInput", -8,
        [&core](WorldPartition &/*w*/, float /*dt*/) {
            if (!state.connected || state.myEntityId == 0)
                return;

            auto &im = core.inputManager();
            im.getOrCreate(state.myEntityId);

            // Sync WASD keys from GLFW atomics to InputManager
            constexpr int trackedKeys[] = {GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D};
            for (int key : trackedKeys)
                im.setKeyState(state.myEntityId, static_cast<uint16_t>(key), keys[key].load());
        },
        {},
        SchedulePhase::PreSwap
    });

    // 6. Spawn System (create local entity if needed)
    core.registerSystem({
        "Spawn", -6,
        [](WorldPartition &w, float /*dt*/) {
            if (state.connected && state.myEntityId != 0 &&
                !w.isEntityRegistered(state.myEntityId))
            {
                Partition::EntitySnapshot ent{};
                ent.id = state.myEntityId;
                ent.position = {0.f, 10.f, 0.f};
                ent.velocity = {0.f, 0.f, 0.f};
                ent.size = {1.f, 2.f, 1.f};
                ent.health = 100;
                w.addEntity(ent);
            }
        },
        {{ComponentId::Position, AccessMode::Write}},
        SchedulePhase::PreSwap
    });

    // 7. Movement (client-side prediction via InputManager)
    core.registerSystem(Systems::MovementSystem(core.inputManager()));

    // 8. Physics
    core.registerSystem(Systems::PhysicsSystem());

    // 9. Input Send System (serialize inputs → send to server)
    core.registerSystem({
        "InputSend", 15,
        [&core](WorldPartition &/*w*/, float /*dt*/) {
            if (!state.connected || state.myEntityId == 0)
                return;

            std::vector<uint8_t> kData, aData, nData;

            // WASD key states
            constexpr int trackedKeys[] = {GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D};
            for (int key : trackedKeys)
            {
                bool currentKey = keys[key].load();
                uint16_t packed = (static_cast<uint16_t>(key) & 0x7FFF);
                if (currentKey)
                    packed |= 0x8000;

                kData.push_back(static_cast<uint8_t>(packed & 0xFF));
                kData.push_back(static_cast<uint8_t>((packed >> 8) & 0xFF));
            }

            // Neural data
            {
                std::lock_guard<std::mutex> lock(state.neuralMutex);
                nData.resize(13);
                memcpy(nData.data(), &state.neuralState.alphaPower, 4);
                memcpy(nData.data() + 4, &state.neuralState.betaPower, 4);
                memcpy(nData.data() + 8, &state.neuralState.concentration, 4);
                nData[12] = state.neuralState.blinkDetected ? 1 : 0;
            }

            if (!kData.empty() || !aData.empty() || !nData.empty())
                core.network().send_inputs(state.myEntityId, kData, aData, nData);
        },
        {},
        SchedulePhase::PreSwap
    });

    // ── PostSwap Systems ─────────────────────────────────────

    // 10. Camera System (PostSwap — reads from read buffer)
    core.registerSystem({
        "Camera", 0,
        [](WorldPartition &/*w*/, float dt) { updateCameraSystem(dt); },
        {},
        SchedulePhase::PostSwap
    });

    // 11. Render System (PostSwap — reads from read buffer)
    core.registerSystem({
        "Render", 10,
        [](WorldPartition &/*w*/, float /*dt*/) {
            if (state.window)
                renderSystem(state.window);
            else
                glfwPollEvents();
        },
        {
            {ComponentId::Position, AccessMode::Read},
            {ComponentId::Size,     AccessMode::Read},
            {ComponentId::Health,   AccessMode::Read},
        },
        SchedulePhase::PostSwap
    });

    core.buildSchedule();

#ifdef LPL_MONITORING
    core.printSchedule();
#endif
}

// ─── Init Window ──────────────────────────────────────────────

static GLFWwindow *initWindow()
{
    if (!glfwInit())
    {
        std::cerr << "[ERROR] GLFW init failed\n";
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow *window =
        glfwCreateWindow(1440, 900, "LplPlugin Client — WASD: move | Arrows: camera | ESC: quit", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "[ERROR] Window creation failed\n";
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "[ERROR] GLEW init failed\n";
        return nullptr;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.05f, 0.05f, 0.1f, 1.f);

    updateCameraDirection();

    return window;
}

// ─── MAIN ─────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    const char *serverIp = "127.0.0.1";
    uint16_t serverPort = 7777;

    // Parse BCI arguments (--bci-mode, --csv-file, --calib-duration, etc.)
    BciConfig bciCfg = bci_parse_args(argc, argv);

    // Parse legacy positional args (IP, port)
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg.rfind("--", 0) != 0)
        {
            if (serverIp == std::string("127.0.0.1") && arg.find('.') != std::string::npos)
                serverIp = argv[i];
            else
            {
                int p = atoi(argv[i]);
                if (p > 0 && p < 65536)
                    serverPort = static_cast<uint16_t>(p);
            }
        }
    }

    std::cout << "[MAIN] Starting LplPlugin Client...\n";

    // 1. Init window
    std::cout << "[MAIN] Initializing window...\n";
    GLFWwindow *window = initWindow();
    if (!window)
    {
        std::cerr << "[FATAL] Window init failed\n";
        return 1;
    }
    state.window = window;
    std::cout << "[MAIN] Window created successfully\n";

    // 2. Init Core (GPU + Network)
    std::cout << "[MAIN] Initializing Core...\n";
    Core core;
    state.core = &core;

    // 3. Init client network
    std::cout << "[MAIN] Initializing network...\n";
    core.initClientNetwork(serverIp, serverPort);
    std::cout << "[MAIN] Network initialized\n";

    // 4. Init BCI (abstracted source with fallback)
    std::cout << "[MAIN] Initializing BCI (mode: " << bci_mode_name(bciCfg.mode) << ")...\n";
    state.bci = BciSourceFactory::createAndInit(bciCfg);
    if (state.bci)
    {
        std::cout << "[CLIENT] BCI source active: " << state.bci->name() << "\n";
        state.calibration = Calibration(bciCfg.calibDuration);
        state.calibration.start();
    }
    else
    {
        std::cout << "[CLIENT] BCI init failed (continuing with keyboard only)\n";
    }

    // 5. Setup ECS Systems
    std::cout << "[MAIN] Setting up ECS systems...\n";
    setupSystems(core);
    std::cout << "[MAIN] ECS systems ready\n";

    std::cout << "[CLIENT] Waiting for MSG_WELCOME...\n"
              << "  WASD    : move entity\n"
              << "  Arrows  : rotate camera\n"
              << "  ESC     : quit\n\n";

    // 6. Main loop using Core.runVariableDt
    std::cout << "[MAIN] Entering main loop...\n";
    double lastFrameTime = nowSeconds();

    core.runVariableDt(
        [&lastFrameTime]() -> float {
            double currentTime = nowSeconds();
            float dt = static_cast<float>(currentTime - lastFrameTime);
            lastFrameTime = currentTime;
            return dt;
        },
        [](float /*dt*/) {
            // Post-loop: check if we need to stop
            if (!running)
                state.core->stop();
        }
    );

    // 7. Cleanup
    std::cout << "[MAIN] Shutting down...\n";
    if (state.bci)
        state.bci->stop();
    glfwTerminate();

    std::cout << "[CLIENT] Shutdown complete\n";
    return 0;
}
