// --- LAPLACE TEST CLIENT (3D) --- //
// TEMPORARY TEST CLIENT — will be replaced by production architecture
// File: visual.cpp
// Description: Client OpenGL qui se connecte au serveur LplPlugin (UDP 7777),
//              fait une simulation locale (prediction) et corrige avec MSG_STATE.
// Auteur: MasterLaplace & Copilot

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
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "Network.hpp"
#include "OpenBCIDriver.hpp"
#include "SystemScheduler.hpp"
#include "WorldPartition.hpp"
#include "lpl_protocol.h"

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
    WorldPartition world;
    Camera camera;
    Network network;
    OpenBCIDriver bci;
    SystemScheduler scheduler;
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
    uint32_t readIdx = state.world.getReadIdx();
    state.world.forEachChunk([&](Partition &p) {
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
    state.world.forEachChunk([&](Partition &p) { total += p.getEntityCount(); });
    return total;
}

// ─── HUD ──────────────────────────────────────────────────────

static void drawHUD(GLFWwindow *window)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Switch to 2D orthographic projection for HUD
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    // Connection status indicator (top-left corner)
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

    // Entity count indicator (small bar)
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

// ─── Input Helpers ───────────────────────────────────────────

static Vec3 getInputDirection()
{
    glm::vec3 dir(0.f);
    if (keys[GLFW_KEY_W])
        dir += state.camera.front;
    if (keys[GLFW_KEY_S])
        dir -= state.camera.front;
    if (keys[GLFW_KEY_A])
        dir -= glm::normalize(glm::cross(state.camera.front, state.camera.up));
    if (keys[GLFW_KEY_D])
        dir += glm::normalize(glm::cross(state.camera.front, state.camera.up));

    // Flatten to XZ plane
    dir.y = 0.f;

    if (glm::length(dir) > 0.1f)
        dir = glm::normalize(dir);
    else
        dir = glm::vec3(0.f);

    return Vec3{dir.x, dir.y, dir.z};
}

// ─── Camera System ────────────────────────────────────────────

static bool getEntityPosition(uint32_t id, Vec3 &outPos)
{
    int localIdx = -1;
    Partition *chunk = state.world.findEntity(id, localIdx);
    if (!chunk || localIdx < 0)
        return false;

    uint32_t readIdx = state.world.getReadIdx();
    auto ent = chunk->getEntity(static_cast<size_t>(localIdx), readIdx);
    outPos = ent.position;
    return true;
}

static void updateCameraSystem(float dt)
{
    constexpr float ROT_SPEED = 60.f;

    // Camera rotation with arrow keys
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

    // Follow player entity if connected
    if (state.connected && state.myEntityId != 0)
    {
        Vec3 pos;
        if (getEntityPosition(state.myEntityId, pos))
        {
            glm::vec3 target(pos.x, pos.y, pos.z);
            state.camera.position = target - state.camera.front * 60.f + glm::vec3(0.f, 30.f, 0.f);
            return; // Camera locked on player, don't do free movement
        }
    }

    // Free camera movement with WASD (when not following player)
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
        running = false;
}

// ─── Systems Setup ────────────────────────────────────────────

static void setupSystems(SystemScheduler &sched)
{
    // 1. Network System (Receive)
    sched.registerSystem({
        "NetworkSystem",
        -20,
        [](WorldPartition &w, float dt) {
            state.network.network_consume_packets(w);

            if (!state.connected && state.network.is_connected())
            {
                state.connected = true;
                state.myEntityId = state.network.get_local_entity_id();
                std::cout << "[CLIENT] Connected! Entity ID: " << state.myEntityId << "\n";
            }
                                         },
        {{ComponentId::Position, AccessMode::Write},
                                         {ComponentId::Velocity, AccessMode::Write},
                                         {ComponentId::Health, AccessMode::Write}}
    });

    // 2. BCI System
    sched.registerSystem({"BCISystem",
                          -10,
                          [](WorldPartition &w, float dt) {
                              std::lock_guard<std::mutex> lock(state.neuralMutex);
                              state.bci.update(state.neuralState);
                          },
                          {}});

    // 3. Spawn System
    sched.registerSystem({"SpawnSystem",
                          -5,
                          [](WorldPartition &w, float dt) {
                              if (state.connected && state.myEntityId != 0 &&
                                  !state.world.isEntityRegistered(state.myEntityId))
                              {
                                  Partition::EntitySnapshot ent{};
                                  ent.id = state.myEntityId;
                                  ent.position = {0.f, 10.f, 0.f};
                                  ent.velocity = {0.f, 0.f, 0.f};
                                  ent.size = {1.f, 2.f, 1.f};
                                  ent.health = 100;
                                  state.world.addEntity(ent);
                              }
                          },
                          {{ComponentId::Position, AccessMode::Write}}});

    // 4. Local Prediction System
    sched.registerSystem({"PredictionSystem",
                          0,
                          [](WorldPartition &w, float dt) {
                              if (!state.connected || state.myEntityId == 0)
                                  return;

                              int localIdx = -1;
                              Partition *chunk = w.findEntity(state.myEntityId, localIdx);
                              if (!chunk || localIdx < 0)
                                  return;

                              uint32_t writeIdx = w.getWriteIdx();
                              constexpr float SPEED = 50.0f;

                              Vec3 inputDir = getInputDirection();
                              Vec3 targetVel = inputDir * SPEED;

                              // Apply BCI blink for jump
                              {
                                  std::lock_guard<std::mutex> lock(state.neuralMutex);
                                  if (state.neuralState.blinkDetected)
                                      targetVel.y += 10.0f;
                              }

                              if (glm::length(glm::vec3(inputDir.x, inputDir.y, inputDir.z)) > 0.01f)
                              {
                                  chunk->setVelocity(static_cast<uint32_t>(localIdx), targetVel, writeIdx);
                                  chunk->wakeEntity(static_cast<uint32_t>(localIdx));
                              }
                          },
                          {{ComponentId::Velocity, AccessMode::Write}}});

    // 5. Physics System
    sched.registerSystem({
        "PhysicsSystem",
        5,
        [](WorldPartition &w, float dt) { w.step(dt); },
        {{ComponentId::Position, AccessMode::Write}, {ComponentId::Velocity, AccessMode::Write}}
    });

    // 6. Camera System
    sched.registerSystem({"CameraSystem", 10, [](WorldPartition &w, float dt) { updateCameraSystem(dt); }, {}});

    // 7. Input System (Send to Server)
    sched.registerSystem({"InputSystem",
                          15,
                          [](WorldPartition &w, float dt) {
                              if (!state.connected || state.myEntityId == 0)
                                  return;

                              static uint32_t frameCounter = 0;
                              frameCounter++;

                              std::vector<uint8_t> kData, aData, nData;

                              // Send key states every frame (bitpacked)
                              // We track which keys changed to minimize bandwidth
                              static bool lastKeys[512] = {};

                              for (int k = 0; k < 512; k++)
                              {
                                  bool currentKey = keys[k].load();
                                  if (currentKey != lastKeys[k])
                                  {
                                      lastKeys[k] = currentKey;

                                      // Bitpack: Key(15) | Pressed(1)
                                      uint16_t packed = (static_cast<uint16_t>(k) & 0x7FFF);
                                      if (currentKey)
                                          packed |= 0x8000;

                                      kData.push_back(static_cast<uint8_t>(packed & 0xFF));
                                      kData.push_back(static_cast<uint8_t>((packed >> 8) & 0xFF));
                                  }
                              }

                              // Send neural data every 10 frames
                              if (frameCounter % 10 == 0)
                              {
                                  std::lock_guard<std::mutex> lock(state.neuralMutex);

                                  nData.resize(13);
                                  memcpy(nData.data(), &state.neuralState.alphaPower, 4);
                                  memcpy(nData.data() + 4, &state.neuralState.betaPower, 4);
                                  memcpy(nData.data() + 8, &state.neuralState.concentration, 4);
                                  nData[12] = state.neuralState.blinkDetected ? 1 : 0;
                              }

                              // Only send if we have data
                              if (!kData.empty() || !aData.empty() || !nData.empty())
                              {
                                  state.network.send_inputs(state.myEntityId, kData, aData, nData);
                              }
                          },
                          {}});

    sched.buildSchedule();

#ifdef LPL_MONITORING
    sched.printSchedule();
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

    GLFWwindow *window = glfwCreateWindow(1440, 900, "LplPlugin Client — WASD: move | Arrows: camera | ESC: quit", nullptr, nullptr);
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

// ─── Init Network ─────────────────────────────────────────────

static bool initNetwork(const char *serverIp, uint16_t serverPort)
{
    if (!state.network.network_init())
    {
        std::cerr << "[ERROR] Network init failed\n";
        return false;
    }

    state.network.set_server_info(serverIp, serverPort);
    state.network.send_connect(serverIp, serverPort);

    std::cout << "[CLIENT] MSG_CONNECT sent to " << serverIp << ":" << serverPort << "\n";
    return true;
}

// ─── MAIN ─────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    const char *serverIp = "127.0.0.1";
    uint16_t serverPort = 7777;

    if (argc >= 2)
        serverIp = argv[1];
    if (argc >= 3)
        serverPort = static_cast<uint16_t>(atoi(argv[2]));

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

    // 2. Init network
    std::cout << "[MAIN] Initializing network...\n";
    if (!initNetwork(serverIp, serverPort))
    {
        std::cerr << "[FATAL] Network init failed\n";
        glfwTerminate();
        return 1;
    }
    std::cout << "[MAIN] Network initialized\n";

    // 3. Init BCI (optional)
    std::cout << "[MAIN] Initializing BCI...\n";
    if (state.bci.init("/dev/ttyUSB0"))
        std::cout << "[CLIENT] OpenBCI driver started on /dev/ttyUSB0\n";
    else
        std::cout << "[CLIENT] OpenBCI init failed (continuing with keyboard only)\n";

    // 4. Setup ECS Systems
    std::cout << "[MAIN] Setting up ECS systems...\n";
    setupSystems(state.scheduler);
    std::cout << "[MAIN] ECS systems ready\n";

    std::cout << "[CLIENT] Waiting for MSG_WELCOME...\n"
              << "  WASD    : move entity\n"
              << "  Arrows  : rotate camera\n"
              << "  ESC     : quit\n\n";

    // 5. Main loop
    std::cout << "[MAIN] Entering main loop...\n";
    double lastFrameTime = nowSeconds();

    while (running)
    {
        double currentTime = nowSeconds();
        float dt = static_cast<float>(currentTime - lastFrameTime);
        lastFrameTime = currentTime;

        // Limit delta to avoid huge jumps
        if (dt > 0.1f)
            dt = 0.1f;

        // Run all ECS systems
        state.scheduler.ordered_tick(state.world, dt);

        // Swap buffers after all systems ran
        state.world.swapBuffers();

        // Render (must be on main thread for OpenGL)
        if (state.window)
        {
            renderSystem(state.window);
        }
        else
        {
            // If no window, still need to poll events to check for window close request
            glfwPollEvents();
        }
    }

    // 6. Cleanup
    std::cout << "[MAIN] Shutting down...\n";
    running = false;
    state.bci.stop();
    state.network.network_cleanup();
    glfwTerminate();

    std::cout << "[CLIENT] Shutdown complete\n";
    return 0;
}
