// --- LAPLACE TEST CLIENT (3D) --- //
// TEMPORARY TEST CLIENT — will be replaced by production architecture
// File: visual3d.cpp
// Description: Client OpenGL qui se connecte au serveur LplPlugin (UDP 7777),
//              fait une simulation locale (prediction) et corrige avec MSG_STATE.
// Auteur: MasterLaplace & Copilot

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "lpl_protocol.h" // MSG_*, MAX_PACKET_SIZE
#include "Math.hpp"        // Vec3
#include "WorldPartition.hpp"
#include "Network.hpp"

// ─── Camera ───────────────────────────────────────────────────

struct Camera {
    glm::vec3 position = {0.f, 80.f, -120.f};
    glm::vec3 front = {0.f, -0.4f, 1.f};
    glm::vec3 up = {0.f, 1.f, 0.f};
    float speed = 300.f;
    float yaw = 90.f;
    float pitch = -20.f;
};

// ─── Network Updates ─────────────────────────────────────────

struct NetEntityUpdate {
    uint32_t id;
    Vec3 pos;
    Vec3 size;
    int32_t health;
    double time;
};

struct ServerHistory {
    Vec3 pos;
    double time = 0.0;
    bool valid = false;
};

// ─── Client State ─────────────────────────────────────────────

struct ClientState {
    WorldPartition world;
    Camera camera;
    Network network;
    uint32_t myEntityId = 0;
    bool connected = false;
};

static ClientState state;
static std::atomic<bool> running{true};
static bool keys[512] = {};

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
    if (key < 0 || key >= 512) return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        keys[key] = true;
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
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
    glVertex3f(-hx, -hy,  hz); glVertex3f( hx, -hy,  hz);
    glVertex3f( hx,  hy,  hz); glVertex3f(-hx,  hy,  hz);
    // Back
    glVertex3f(-hx, -hy, -hz); glVertex3f(-hx,  hy, -hz);
    glVertex3f( hx,  hy, -hz); glVertex3f( hx, -hy, -hz);
    // Left
    glVertex3f(-hx, -hy, -hz); glVertex3f(-hx, -hy,  hz);
    glVertex3f(-hx,  hy,  hz); glVertex3f(-hx,  hy, -hz);
    // Right
    glVertex3f( hx, -hy, -hz); glVertex3f( hx,  hy, -hz);
    glVertex3f( hx,  hy,  hz); glVertex3f( hx, -hy,  hz);
    // Top
    glVertex3f(-hx,  hy, -hz); glVertex3f(-hx,  hy,  hz);
    glVertex3f( hx,  hy,  hz); glVertex3f( hx,  hy, -hz);
    // Bottom
    glVertex3f(-hx, -hy, -hz); glVertex3f( hx, -hy, -hz);
    glVertex3f( hx, -hy,  hz); glVertex3f(-hx, -hy,  hz);
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
        glVertex3f(static_cast<float>(x), -1.f,  2000.f);
    }
    for (int z = -2000; z <= 2000; z += 500)
    {
        glVertex3f(-2000.f, -1.f, static_cast<float>(z));
        glVertex3f( 2000.f, -1.f, static_cast<float>(z));
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

            glVertex3f(minX, 0.f, minZ); glVertex3f(maxX, 0.f, minZ);
            glVertex3f(maxX, 0.f, minZ); glVertex3f(maxX, 0.f, maxZ);
            glVertex3f(maxX, 0.f, maxZ); glVertex3f(minX, 0.f, maxZ);
            glVertex3f(minX, 0.f, maxZ); glVertex3f(minX, 0.f, minZ);
        }
    }
    glEnd();
}


// ─── Input Helpers ───────────────────────────────────────────

static Vec3 getInputDirection()
{
    glm::vec3 dir(0.f);
    if (keys[GLFW_KEY_W]) dir.z += 1.f;
    if (keys[GLFW_KEY_S]) dir.z -= 1.f;
    if (keys[GLFW_KEY_A]) dir.x -= 1.f;
    if (keys[GLFW_KEY_D]) dir.x += 1.f;

    if (glm::length(dir) > 0.1f)
        dir = glm::normalize(dir);
    else
        dir = glm::vec3(0.f);

    return Vec3{dir.x, dir.y, dir.z};
}

static void sendInput(const Vec3 &dir)
{
    if (!state.connected || state.myEntityId == 0)
        return;

    state.network.send_input(state.myEntityId, dir);
}

// ─── Update Camera ────────────────────────────────────────────

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

static void updateCamera(double dt)
{
    constexpr float ROT_SPEED = 60.f;

    if (keys[GLFW_KEY_LEFT])  { state.camera.yaw -= ROT_SPEED * static_cast<float>(dt); updateCameraDirection(); }
    if (keys[GLFW_KEY_RIGHT]) { state.camera.yaw += ROT_SPEED * static_cast<float>(dt); updateCameraDirection(); }
    if (keys[GLFW_KEY_UP])
    {
        state.camera.pitch += ROT_SPEED * static_cast<float>(dt);
        if (state.camera.pitch > 89.f) state.camera.pitch = 89.f;
        updateCameraDirection();
    }
    if (keys[GLFW_KEY_DOWN])
    {
        state.camera.pitch -= ROT_SPEED * static_cast<float>(dt);
        if (state.camera.pitch < -89.f) state.camera.pitch = -89.f;
        updateCameraDirection();
    }

    // Camera follows player entity with offset
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

    // Free camera avant connexion / entite absente
    glm::vec3 movement(0.f);
    if (keys[GLFW_KEY_W]) movement += state.camera.front;
    if (keys[GLFW_KEY_S]) movement -= state.camera.front;
    if (keys[GLFW_KEY_A]) movement -= glm::normalize(glm::cross(state.camera.front, state.camera.up));
    if (keys[GLFW_KEY_D]) movement += glm::normalize(glm::cross(state.camera.front, state.camera.up));

    if (glm::length(movement) > 0.1f)
        state.camera.position += glm::normalize(movement) * state.camera.speed * static_cast<float>(dt);
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
                color = {
                    0.4f + (id % 5) * 0.12f,
                    0.3f + (id % 7) * 0.08f,
                    0.7f - (id % 3) * 0.15f
                };

            drawCube(pos, halfSize, color);
        }
    });
}

static size_t countEntities()
{
    size_t total = 0;
    state.world.forEachChunk([&](Partition &p) {
        total += p.getEntityCount();
    });
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

// ─── Init Window ──────────────────────────────────────────────

static GLFWwindow *initWindow()
{
    if (!glfwInit())
    {
        std::cerr << "GLFW initialization failed\n";
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow *window = glfwCreateWindow(1440, 900,
        "LplPlugin Client — WASD: move | Arrows: camera | ESC: quit", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Window creation failed\n";
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "GLEW initialization failed\n";
        return nullptr;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.05f, 0.05f, 0.1f, 1.f);
    return window;
}

// ─── Init Network ─────────────────────────────────────────────

static bool initNetwork(const char *serverIp, uint16_t serverPort)
{
    if (!state.network.network_init())
    {
        printf("[FATAL] Network init failed (driver missing?)\n");
        return false;
    }

    state.network.set_server_info(serverIp, serverPort);
    state.network.send_connect(serverIp, serverPort);

    std::cout << "[CLIENT] MSG_CONNECT envoye a " << serverIp << ":" << serverPort << "\n";
    return true;
}

// ─── MAIN ─────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    const char *serverIp = "127.0.0.1";
    uint16_t serverPort = 7777;

    if (argc >= 2) serverIp = argv[1];
    if (argc >= 3) serverPort = static_cast<uint16_t>(atoi(argv[2]));

    // 1. Init window
    GLFWwindow *window = initWindow();
    if (!window)
        return 1;

    // 2. Init network + connect
    if (!initNetwork(serverIp, serverPort))
    {
        glfwTerminate();
        return 1;
    }

    std::cout << "[CLIENT] En attente de MSG_WELCOME du serveur...\n"
              << "  WASD    : deplacer l'entite\n"
              << "  Fleches : orienter la camera\n"
              << "  ESC     : quitter\n\n";

    // 3. Main loop
    double lastFrameTime = nowSeconds();

    while (!glfwWindowShouldClose(window) && running)
    {
        double currentTime = nowSeconds();
        double dt = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        // Sync server updates
        state.network.network_consume_packets(state.world);

        if (!state.connected && state.network.is_connected()) {
            state.connected = true;
            state.myEntityId = state.network.get_local_entity_id();
            std::cout << "[CLIENT] Connected (synced with Network class) Entity: " << state.myEntityId << "\n";
        }

        // Ensure local player exists for prediction
        if (state.connected && state.myEntityId != 0 &&
            !state.world.isEntityRegistered(state.myEntityId))
        {
            Partition::EntitySnapshot ent{};
            ent.id = state.myEntityId;
            ent.position = {0.f, 10.f, 0.f};
            ent.rotation = {0.f, 0.f, 0.f, 1.f};
            ent.velocity = {0.f, 0.f, 0.f};
            ent.mass = 1.f;
            ent.force = {0.f, 0.f, 0.f};
            ent.size = {1.f, 2.f, 1.f};
            ent.health = 100;
            state.world.addEntity(ent);
        }

        // Local prediction: apply input velocity
        Vec3 inputDir = getInputDirection();
        if (state.connected && state.myEntityId != 0)
        {
            int localIdx = -1;
            Partition *chunk = state.world.findEntity(state.myEntityId, localIdx);
            if (chunk && localIdx >= 0)
            {
                constexpr float PLAYER_SPEED = 50.0f;
                Vec3 vel = inputDir * PLAYER_SPEED;
                uint32_t writeIdx = state.world.getWriteIdx();
                chunk->setVelocity(static_cast<uint32_t>(localIdx), vel, writeIdx);
                chunk->wakeEntity(static_cast<uint32_t>(localIdx));
            }
        }

        // Step local simulation
        state.world.step(static_cast<float>(dt));
        state.world.swapBuffers();

        // Send inputs to server
        sendInput(inputDir);

        // Update camera
        updateCamera(dt);

        // Render
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glm::mat4 proj = glm::perspective(glm::radians(45.f),
            static_cast<float>(width) / static_cast<float>(height), 0.1f, 5000.f);
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
    }

    // 4. Cleanup
    running = false;
    state.network.network_cleanup();
    glfwTerminate();
    return 0;
}
