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
    std::unordered_map<uint32_t, ServerHistory> history;
    std::vector<NetEntityUpdate> pendingUpdates;
    std::mutex pendingMutex;
    Camera camera;
    uint32_t myEntityId = 0;
    std::atomic<bool> connected{false};
    int sock = -1;
    sockaddr_in serverAddr{};
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

// ─── Network Thread ───────────────────────────────────────────

static void networkThread()
{
    uint8_t buf[1400];

    while (running)
    {
        sockaddr_in from{};
        socklen_t fromLen = sizeof(from);
        ssize_t n = recvfrom(state.sock, buf, sizeof(buf), 0,
                             reinterpret_cast<sockaddr *>(&from), &fromLen);
        if (n <= 0)
            continue;

        switch (buf[0])
        {
        case MSG_WELCOME: {
            if (n >= 5)
            {
                state.myEntityId = *reinterpret_cast<uint32_t *>(buf + 1);
                state.connected = true;
                std::cout << "[CLIENT] Connecte au serveur! Mon entite: #"
                          << state.myEntityId << "\n";
            }
            break;
        }
        case MSG_STATE: {
            if (n < 3) break;
            uint16_t count = *reinterpret_cast<uint16_t *>(buf + 1);
            uint8_t *cursor = buf + 3;
            double t = nowSeconds();

            std::vector<NetEntityUpdate> local;
            local.reserve(count);

            for (uint16_t i = 0; i < count && (cursor - buf) + 32 <= n; ++i)
            {
                uint32_t id = *reinterpret_cast<uint32_t *>(cursor);
                cursor += 4;
                Vec3 pos{
                    *reinterpret_cast<float *>(cursor + 0),
                    *reinterpret_cast<float *>(cursor + 4),
                    *reinterpret_cast<float *>(cursor + 8)
                };
                cursor += 12;
                Vec3 size{
                    *reinterpret_cast<float *>(cursor + 0),
                    *reinterpret_cast<float *>(cursor + 4),
                    *reinterpret_cast<float *>(cursor + 8)
                };
                cursor += 12;
                int32_t health = *reinterpret_cast<int32_t *>(cursor);
                cursor += 4;

                local.push_back({id, pos, size, health, t});
            }

            if (!local.empty())
            {
                std::lock_guard<std::mutex> lock(state.pendingMutex);
                state.pendingUpdates.insert(state.pendingUpdates.end(), local.begin(), local.end());
            }
            break;
        }
        default:
            break;
        }
    }
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

    uint8_t pkt[17]; // 1B type + 4B id + 12B direction
    pkt[0] = MSG_INPUT;
    *reinterpret_cast<uint32_t *>(pkt + 1) = state.myEntityId;
    *reinterpret_cast<float *>(pkt + 5)  = dir.x;
    *reinterpret_cast<float *>(pkt + 9)  = dir.y;
    *reinterpret_cast<float *>(pkt + 13) = dir.z;

    sendto(state.sock, pkt, 17, 0,
           reinterpret_cast<sockaddr *>(&state.serverAddr), sizeof(state.serverAddr));
}

// ─── World Sync ───────────────────────────────────────────────

static void applyNetworkUpdates()
{
    std::vector<NetEntityUpdate> updates;
    {
        std::lock_guard<std::mutex> lock(state.pendingMutex);
        std::swap(updates, state.pendingUpdates);
    }

    if (updates.empty())
        return;

    uint32_t writeIdx = state.world.getWriteIdx();
    uint32_t readIdx = state.world.getReadIdx();

    for (const auto &u : updates)
    {
        Vec3 vel{0.f, 0.f, 0.f};
        auto &hist = state.history[u.id];
        if (hist.valid)
        {
            double dt = u.time - hist.time;
            if (dt > 0.0001)
                vel = (u.pos - hist.pos) * static_cast<float>(1.0 / dt);
        }
        hist.pos = u.pos;
        hist.time = u.time;
        hist.valid = true;

        int localIdx = -1;
        Partition *chunk = state.world.findEntity(u.id, localIdx);
        if (chunk && localIdx >= 0)
        {
            chunk->setPosition(static_cast<uint32_t>(localIdx), u.pos, writeIdx);
            chunk->setPosition(static_cast<uint32_t>(localIdx), u.pos, readIdx);
            chunk->setVelocity(static_cast<uint32_t>(localIdx), vel, writeIdx);
            chunk->setVelocity(static_cast<uint32_t>(localIdx), vel, readIdx);
            chunk->setSize(static_cast<uint32_t>(localIdx), u.size);
            chunk->setHealth(static_cast<uint32_t>(localIdx), u.health);
        }
        else
        {
            Partition::EntitySnapshot snap{};
            snap.id = u.id;
            snap.position = u.pos;
            snap.rotation = {0.f, 0.f, 0.f, 1.f};
            snap.velocity = vel;
            snap.force = {0.f, 0.f, 0.f};
            snap.mass = 1.0f;
            snap.size = u.size;
            snap.health = u.health;
            state.world.addEntity(snap);
        }
    }
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
    state.sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (state.sock < 0)
    {
        perror("[ERROR] socket()");
        return false;
    }

    state.serverAddr.sin_family = AF_INET;
    state.serverAddr.sin_port = htons(serverPort);
    inet_pton(AF_INET, serverIp, &state.serverAddr.sin_addr);

    // Timeout pour recvfrom (shutdown propre)
    struct timeval tv{0, 100000}; // 100ms
    setsockopt(state.sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Envoyer MSG_CONNECT
    uint8_t pkt[1] = {MSG_CONNECT};
    sendto(state.sock, pkt, 1, 0,
           reinterpret_cast<sockaddr *>(&state.serverAddr), sizeof(state.serverAddr));

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

    // 3. Start network thread
    std::thread netThread(networkThread);

    std::cout << "[CLIENT] En attente de MSG_WELCOME du serveur...\n"
              << "  WASD    : deplacer l'entite\n"
              << "  Fleches : orienter la camera\n"
              << "  ESC     : quitter\n\n";

    // 4. Main loop
    double lastFrameTime = nowSeconds();

    while (!glfwWindowShouldClose(window) && running)
    {
        double currentTime = nowSeconds();
        double dt = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        // Sync server updates
        applyNetworkUpdates();

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

    // 5. Cleanup
    running = false;
    netThread.join();
    close(state.sock);
    glfwTerminate();
    return 0;
}

// BUILD: g++ -std=c++20 -O3 -o visual3d visual3d.cpp -lglfw -lGLEW -lGL -lm -lpthread
