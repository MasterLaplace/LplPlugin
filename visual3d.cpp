#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <random>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "WorldPartition.hpp"

const float CHUNK_SIZE = 256.0f;
const float WORLD_BIAS = 2048.0f;

struct Camera {
    glm::vec3 position = {0.0f, 500.0f, -500.0f};
    glm::vec3 front = {0.0f, -0.3f, 1.0f};
    glm::vec3 up = {0.0f, 1.0f, 0.0f};
    float speed = 300.0f;
    float sensitivity = 0.003f;
    float yaw = 90.0f;
    float pitch = -15.0f;
};

struct State {
    WorldPartition world;
    std::vector<uint32_t> entityIds;
    Camera camera;
    bool paused = false;
    float timeScale = 1.0f;
    double lastTime = 0.0;
    int frameCount = 0;
    double avgFps = 0.0;
} state;

bool keys[512] = {};
double mouseX = 0, mouseY = 0;
bool firstMouse = true;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)window; (void)scancode; (void)mods;
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        if (key < 512) keys[key] = true;

        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
        if (key == GLFW_KEY_SPACE) state.paused = !state.paused;
        if (key == GLFW_KEY_UP) state.timeScale = std::min(4.0f, state.timeScale + 0.5f);
        if (key == GLFW_KEY_DOWN) state.timeScale = std::max(0.1f, state.timeScale - 0.5f);
    }
    else if (action == GLFW_RELEASE)
    {
        if (key < 512) keys[key] = false;
    }
}

void updateCameraDirection()
{
    glm::vec3 front;
    front.x = cos(glm::radians(state.camera.yaw)) * cos(glm::radians(state.camera.pitch));
    front.y = sin(glm::radians(state.camera.pitch));
    front.z = sin(glm::radians(state.camera.yaw)) * cos(glm::radians(state.camera.pitch));
    state.camera.front = glm::normalize(front);
}

void drawBox(const glm::vec3 &min, const glm::vec3 &max, const glm::vec3 &color)
{
    glColor3f(color.x, color.y, color.z);

    glBegin(GL_LINES);
    // bottom square
    glVertex3f(min.x, min.y, min.z); glVertex3f(max.x, min.y, min.z);
    glVertex3f(max.x, min.y, min.z); glVertex3f(max.x, min.y, max.z);
    glVertex3f(max.x, min.y, max.z); glVertex3f(min.x, min.y, max.z);
    glVertex3f(min.x, min.y, max.z); glVertex3f(min.x, min.y, min.z);

    // top square
    glVertex3f(min.x, max.y, min.z); glVertex3f(max.x, max.y, min.z);
    glVertex3f(max.x, max.y, min.z); glVertex3f(max.x, max.y, max.z);
    glVertex3f(max.x, max.y, max.z); glVertex3f(min.x, max.y, max.z);
    glVertex3f(min.x, max.y, max.z); glVertex3f(min.x, max.y, min.z);

    // vertical edges
    glVertex3f(min.x, min.y, min.z); glVertex3f(min.x, max.y, min.z);
    glVertex3f(max.x, min.y, min.z); glVertex3f(max.x, max.y, min.z);
    glVertex3f(max.x, min.y, max.z); glVertex3f(max.x, max.y, max.z);
    glVertex3f(min.x, min.y, max.z); glVertex3f(min.x, max.y, max.z);
    glEnd();
}

void drawCube(const glm::vec3 &center, float size, const glm::vec3 &color)
{
    float half = size * 0.5f;
    glColor3f(color.x, color.y, color.z);
    glPushMatrix();
    glTranslatef(center.x, center.y, center.z);

    glBegin(GL_QUADS);
    // Front
    glVertex3f(-half, -half,  half);
    glVertex3f( half, -half,  half);
    glVertex3f( half,  half,  half);
    glVertex3f(-half,  half,  half);

    // Back
    glVertex3f(-half, -half, -half);
    glVertex3f(-half,  half, -half);
    glVertex3f( half,  half, -half);
    glVertex3f( half, -half, -half);

    // Left
    glVertex3f(-half, -half, -half);
    glVertex3f(-half, -half,  half);
    glVertex3f(-half,  half,  half);
    glVertex3f(-half,  half, -half);

    // Right
    glVertex3f( half, -half, -half);
    glVertex3f( half,  half, -half);
    glVertex3f( half,  half,  half);
    glVertex3f( half, -half,  half);

    // Top
    glVertex3f(-half,  half, -half);
    glVertex3f(-half,  half,  half);
    glVertex3f( half,  half,  half);
    glVertex3f( half,  half, -half);

    // Bottom
    glVertex3f(-half, -half, -half);
    glVertex3f( half, -half, -half);
    glVertex3f( half, -half,  half);
    glVertex3f(-half, -half,  half);
    glEnd();

    glPopMatrix();
}

GLFWwindow *initWindow()
{
    if (!glfwInit())
    {
        std::cerr << "GLFW initialization failed\n";
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow *window = glfwCreateWindow(1440, 900, "WorldPartition v7.0 - 3D Visualization", nullptr, nullptr);
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
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    return window;
}

void initWorld()
{
    std::cout << "Initializing 5000 entities...\n";
    std::cout << "[ ] World initialized.\r" << std::flush;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posXDist(-2000.0f, 2000.0f);
    std::uniform_real_distribution<float> posYDist(0.0f, 100.0f);
    std::uniform_real_distribution<float> posZDist(-2000.0f, 2000.0f);
    std::uniform_real_distribution<float> velDist(-30.0f, 30.0f);
    std::uniform_real_distribution<float> sizeDist(2.0f, 10.0f);

    for (uint32_t i = 0u; i < 5000u; ++i)
    {
        Partition::EntitySnapshot entity;
        entity.id = i + 1u;
        entity.position = {posXDist(rng), posYDist(rng), posZDist(rng)};
        entity.rotation = Quat::identity();
        entity.velocity = {velDist(rng), 0.0f, velDist(rng)};
        entity.mass = 1.0f;
        entity.force = {0.0f, 0.0f, 0.0f};
        float s = sizeDist(rng);
        entity.size = {s, s * 1.2f, s};

        state.world.addEntity(entity);
        state.entityIds.push_back(entity.id);
    }

    std::cout << "[x] World initialized. Controls:\n"
              << " - WASD: Move\n"
              << " - Arrow Keys: Look around\n"
              << " - SPACE: Pause/Resume\n"
              << " - ESC: Exit\n\n";
}

void updateInputs(double deltaTime)
{
    float rotationSpeed = 60.0f;
    if (keys[GLFW_KEY_LEFT])
    {
        state.camera.yaw -= rotationSpeed * deltaTime;
        updateCameraDirection();
    }
    if (keys[GLFW_KEY_RIGHT])
    {
        state.camera.yaw += rotationSpeed * deltaTime;
        updateCameraDirection();
    }
    if (keys[GLFW_KEY_UP])
    {
        state.camera.pitch += rotationSpeed * deltaTime;
        if (state.camera.pitch > 89.0f) state.camera.pitch = 89.0f;
        updateCameraDirection();
    }
    if (keys[GLFW_KEY_DOWN])
    {
        state.camera.pitch -= rotationSpeed * deltaTime;
        if (state.camera.pitch < -89.0f) state.camera.pitch = -89.0f;
        updateCameraDirection();
    }

    glm::vec3 movement(0.0f);
    if (keys[GLFW_KEY_W]) movement += state.camera.front;
    if (keys[GLFW_KEY_S]) movement -= state.camera.front;
    if (keys[GLFW_KEY_A]) movement -= glm::normalize(glm::cross(state.camera.front, state.camera.up));
    if (keys[GLFW_KEY_D]) movement += glm::normalize(glm::cross(state.camera.front, state.camera.up));

    if (glm::length(movement) > 0.1f)
        state.camera.position += glm::normalize(movement) * (float)(state.camera.speed * deltaTime);
}

void rendering(GLFWwindow *window)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)width / height, 0.1f, 10000.0f);
    glLoadMatrixf(glm::value_ptr(proj));

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glm::vec3 target = state.camera.position + state.camera.front;
    glm::mat4 view = glm::lookAt(state.camera.position, target, state.camera.up);
    glLoadMatrixf(glm::value_ptr(view));
}

void drawChunkGrid()
{
    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_LINES);
    for (int x = -4000; x <= 4000; x += 500)
    {
        glVertex3f(x, -100.0f, -4000.0f);
        glVertex3f(x, -100.0f, 4000.0f);
    }
    for (int z = -4000; z <= 4000; z += 500)
    {
        glVertex3f(-4000.0f, -100.0f, z);
        glVertex3f(4000.0f, -100.0f, z);
    }
    glEnd();

    glColor3f(0.2f, 0.2f, 0.5f);
    glBegin(GL_LINES);
    for (int cx = -16; cx <= 16; ++cx)
    {
        for (int cz = -16; cz <= 16; ++cz)
        {
            float minX = cx * CHUNK_SIZE - WORLD_BIAS;
            float maxX = minX + CHUNK_SIZE;
            float minZ = cz * CHUNK_SIZE - WORLD_BIAS;
            float maxZ = minZ + CHUNK_SIZE;

            glVertex3f(minX, 0.0f, minZ); glVertex3f(maxX, 0.0f, minZ);
            glVertex3f(maxX, 0.0f, minZ); glVertex3f(maxX, 0.0f, maxZ);
            glVertex3f(maxX, 0.0f, maxZ); glVertex3f(minX, 0.0f, maxZ);
            glVertex3f(minX, 0.0f, maxZ); glVertex3f(minX, 0.0f, minZ);
        }
    }
    glEnd();
}

void drawEntities()
{
    for (uint32_t entityId : state.entityIds)
    {
        uint64_t chunkKey = state.world.getEntityChunkKey(entityId);
        if (chunkKey == std::numeric_limits<uint64_t>::max()) continue;

        Partition* chunk = state.world.getChunk(chunkKey);
        if (!chunk) continue;

        int idx = chunk->findEntityIndex(entityId);
        if (idx == -1) continue;

        auto entity = chunk->getEntity(static_cast<size_t>(idx));

        // Color based on position for visual variety
        float h = entity.position.x / 100.0f;
        float w = entity.position.z / 100.0f;
        glm::vec3 color(h * 0.5f + 0.2f, 0.6f - h * 0.3f, 0.8f - w * 0.2f);
        glm::vec3 pos(entity.position.x, entity.position.y, entity.position.z);
        float size = (entity.size.x + entity.size.y + entity.size.z) / 3.0f;

        drawCube(pos, size, color);
    }
}

int main()
{
    GLFWwindow *window = initWindow();
    if (!window)
        return 1;

    initWorld();

    auto lastFrameTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        if (!state.paused)
            state.world.step(0.016f * state.timeScale);

        updateInputs(deltaTime);

        state.frameCount++;
        if (state.frameCount % 60 == 0)
            state.avgFps = 60.0 / (glfwGetTime() - lastFrameTime + 0.0001);

        rendering(window);
        drawChunkGrid();
        drawEntities();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

// BUILD: g++ -std=c++20 -O3 -o visual3d visual3d.cpp -lglfw -lGLEW -lGL -lm -lpthread
