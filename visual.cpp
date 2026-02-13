#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <signal.h>
#include <atomic>

#include "WorldPartition.hpp"

// ============================================================================
// ANSI Color Codes
// ============================================================================
#define RESET   "\x1b[0m"
#define BOLD    "\x1b[1m"
#define DIM     "\x1b[2m"
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define WHITE   "\x1b[37m"

#define HIDE_CURSOR     "\x1b[?25l"
#define SHOW_CURSOR     "\x1b[?25h"
#define CLEAR_SCREEN    "\x1b[2J"
#define MOVE_HOME       "\x1b[H"
#define ALT_BUFFER_ON   "\x1b[?1049h"
#define ALT_BUFFER_OFF  "\x1b[?1049l"

// ============================================================================
// Global Signal Handler
// ============================================================================
static std::atomic<bool> shouldQuit(false);

void signalHandler(int sig) {
    (void)sig;
    shouldQuit = true;
}

// ============================================================================
// Terminal Utils
// ============================================================================

struct TermSize {
    int width = 80;
    int height = 24;

    static TermSize get() {
        TermSize size;
        struct winsize w;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
            size.width = w.ws_col;
            size.height = w.ws_row;
        }
        return size;
    }
};

void moveCursor(int x, int y) {
    printf("\x1b[%d;%dH", y, x);
}

void setColor(const char* color) {
    printf("%s", color);
}

// ============================================================================
// Visualization
// ============================================================================

class WorldVisualizer {
private:
    WorldPartition world;
    std::vector<uint32_t> entityIds;
    int numEntities = 0;

    struct Stats {
        double totalTime = 0.0;
        int frameCount = 0;
        double avgFps = 0.0;
        int activeChunks = 0;
        int totalMigrations = 0;
    } stats;

    bool paused = false;
    float timeScale = 1.0f;
    float dt = 0.016f;

    std::chrono::high_resolution_clock::time_point lastTime =
        std::chrono::high_resolution_clock::now();

public:
    WorldVisualizer(int entities = 5000) : numEntities(entities) {
        // Generate random entities
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> posXDist(-2048.0f, 2048.0f);
        std::uniform_real_distribution<float> posYDist(0.0f, 50.0f);
        std::uniform_real_distribution<float> posZDist(-2048.0f, 2048.0f);
        std::uniform_real_distribution<float> velDist(-30.0f, 30.0f);

        for (int i = 0; i < numEntities; ++i) {
            Partition::EntitySnapshot entity;
            entity.id = i + 1;
            entity.position = {posXDist(rng), posYDist(rng), posZDist(rng)};
            entity.rotation = Quat::identity();
            entity.velocity = {velDist(rng), 0.0f, velDist(rng)};
            entity.mass = 1.0f;
            entity.force = {0.0f, 0.0f, 0.0f};
            entity.size = {1.0f, 2.0f, 1.0f};

            world.addEntity(entity);
            entityIds.push_back(entity.id);
        }

        lastTime = std::chrono::high_resolution_clock::now();
    }

    void update() {
        if (!paused) {
            world.step(dt * timeScale);

            auto currentTime = std::chrono::high_resolution_clock::now();
            double deltaTime = std::chrono::duration<double>(currentTime - lastTime).count();
            lastTime = currentTime;

            stats.totalTime += deltaTime;
            stats.frameCount++;
        }

        // Update stats
        if (stats.frameCount > 0) {
            stats.avgFps = stats.frameCount / stats.totalTime;
        }

        stats.activeChunks = 0;
        world.forEachChunk([this](const Partition& chunk) {
            (void)chunk;
            stats.activeChunks++;
        });
    }

    void render() {
        printf("%s", MOVE_HOME);

        // Header
        setColor(BOLD);
        setColor(CYAN);
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║  WorldPartition v7.0 - Interactive Visualization           ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n");
        setColor(RESET);

        printf("\n");

        // Stats Panel
        setColor(BOLD);
        setColor(GREEN);
        printf("▶ STATS\n");
        setColor(RESET);

        printf("  Entities:        %6d / %d\n", std::min(numEntities, 9999), numEntities);
        printf("  Active Chunks:   %6d\n", stats.activeChunks);
        printf("  Avg FPS:         %6.1f\n", stats.avgFps);
        printf("  Frame Count:     %6d\n", stats.frameCount);
        printf("  Time Scale:      ");
        setColor(YELLOW);
        printf("%.1fx\n", timeScale);
        setColor(RESET);

        printf("\n");

        // Chunk Density Visualization (simple ASCII)
        setColor(BOLD);
        setColor(GREEN);
        printf("▶ CHUNK DENSITY (8x8 grid, each cell = ~512x512 units)\n");
        setColor(RESET);

        // Count entities per chunk region
        const int GRID_SIZE = 8;
        int density[GRID_SIZE][GRID_SIZE] = {};

        for (uint32_t entityId : entityIds) {
            uint64_t chunkKey = world.getEntityChunkKey(entityId);
            if (chunkKey == std::numeric_limits<uint64_t>::max()) continue;

            Partition* chunk = world.getChunk(chunkKey);
            if (!chunk) continue;

            int idx = chunk->findEntityIndex(entityId);
            if (idx == -1) continue;

            auto entity = chunk->getEntity(static_cast<size_t>(idx));

            // Map world coordinates to grid
            int gridX = (int)((entity.position.x + 2048.0f) / 512.0f) % GRID_SIZE;
            int gridZ = (int)((entity.position.z + 2048.0f) / 512.0f) % GRID_SIZE;

            if (gridX >= 0 && gridX < GRID_SIZE && gridZ >= 0 && gridZ < GRID_SIZE) {
                density[gridZ][gridX]++;
            }
        }

        // Print grid with colors
        printf("  ");
        for (int x = 0; x < GRID_SIZE; ++x) {
            printf("+----");
        }
        printf("+\n");

        for (int z = 0; z < GRID_SIZE; ++z) {
            printf("  |");
            for (int x = 0; x < GRID_SIZE; ++x) {
                int count = density[z][x];

                if (count == 0) {
                    printf(" · |");
                } else if (count < 5) {
                    setColor(GREEN);
                    printf(" %d |", count);
                    setColor(RESET);
                } else if (count < 15) {
                    setColor(YELLOW);
                    printf(" %d |", count);
                    setColor(RESET);
                } else {
                    setColor(RED);
                    printf(" %d |", std::min(count, 99));
                    setColor(RESET);
                }
            }
            printf("\n");
            printf("  ");
            for (int x = 0; x < GRID_SIZE; ++x) {
                printf("+----");
            }
            printf("+\n");
        }

        printf("\n");

        // Status
        setColor(BOLD);
        setColor(BLUE);
        printf("▶ STATUS\n");
        setColor(RESET);

        if (paused) {
            setColor(RED);
            printf("  [PAUSED]");
            setColor(RESET);
        } else {
            setColor(GREEN);
            printf("  [RUNNING]");
            setColor(RESET);
        }
        printf(" | Press CTRL+C to quit, SPACE to pause, +/- for speed\n");

        printf("\n");

        // Sample entities
        setColor(BOLD);
        setColor(MAGENTA);
        printf("▶ SAMPLE ENTITIES (first 8)\n");
        setColor(RESET);

        for (int i = 0; i < std::min(8, (int)entityIds.size()); ++i) {
            uint32_t entityId = entityIds[i];
            uint64_t chunkKey = world.getEntityChunkKey(entityId);

            if (chunkKey == std::numeric_limits<uint64_t>::max()) continue;

            Partition* chunk = world.getChunk(chunkKey);
            if (!chunk) continue;

            int idx = chunk->findEntityIndex(entityId);
            if (idx == -1) continue;

            auto entity = chunk->getEntity(static_cast<size_t>(idx));
            printf("  Entity #%4d: pos=(%.0f, %.0f, %.0f) vel=(%.1f, %.1f, %.1f)\n",
                   entityId,
                   entity.position.x, entity.position.y, entity.position.z,
                   entity.velocity.x, entity.velocity.y, entity.velocity.z);
        }

        fflush(stdout);
    }

    void handleInput() {
        // Non-blocking input (simplified - doesn't work well in all terminals)
        // For proper implementation, would need to use termios/raw mode
    }

    void run() {
        // Enter alternate screen buffer (no scrolling)
        printf("%s", ALT_BUFFER_ON);
        fflush(stdout);
        printf("%s", HIDE_CURSOR);
        fflush(stdout);

        int frameCount = 0;

        while (frameCount < 6000 && !shouldQuit) {  // Up to 100 seconds or until Ctrl+C
            update();
            render();

            frameCount++;

            // Cap to ~60 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }

        // Exit alternate screen buffer
        printf("%s", ALT_BUFFER_OFF);
        printf("%s", SHOW_CURSOR);
        printf("%s", RESET);
        printf("\nSimulation ended after %d frames.\n", frameCount);
        fflush(stdout);
    }
};

int main() {
    // Setup signal handler for graceful shutdown
    signal(SIGINT, signalHandler);

    std::cout << "Initializing WorldPartition Visualizer...\n";
    std::cout << "Press Ctrl+C to exit at any time.\n\n";

    WorldVisualizer viz(500000);
    viz.run();

    return 0;
}

// BUILD: g++ -std=c++20 -O3 -o visual visual.cpp -lpthread
