#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "WorldPartition.hpp"

std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << "(" << std::fixed << std::setprecision(1) << v.x << ", " << v.y << ", " << v.z << ")";
}

struct BenchmarkStats {
    int totalEntities = 0;
    int totalFrames = 0;
    int totalMigrations = 0;
    int totalChunksCreated = 0;

    double totalTime = 0.0;      // Total elapsed time (sec)
    double frameTimes[3] = {0};  // Min, avg, max frame time (ms)
    double lookupTime = 0.0;     // Total EntityRegistry lookup time (µs)
    double lookups = 0;
    double partitionLookupTime = 0.0;  // Total Partition::findEntityIndex time (ns)
    double partitionLookups = 0;
};

void recordFrameTime(double elapsed_ms, double& min_t, double& avg_t, double& max_t, int frame_count) {
    if (frame_count == 1) {
        min_t = elapsed_ms;
        avg_t = elapsed_ms;
        max_t = elapsed_ms;
    } else {
        min_t = std::min(min_t, elapsed_ms);
        max_t = std::max(max_t, elapsed_ms);
        avg_t = (avg_t * (frame_count - 1) + elapsed_ms) / frame_count;
    }
}

int main(int argc, char** argv) {
    // Parse arguments
    int numEntities = 10000;  // Default
    int numFrames = 100;

    if (argc > 1) numEntities = std::atoi(argv[1]);
    if (argc > 2) numFrames = std::atoi(argv[2]);

    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         WorldPartition CPU Benchmark Suite v1.0             ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Config: " << std::setw(8) << numEntities << " entities, "
              << std::setw(5) << numFrames << " frames" << std::string(30, ' ') << "║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;

    BenchmarkStats stats;
    stats.totalEntities = numEntities;
    stats.totalFrames = numFrames;

    // Initialize RNG
    std::mt19937 rng(12345);  // Deterministic seed for reproducibility
    std::uniform_real_distribution<float> posXDist(-5000.0f, 5000.0f);
    std::uniform_real_distribution<float> posYDist(0.0f, 100.0f);
    std::uniform_real_distribution<float> posZDist(-5000.0f, 5000.0f);
    std::uniform_real_distribution<float> velDist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> massDist(0.5f, 5.0f);

    // Step 1: Create world and entities
    WorldPartition world;
    std::cout << "[1/3] Creating " << numEntities << " entities..." << std::flush;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> entityIds;

    for (int i = 0; i < numEntities; ++i) {
        Partition::EntitySnapshot entity;
        entity.id = i + 1;
        entity.position = {posXDist(rng), posYDist(rng), posZDist(rng)};
        entity.rotation = Quat::identity();
        entity.velocity = {velDist(rng), 0.0f, velDist(rng)};
        entity.mass = massDist(rng);
        entity.force = {0.0f, 0.0f, 0.0f};
        entity.size = {1.0f, 2.0f, 1.0f};

        world.addEntity(entity);
        entityIds.push_back(entity.id);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double creationTime = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " ✓ (" << std::fixed << std::setprecision(3) << creationTime << "s)" << std::endl;

    // Step 2: Warm-up phase
    std::cout << "[2/3] Warm-up phase (5 frames)..." << std::flush;
    float dt = 0.016f;  // ~60 FPS
    for (int i = 0; i < 5; ++i) {
        world.step(dt);
    }
    std::cout << " ✓" << std::endl;

    // Step 3: Benchmark main loop
    std::cout << "[3/3] Running " << numFrames << " benchmark frames..." << std::endl << std::endl;

    double min_frame_ms = 1e10, avg_frame_ms = 0.0, max_frame_ms = 0.0;

    auto benchmarkStart = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < numFrames; ++frame) {
        // Measure frame time
        auto frameStart = std::chrono::high_resolution_clock::now();

        // Physics step
        world.step(dt);

        auto frameEnd = std::chrono::high_resolution_clock::now();
        double frameElapsed = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        recordFrameTime(frameElapsed, min_frame_ms, avg_frame_ms, max_frame_ms, frame + 1);

        // Sample EntityRegistry lookups every 10 frames
        if (frame % 10 == 0) {
            auto lookupStart = std::chrono::high_resolution_clock::now();
            for (int lookups = 0; lookups < 100; ++lookups) {
                int idx = rng() % entityIds.size();
                uint32_t eId = entityIds[idx];
                volatile uint64_t chunkKey = world.getEntityChunkKey(eId);
                (void)chunkKey;  // Use value to prevent optimization
            }
            auto lookupEnd = std::chrono::high_resolution_clock::now();
            stats.lookupTime += std::chrono::duration<double, std::micro>(lookupEnd - lookupStart).count();
            stats.lookups += 100;

            // Sample Partition::findEntityIndex lookups
            auto partitionLookupStart = std::chrono::high_resolution_clock::now();
            for (int lookups = 0; lookups < 100; ++lookups) {
                int idx = rng() % entityIds.size();
                uint32_t eId = entityIds[idx];
                int outLocalIdx = -1;
                Partition* chunk = world.findEntity(eId, outLocalIdx);
                (void)chunk;  // Prevent optimization
                (void)outLocalIdx;
            }
            auto partitionLookupEnd = std::chrono::high_resolution_clock::now();
            stats.partitionLookupTime += std::chrono::duration<double, std::nano>(partitionLookupEnd - partitionLookupStart).count();
            stats.partitionLookups += 100;
        }

        // Progress indicator every 20 frames
        if ((frame + 1) % 20 == 0) {
            std::cout << "  ✓ Frame " << std::setw(3) << (frame + 1) << " / " << numFrames
                      << "  (current: " << std::fixed << std::setprecision(2) << frameElapsed
                      << " ms)" << std::endl;
        }
    }

    auto benchmarkEnd = std::chrono::high_resolution_clock::now();
    stats.totalTime = std::chrono::duration<double>(benchmarkEnd - benchmarkStart).count();

    // Step 4: Results
    std::cout << std::endl;
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                      BENCHMARK RESULTS                     ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;

    // Frame timing
    double entitiesesPerSec = (numEntities * numFrames) / stats.totalTime;
    std::cout << "║ Total Simulation Time       : " << std::fixed << std::setprecision(3)
              << std::setw(12) << stats.totalTime << " sec" << std::string(23, ' ') << "║" << std::endl;
    std::cout << "║ Entity Updates (throughput) : " << std::fixed << std::setprecision(1)
              << std::setw(12) << entitiesesPerSec / 1000000.0 << " M ops/sec" << std::string(14, ' ') << "║" << std::endl;
    std::cout << "║ Avg FPS                     : " << std::fixed << std::setprecision(1)
              << std::setw(12) << (1000.0 / avg_frame_ms) << " fps" << std::string(25, ' ') << "║" << std::endl;

    // EntityRegistry lookup
    if (stats.lookups > 0) {
        double avgLookupMicros = stats.lookupTime / stats.lookups;
        std::cout << "║ EntityRegistry Lookup       : " << std::fixed << std::setprecision(3)
                  << std::setw(12) << avgLookupMicros << " µs/lookup" << std::string(14, ' ') << "║" << std::endl;
    }

    // Partition::findEntityIndex lookup
    if (stats.partitionLookups > 0) {
        double avgPartitionLookupNanos = stats.partitionLookupTime / stats.partitionLookups;
        std::cout << "║ Partition::findEntityIndex  : " << std::fixed << std::setprecision(1)
                  << std::setw(12) << avgPartitionLookupNanos << " ns/lookup" << std::string(14, ' ') << "║" << std::endl;
    }

    // Chunk statistics
    int activatedChunks = 0;
    world.forEachChunk([&](const Partition& chunk) {
        (void)chunk;
        activatedChunks++;
    });

    std::cout << "║ Active Chunks Created       : " << std::fixed << std::setprecision(0)
              << std::setw(12) << activatedChunks << " chunks" << std::string(16, ' ') << "║" << std::endl;

    if (activatedChunks > 0) {
        double entitiesPerChunk = (double)numEntities / activatedChunks;
        std::cout << "║ Entities per Chunk (avg)    : " << std::fixed << std::setprecision(1)
                  << std::setw(12) << entitiesPerChunk << std::string(24, ' ') << "║" << std::endl;
    }

    std::cout << "║ Creation Overhead           : " << std::fixed << std::setprecision(3)
              << std::setw(12) << creationTime << " sec" << std::string(23, ' ') << "║" << std::endl;

    std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;

    // Summary
    double throughputMOpsSec = entitiesesPerSec / 1000000.0;
    if (throughputMOpsSec > 50.0) {
        std::cout << "║ PERFORMANCE: EXCELLENT (>50M ops/sec)                      ║" << std::endl;
    } else if (throughputMOpsSec > 20.0) {
        std::cout << "║ PERFORMANCE: GOOD (20-50M ops/sec)                        ║" << std::endl;
    } else if (throughputMOpsSec > 5.0) {
        std::cout << "║ PERFORMANCE: ACCEPTABLE (5-20M ops/sec)                   ║" << std::endl;
    } else {
        std::cout << "║ PERFORMANCE: POOR (<5M ops/sec)                           ║" << std::endl;
    }

    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}

// BUILD: g++ -std=c++20 -O3 -march=native -o benchmark benchmark.cpp -lpthread
