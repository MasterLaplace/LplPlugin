// ═══════════════════════════════════════════════════════════════════════════════
//  LplPlugin — Research-Grade Benchmark Suite
//  Measures core engine subsystem performance for academic publication.
//
//  Benchmarks:
//    1. Morton Code Encoding (Spatial Locality)
//    2. FlatAtomicsHashMap (Lock-Free Throughput)
//    3. Ring Buffer (Producer-Consumer IPC)
//    4. Physics Scalability (Entity Count Sweep: 1K → 100K)
//    5. Memory Layout (SoA vs AoS cache efficiency)
//
//  BUILD: make benchmark_research
//  USAGE: ./benchmark_research [iterations=1000000]
// ═══════════════════════════════════════════════════════════════════════════════

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "FlatAtomicsHashMap.hpp"
#include "Morton.hpp"
#include "WorldPartition.hpp"
#include "lpl_protocol.h"

// ─── Timing Utilities ────────────────────────────────────────────────────────

struct BenchResult {
    std::string name;
    double totalNs;
    size_t iterations;

    double nsPerOp() const { return totalNs / iterations; }
    double msTotal() const { return totalNs / 1e6; }
    double opsPerSec() const { return iterations / (totalNs / 1e9); }
};

class ScopedTimer {
public:
    ScopedTimer() : _start(std::chrono::high_resolution_clock::now()) {}
    double elapsedNs() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::nano>(end - _start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point _start;
};

static void printResult(const BenchResult &r)
{
    std::cout << "  │ " << std::left << std::setw(36) << r.name << " │ " << std::right << std::setw(10)
              << std::fixed << std::setprecision(1) << r.nsPerOp() << " ns/op │ " << std::setw(12)
              << std::setprecision(0) << r.opsPerSec() / 1e6 << " Mops/s │" << std::endl;
}

static void printSectionHeader(const std::string &title)
{
    std::cout << "  ┌──────────────────────────────────────┬────────────┬──────────────┐\n";
    std::cout << "  │ " << std::left << std::setw(73) << title << "│\n";
    std::cout << "  ├──────────────────────────────────────┼────────────┼──────────────┤\n";
}

static void printSectionFooter()
{
    std::cout << "  └──────────────────────────────────────┴────────────┴──────────────┘\n\n";
}

// ─── Benchmark 1: Morton Code Encoding ───────────────────────────────────────

BenchResult benchMortonEncode2D(size_t iters)
{
    ScopedTimer t;
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < iters; ++i)
    {
        const uint64_t bias = 1ULL << 20;
        uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(i % 10000) - 5000) + bias;
        uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>((i * 7) % 10000) - 5000) + bias;
        sink = Morton::encode2D(ux, uz);
    }
    (void)sink;
    return {"Morton::encode2D", t.elapsedNs(), iters};
}

BenchResult benchMortonEncode3D(size_t iters)
{
    ScopedTimer t;
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < iters; ++i)
    {
        uint64_t x = i % 1024;
        uint64_t y = (i * 3) % 1024;
        uint64_t z = (i * 7) % 1024;
        sink = Morton::encode3D(x, y, z);
    }
    (void)sink;
    return {"Morton::encode3D", t.elapsedNs(), iters};
}

// ─── Benchmark 2: FlatAtomicsHashMap ─────────────────────────────────────────

BenchResult benchHashMapInsert(size_t iters)
{
    size_t cap = std::min(iters, static_cast<size_t>(65536));
    FlatAtomicsHashMap<uint64_t> map(cap);

    ScopedTimer t;
    for (size_t i = 0; i < cap; ++i)
    {
        uint64_t key = Morton::encode2D(i % 256, i / 256);
        map.insert(key, static_cast<uint64_t>(i));
    }
    return {"FlatAtomicsHashMap::insert", t.elapsedNs(), cap};
}

BenchResult benchHashMapGet(size_t iters)
{
    size_t cap = std::min(iters, static_cast<size_t>(65536));
    FlatAtomicsHashMap<uint64_t> map(cap);

    // Pre-fill
    std::vector<uint64_t> keys;
    for (size_t i = 0; i < cap; ++i)
    {
        uint64_t key = Morton::encode2D(i % 256, i / 256);
        map.insert(key, static_cast<uint64_t>(i));
        keys.push_back(key);
    }

    ScopedTimer t;
    volatile uint64_t* sink = nullptr;
    for (size_t i = 0; i < cap; ++i)
    {
        sink = map.get(keys[i]);
    }
    (void)sink;
    return {"FlatAtomicsHashMap::get", t.elapsedNs(), cap};
}

// ─── Benchmark 3: Ring Buffer Throughput ─────────────────────────────────────

BenchResult benchRingBufferThroughput(size_t iters)
{
    // Simulates the kernel→userspace ring buffer pattern
    RingHeader ring;
    ring.head = 0;
    ring.tail = 0;
    constexpr uint32_t RING_SIZE = 4096;

    ScopedTimer t;
    for (size_t i = 0; i < iters; ++i)
    {
        // Producer: write
        uint32_t next = (ring.head + 1) & (RING_SIZE - 1);
        if (next != ring.tail)
        {
            smp_store_release(&ring.head, next);
        }
        // Consumer: read
        uint32_t head = smp_load_acquire(&ring.head);
        if (head != ring.tail)
        {
            smp_store_release(&ring.tail, (ring.tail + 1) & (RING_SIZE - 1));
        }
    }
    return {"RingBuffer (SPSC atomic)", t.elapsedNs(), iters};
}

// ─── Benchmark 4: Physics Scalability ────────────────────────────────────────

struct ScalabilityResult {
    int entityCount;
    double avgFrameMs;
    double fps;
};

static ScalabilityResult benchPhysicsScale(int numEntities, int numFrames = 50)
{
    WorldPartition world;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posDist(-2000.0f, 2000.0f);
    std::uniform_real_distribution<float> velDist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> massDist(0.5f, 5.0f);

    for (int i = 0; i < numEntities; ++i)
    {
        Partition::EntitySnapshot e;
        e.id = i + 1;
        e.position = {posDist(rng), std::abs(posDist(rng)) * 0.05f, posDist(rng)};
        e.rotation = Quat::identity();
        e.velocity = {velDist(rng), 0.0f, velDist(rng)};
        e.mass = massDist(rng);
        e.force = {0, 0, 0};
        e.size = {1, 2, 1};
        world.addEntity(e);
    }

    // Warm-up
    for (int i = 0; i < 3; ++i)
        world.step(0.016f);

    // Measure
    double totalMs = 0.0;
    for (int f = 0; f < numFrames; ++f)
    {
        ScopedTimer t;
        world.step(0.016f);
        totalMs += t.elapsedNs() / 1e6;
    }

    double avgMs = totalMs / numFrames;
    return {numEntities, avgMs, 1000.0 / avgMs};
}

// ─── Benchmark 5: SoA vs AoS Cache Efficiency ───────────────────────────────

// AoS baseline for comparison
struct EntityAoS {
    float px, py, pz;
    float vx, vy, vz;
    float fx, fy, fz;
    float mass;
};

BenchResult benchAoSPhysics(size_t count)
{
    std::vector<EntityAoS> entities(count);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-10.f, 10.f);
    for (auto &e : entities)
    {
        e.px = d(rng); e.py = d(rng); e.pz = d(rng);
        e.vx = d(rng); e.vy = 0; e.vz = d(rng);
        e.mass = 1.0f;
        e.fx = 0; e.fy = -9.81f; e.fz = 0;
    }

    float dt = 0.016f;
    ScopedTimer t;
    for (size_t i = 0; i < count; ++i)
    {
        auto &e = entities[i];
        if (e.mass > 0.0001f)
        {
            float invM = 1.0f / e.mass;
            e.vx += e.fx * invM * dt;
            e.vy += e.fy * invM * dt;
            e.vz += e.fz * invM * dt;
        }
        e.px += e.vx * dt;
        e.py += e.vy * dt;
        e.pz += e.vz * dt;
    }
    return {"AoS Physics (baseline)", t.elapsedNs(), count};
}

BenchResult benchSoAPhysics(size_t count)
{
    std::vector<float> px(count), py(count), pz(count);
    std::vector<float> vx(count), vy(count), vz(count);
    std::vector<float> fx(count), fy(count), fz(count);
    std::vector<float> mass(count);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-10.f, 10.f);
    for (size_t i = 0; i < count; ++i)
    {
        px[i] = d(rng); py[i] = d(rng); pz[i] = d(rng);
        vx[i] = d(rng); vy[i] = 0; vz[i] = d(rng);
        mass[i] = 1.0f;
        fx[i] = 0; fy[i] = -9.81f; fz[i] = 0;
    }

    float dt = 0.016f;
    ScopedTimer t;
    for (size_t i = 0; i < count; ++i)
    {
        if (mass[i] > 0.0001f)
        {
            float invM = 1.0f / mass[i];
            vx[i] += fx[i] * invM * dt;
            vy[i] += fy[i] * invM * dt;
            vz[i] += fz[i] * invM * dt;
        }
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;
    }
    return {"SoA Physics (engine layout)", t.elapsedNs(), count};
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char **argv)
{
    size_t iters = 1000000;
    if (argc > 1) iters = std::stoull(argv[1]);

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║         LplPlugin Research Benchmark Suite                              ║\n"
              << "║         Modular FullDive Deterministic Engine — Performance Audit        ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║  Iterations: " << std::left << std::setw(58) << iters << "║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // ── Section 1: Spatial Encoding ──
    printSectionHeader("§1  Spatial Encoding (Morton / Z-order)");
    printResult(benchMortonEncode2D(iters));
    printResult(benchMortonEncode3D(iters));
    printSectionFooter();

    // ── Section 2: Lock-Free Data Structures ──
    printSectionHeader("§2  Lock-Free Data Structures");
    printResult(benchHashMapInsert(iters));
    printResult(benchHashMapGet(iters));
    printResult(benchRingBufferThroughput(iters));
    printSectionFooter();

    // ── Section 3: Memory Layout (SoA vs AoS) ──
    size_t layoutCount = std::min(iters, static_cast<size_t>(100000));
    printSectionHeader("§3  Memory Layout (SoA vs AoS)");
    auto aosResult = benchAoSPhysics(layoutCount);
    auto soaResult = benchSoAPhysics(layoutCount);
    printResult(aosResult);
    printResult(soaResult);
    double speedup = aosResult.nsPerOp() / soaResult.nsPerOp();
    std::cout << "  │ SoA Speedup Factor                  │ " << std::right << std::setw(9) << std::fixed
              << std::setprecision(2) << speedup << "x │              │\n";
    printSectionFooter();

    // ── Section 4: Physics Scalability Sweep ──
    std::cout << "  ┌──────────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │ §4  Physics Scalability Sweep (CPU, 50 frames each)             │\n";
    std::cout << "  ├──────────────┬──────────────┬──────────────┬────────────────────┤\n";
    std::cout << "  │   Entities   │  Avg Frame   │     FPS      │      Rating        │\n";
    std::cout << "  ├──────────────┼──────────────┼──────────────┼────────────────────┤\n";

    std::vector<int> entityCounts = {1000, 5000, 10000, 25000, 50000};
    // Only include 100K if explicitly requested (slow)
    if (iters >= 10000000) entityCounts.push_back(100000);

    for (int n : entityCounts)
    {
        auto r = benchPhysicsScale(n);
        const char *rating = r.fps >= 60.0 ? "[OK] REAL-TIME" : (r.fps >= 30.0 ? "[OK] PLAYABLE" : "[KO] TOO SLOW");
        std::cout << "  │ " << std::right << std::setw(10) << r.entityCount << "   │ " << std::setw(9)
                  << std::fixed << std::setprecision(2) << r.avgFrameMs << " ms │ " << std::setw(10)
                  << std::setprecision(1) << r.fps << " Hz │ " << std::left << std::setw(19) << rating
                  << "│\n";
    }

    std::cout << "  └──────────────┴──────────────┴────────────────┴───────────────────┘\n\n";

    std::cout << "══════════════════════════════════════════════════════════════════════════\n"
              << "  Benchmark complete. Results are deterministic (fixed RNG seeds).\n"
              << "  Run with higher iteration count for publication-grade measurements.\n"
              << "══════════════════════════════════════════════════════════════════════════\n\n";

    return 0;
}

// BUILD: make benchmark_research
