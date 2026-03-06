/**
 * @file main.cpp
 * @brief LplPlugin benchmark entry-point.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Types.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/core/Constants.hpp>
#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/memory/PoolAllocator.hpp>
#include <lpl/container/RingBuffer.hpp>
#include <lpl/container/FlatAtomicHashMap.hpp>
#include <lpl/container/SparseSet.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/Morton.hpp>
#include <lpl/math/Cordic.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/physics/CollisionDetector.hpp>
#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/concurrency/ThreadPool.hpp>

#include <chrono>
#include <cstdio>
#include <vector>
#include <atomic>
#include <future>

using namespace lpl;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns elapsed milliseconds for a single timed run of @p fn.
template <typename Fn>
core::f64 benchmarkMs(const char* label, Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const core::f64 ms = std::chrono::duration<core::f64, std::milli>(end - start).count();
    std::printf("  %-42s %10.3f ms\n", label, ms);
    return ms;
}

/// Converts a per-frame elapsed time to a human-readable frame-rate verdict.
static const char* frameRateVerdict(core::f64 msPerFrame)
{
    const core::f64 fps = 1000.0 / (msPerFrame > 0.0 ? msPerFrame : 1.0);
    if (fps >= 60.0) return "REAL-TIME  (>=60 fps)";
    if (fps >= 30.0) return "PLAYABLE   (>=30 fps)";
    return "TOO SLOW   (<30 fps) ";
}

void benchmarkArena()
{
    memory::ArenaAllocator arena{4 * 1024 * 1024};
    benchmarkMs("ArenaAllocator 10k allocs", [&]()
    {
        for (int i = 0; i < 10000; ++i)
        {
            [[maybe_unused]] auto* p = arena.allocate(128, 16);
        }
        arena.reset();
    });
}

void benchmarkFixedMath()
{
    benchmarkMs("Fixed32 mul 1M ops", []()
    {
        math::Fixed32 a = math::Fixed32::fromFloat(3.14159f);
        math::Fixed32 b = math::Fixed32::fromFloat(2.71828f);
        math::Fixed32 r{0};
        for (int i = 0; i < 1000000; ++i)
        {
            r = a * b;
            a = r;
            b = b + math::Fixed32{1};
        }
    });
}

void benchmarkMorton()
{
    benchmarkMs("Morton encode3D 1M", []()
    {
        for (core::u32 i = 0; i < 1000000; ++i)
        {
            [[maybe_unused]] auto code = math::morton::encode3D(
                static_cast<core::i32>(i),
                static_cast<core::i32>(i + 1),
                static_cast<core::i32>(i + 2));
        }
    });
}

void benchmarkCordic()
{
    benchmarkMs("CORDIC sin 1M", []()
    {
        for (int i = 0; i < 1000000; ++i)
        {
            [[maybe_unused]] auto s = math::Cordic::sin(
                math::Fixed32{static_cast<core::i32>(i % 65536)});
        }
    });
}

void benchmarkRegistry()
{
    ecs::Registry reg;
    ecs::Archetype arch;
    arch.add(ecs::ComponentId::Position);

    benchmarkMs("Registry create/destroy 10k entities", [&]()
    {
        for (int i = 0; i < 10000; ++i)
        {
            auto e = reg.createEntity(arch);
            if (e) { [[maybe_unused]] auto r = reg.destroyEntity(e.value()); }
        }
    });

    benchmarkMs("Registry create batch 100k entities", [&]()
    {
        for (int i = 0; i < 100000; ++i)
        {
            [[maybe_unused]] auto e = reg.createEntity(arch);
        }
    });
}

void benchmarkPhysics()
{
    ecs::Registry reg;
    ecs::WorldPartition world(math::Fixed32{10});

    ecs::Archetype arch;
    arch.add(ecs::ComponentId::Position);
    arch.add(ecs::ComponentId::AABB);

    for (int i = 0; i < 10000; ++i)
    {
        [[maybe_unused]] auto e = reg.createEntity(arch);
    }

    benchmarkMs("Physics tick (10k entities, broadphase)", [&]()
    {
        world.step(0.0166f);
    });
}

void benchmarkPhysicsScalability()
{
    constexpr core::u32 kCounts[] = {1000, 5000, 10000, 50000, 100000};

    std::printf("\n  --- Physics Scalability Sweep ---\n");

    for (core::u32 count : kCounts)
    {
        ecs::Registry reg;
        ecs::WorldPartition world(math::Fixed32{100});

        ecs::Archetype arch;
        arch.add(ecs::ComponentId::Position);
        arch.add(ecs::ComponentId::Velocity);
        arch.add(ecs::ComponentId::Mass);
        arch.add(ecs::ComponentId::AABB);

        for (core::u32 i = 0; i < count; ++i)
        {
            [[maybe_unused]] auto e = reg.createEntity(arch);
        }

        char label[64];
        std::snprintf(label, sizeof(label), "Physics step (%uk entities)", count / 1000);

        const core::f64 ms = benchmarkMs(label, [&]()
        {
            world.step(0.0166f);
        });

        const core::f64 entPerSec = static_cast<core::f64>(count) / (ms / 1000.0);
        std::printf("    -> Throughput: %.0f ent/sec  [%s]\n",
                    entPerSec, frameRateVerdict(ms));
    }
}

void benchmarkWorldPartition()
{
    ecs::Registry reg;
    ecs::WorldPartition world(math::Fixed32{10});

    ecs::Archetype arch;
    arch.add(ecs::ComponentId::Position);
    arch.add(ecs::ComponentId::Velocity);
    arch.add(ecs::ComponentId::Mass);
    arch.add(ecs::ComponentId::AABB);
    arch.add(ecs::ComponentId::Health);

    // Deterministic LCG for reproducible placement
    core::u32 seed = 12345;
    auto nextRand = [&seed]() -> float {
        seed = seed * 1103515245u + 12345u;
        return static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
    };

    static constexpr int kEntityCount = 10000;
    std::vector<ecs::EntityId> entityIds;
    entityIds.reserve(kEntityCount);

    benchmarkMs("WorldPartition create 10k entities", [&]()
    {
        for (int i = 0; i < kEntityCount; ++i)
        {
            auto entityResult = reg.createEntity(arch);
            if (!entityResult.has_value())
                continue;

            auto entityId = entityResult.value();
            entityIds.push_back(entityId);

            float px = (nextRand() - 0.5f) * 10000.0f;
            float py = nextRand() * 100.0f;
            float pz = (nextRand() - 0.5f) * 10000.0f;

            auto fixedPos = math::Vec3<math::Fixed32>{
                math::Fixed32::fromFloat(px),
                math::Fixed32::fromFloat(py),
                math::Fixed32::fromFloat(pz)
            };
            [[maybe_unused]] auto r = world.insertOrUpdate(entityId, fixedPos);
        }
    });

    // Warm up
    for (int i = 0; i < 5; ++i)
        world.step(0.016f);

    benchmarkMs("WorldPartition step 100 frames (10k ents)", [&]()
    {
        for (int i = 0; i < 100; ++i)
            world.step(0.016f);
    });
}

void benchmarkFlatAtomicHashMap()
{
    static constexpr core::u32 kPoolCap = 16384;

    benchmarkMs("FlatAtomicHashMap insert 16k", []()
    {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            map.insert(static_cast<core::u64>(i + 1));
        }
    });

    benchmarkMs("FlatAtomicHashMap get 16k", []()
    {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            map.insert(static_cast<core::u64>(i + 1));
        }
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            [[maybe_unused]] auto* v = map.get(static_cast<core::u64>(i + 1));
        }
    });

    benchmarkMs("FlatAtomicHashMap forEach 16k", []()
    {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto* v = map.insert(static_cast<core::u64>(i + 1));
            if (v)
                *v = static_cast<int>(i);
        }
        core::u64 sum = 0;
        map.forEach([&sum](int& val) { sum += static_cast<core::u64>(val); });
    });

    benchmarkMs("FlatAtomicHashMap forEachParallel 16k", []()
    {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto* v = map.insert(static_cast<core::u64>(i + 1));
            if (v)
                *v = static_cast<int>(i);
        }
        concurrency::ThreadPool pool{4};
        std::atomic<core::u64> sum{0};
        map.forEachParallel(pool, [&sum](int& val) {
            sum.fetch_add(static_cast<core::u64>(val), std::memory_order_relaxed);
        });
    });

    benchmarkMs("FlatAtomicHashMap insert/remove churn 16k", []()
    {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        // Insert all
        for (core::u32 i = 0; i < kPoolCap; ++i)
            map.insert(static_cast<core::u64>(i + 1));
        // Remove half
        for (core::u32 i = 0; i < kPoolCap / 2; ++i)
            map.remove(static_cast<core::u64>(i + 1));
        // Re-insert (tests pool recycling)
        for (core::u32 i = 0; i < kPoolCap / 2; ++i)
            map.insert(static_cast<core::u64>(i + 1));
    });
}

void benchmarkThreadPool()
{
    concurrency::ThreadPool pool{4};

    benchmarkMs("ThreadPool dispatch 10k tasks", [&]()
    {
        std::atomic<core::u64> counter{0};
        std::vector<std::future<void>> futures;
        futures.reserve(10000);
        for (int i = 0; i < 10000; ++i)
        {
            futures.push_back(pool.enqueue([&counter]() {
                counter.fetch_add(1, std::memory_order_relaxed);
            }));
        }
        for (auto& f : futures)
            f.get();
    });
}

// ---------------------------------------------------------------------------
// SoA vs AoS memory layout benchmark
//   Demonstrates the cache-miss penalty of Array-of-Structs vs Structure-of-
//   Arrays when iterating over a single component (x-coordinate only).
// ---------------------------------------------------------------------------
void benchmarkSoA_vs_AoS()
{
    constexpr int kN = 1'000'000;
    std::printf("\n  --- SoA vs AoS layout (%dM elements, sum-x) ---\n", kN / 1'000'000);

    // AoS: each element stores (x,y,z) interleaved
    struct AoS_Vec3 { float x, y, z; };
    std::vector<AoS_Vec3> aos(kN);
    for (int i = 0; i < kN; ++i) aos[i] = {static_cast<float>(i), 1.0f, 2.0f};

    // SoA: three contiguous float arrays
    std::vector<float> xs(kN), ys(kN), zs(kN);
    for (int i = 0; i < kN; ++i) { xs[i] = static_cast<float>(i); ys[i] = 1.0f; zs[i] = 2.0f; }

    double sumAoS = 0.0;
    benchmarkMs("Sum-x AoS (Vec3[]) 1M", [&]()
    {
        double acc = 0.0;
        for (int i = 0; i < kN; ++i) acc += static_cast<double>(aos[i].x);
        sumAoS = acc;
    });

    double sumSoA = 0.0;
    benchmarkMs("Sum-x SoA (float[]) 1M", [&]()
    {
        double acc = 0.0;
        for (int i = 0; i < kN; ++i) acc += static_cast<double>(xs[i]);
        sumSoA = acc;
    });

    // Prevent dead-code elimination
    if (sumAoS != sumSoA)
        std::printf("    [sanity] sums differ — unexpected\n");
}

// ---------------------------------------------------------------------------
// Entity lookup latency: sparse set O(1) vs linear scan O(n)
// ---------------------------------------------------------------------------
void benchmarkEntityLookup()
{
    constexpr core::u32 kN = 10000;
    std::printf("\n  --- Entity lookup: SparseSet O(1) vs linear scan O(n) ---\n");

    // Build a SparseSet and a plain vector
    container::SparseSet<core::u32> sparseSet{1u << 14};
    std::vector<core::u32> denseVec;
    denseVec.reserve(kN);

    core::u32 seed = 42;
    for (core::u32 i = 0; i < kN; ++i)
    {
        seed = seed * 6364136223846793005u + 1442695040888963407u;
        const core::u32 slot = seed & 0x3FFF;  // 14-bit slot
        sparseSet.insert(slot, i);
        denseVec.push_back(slot);
    }

    // Look up the LAST 1000 slots to stress worst-case for linear
    constexpr int kLookups = 1000;

    benchmarkMs("SparseSet lookup 1k slots", [&]()
    {
        core::u32 sink = 0;
        for (int i = 0; i < kLookups; ++i)
        {
            const core::u32 slot = denseVec[kN - 1 - static_cast<core::u32>(i) % kN];
            if (const core::u32* p = sparseSet.find(slot))
                sink = *p;
        }
        if (sink == ~0u) std::printf("[debug] sink=%u\n", sink);
    });

    benchmarkMs("Linear scan lookup 1k slots in 10k vec", [&]()
    {
        core::u32 sink = 0;
        for (int i = 0; i < kLookups; ++i)
        {
            const core::u32 target = denseVec[kN - 1 - static_cast<core::u32>(i) % kN];
            for (core::u32 j = 0; j < kN; ++j)
            {
                if (denseVec[j] == target) { sink = j; break; }
            }
        }
        if (sink == ~0u) std::printf("[debug] sink=%u\n", sink);
    });
}

// ---------------------------------------------------------------------------
// Collision detection: N² brute-force vs broad-phase (WorldPartition cells)
// ---------------------------------------------------------------------------
void benchmarkCollisionBroadphase()
{
    constexpr int kN = 2000;
    std::printf("\n  --- Collision N\u00b2 vs broad-phase (%d entities) ---\n", kN);

    // Simple AABB for N² test
    struct AABB { float xmin, xmax, ymin, ymax, zmin, zmax; };
    std::vector<AABB> boxes(kN);
    core::u32 rng = 99999;
    for (int i = 0; i < kN; ++i)
    {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        const float cx = static_cast<float>(rng & 0xFFFF) / 65535.0f * 1000.0f;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        const float cy = static_cast<float>(rng & 0xFFFF) / 65535.0f * 1000.0f;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        const float cz = static_cast<float>(rng & 0xFFFF) / 65535.0f * 1000.0f;
        boxes[i] = {cx - 0.5f, cx + 0.5f, cy - 0.5f, cy + 0.5f, cz - 0.5f, cz + 0.5f};
    }

    benchmarkMs("N\u00b2 AABB collision check (2k)", [&]()
    {
        int pairs = 0;
        for (int a = 0; a < kN; ++a)
            for (int b = a + 1; b < kN; ++b)
                if (boxes[a].xmax >= boxes[b].xmin && boxes[b].xmax >= boxes[a].xmin &&
                    boxes[a].ymax >= boxes[b].ymin && boxes[b].ymax >= boxes[a].ymin &&
                    boxes[a].zmax >= boxes[b].zmin && boxes[b].zmax >= boxes[a].zmin)
                    ++pairs;
        // Prevent dead-code elimination
        if (pairs < 0) std::printf("  [debug] pairs=%d\n", pairs);
    });

    // Broad-phase via WorldPartition queryRadius
    ecs::WorldPartition world(math::Fixed32{10});
    ecs::Registry reg;
    ecs::Archetype arch;
    arch.add(ecs::ComponentId::Position);
    arch.add(ecs::ComponentId::AABB);

    for (int i = 0; i < kN; ++i)
    {
        if (auto e = reg.createEntity(arch))
        {
            math::Vec3<math::Fixed32> pos{
                math::Fixed32::fromFloat(boxes[i].xmin + 0.5f),
                math::Fixed32::fromFloat(boxes[i].ymin + 0.5f),
                math::Fixed32::fromFloat(boxes[i].zmin + 0.5f)};
            [[maybe_unused]] auto r = world.insertOrUpdate(e.value(), pos);
        }
    }

    benchmarkMs("Broad-phase queryRadius (2k, r=10)", [&]()
    {
        core::u32 total = 0;
        for (int i = 0; i < kN; ++i)
        {
            std::vector<ecs::EntityId> hits;
            const math::Vec3<math::Fixed32> center{
                math::Fixed32::fromFloat(boxes[i].xmin + 0.5f),
                math::Fixed32::fromFloat(boxes[i].ymin + 0.5f),
                math::Fixed32::fromFloat(boxes[i].zmin + 0.5f)};
            world.queryRadius(center, math::Fixed32{10}, hits);
            total += static_cast<core::u32>(hits.size());
        }
        if (total == ~0u) std::printf("[debug] total=%u\n", total);
    });
}

} // anonymous namespace

int main(int /*argc*/, char* /*argv*/[])
{
    core::Log::info("=== LplPlugin Benchmark ===");
    std::printf("\n");

    benchmarkArena();
    benchmarkFixedMath();
    benchmarkMorton();
    benchmarkCordic();
    benchmarkRegistry();
    benchmarkPhysics();
    benchmarkPhysicsScalability();
    benchmarkWorldPartition();
    benchmarkFlatAtomicHashMap();
    benchmarkThreadPool();
    benchmarkSoA_vs_AoS();
    benchmarkEntityLookup();
    benchmarkCollisionBroadphase();

    std::printf("\nDone.\n");
    return 0;
}
