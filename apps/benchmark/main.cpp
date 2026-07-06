/**
 * @file main.cpp
 * @brief LplPlugin benchmark entry-point: defines the individual benchmark
 *        cases and runs them. The measurement machinery (timing, statistics,
 *        DCE barriers) and host introspection live in the lpl::bench module.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/bench/Harness.hpp>
#include <lpl/bench/SystemInfo.hpp>

#include <lpl/concurrency/ThreadPool.hpp>
#include <lpl/container/FlatAtomicHashMap.hpp>
#include <lpl/container/SparseSet.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/math/Cordic.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Morton.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/memory/PoolAllocator.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <vector>

using namespace lpl;

namespace {

void benchmarkArena()
{
    memory::ArenaAllocator arena{4 * 1024 * 1024};
    bench::run("ArenaAllocator 10k allocs", [&]() {
        for (int i = 0; i < 10000; ++i)
        {
            void *p = arena.allocate(128, 16);
            bench::doNotOptimize(p);
        }
        arena.reset();
    });
}

void benchmarkFixedMath()
{
    bench::run("Fixed32 mul 1M ops", []() {
        math::Fixed32 a = math::Fixed32::fromFloat(3.14159f);
        math::Fixed32 b = math::Fixed32::fromFloat(2.71828f);
        math::Fixed32 r{0};
        for (int i = 0; i < 1000000; ++i)
        {
            r = a * b;
            a = r;
            b = b + math::Fixed32{1};
        }
        bench::doNotOptimize(r);
    });
}

void benchmarkMorton()
{
    bench::run("Morton encode3D 1M", []() {
        for (core::u32 i = 0; i < 1000000; ++i)
        {
            auto code = math::morton::encode3D(static_cast<core::i32>(i), static_cast<core::i32>(i + 1),
                                               static_cast<core::i32>(i + 2));
            bench::doNotOptimize(code);
        }
    });
}

// ---------------------------------------------------------------------------
// Trigonometry: CORDIC (fixed-point, shift-add only) vs std::sin (double,
// libm). Reports throughput for both AND the numerical accuracy of the CORDIC
// approximation — a speed number for an approximate function is meaningless
// without its error, so we measure max-absolute and RMS error over [-π, π].
// ---------------------------------------------------------------------------
void benchmarkTrigonometry()
{
    constexpr int kN = 1'000'000;
    constexpr core::f64 kPi = 3.141592653589793;
    // This CORDIC has no argument range reduction, so it only converges within
    // the rotation range (≈ ±1.74 rad). We benchmark over the reduced domain
    // [-π/2, π/2] — the range a real caller would fold angles into — for a fair
    // speed/accuracy comparison against libm.
    const auto angleAt = [](core::f64 t) { return (t - 0.5) * kPi; }; // t∈[0,1] → [-π/2, π/2]

    std::printf("\n  --- Trigonometry: CORDIC (Fixed32) vs std::sin (double), domain [-π/2, π/2] ---\n");

    bench::run("CORDIC sin 1M", [&]() {
        math::Fixed32 acc{0};
        for (int i = 0; i < kN; ++i)
        {
            const float ang = static_cast<float>(angleAt(static_cast<core::f64>(i) / kN));
            acc = acc + math::Cordic::sin(math::Fixed32::fromFloat(ang));
        }
        bench::doNotOptimize(acc);
    });

    bench::run("std::sin 1M (double, libm)", [&]() {
        core::f64 acc = 0.0;
        for (int i = 0; i < kN; ++i)
        {
            acc += std::sin(angleAt(static_cast<core::f64>(i) / kN));
        }
        bench::doNotOptimize(acc);
    });

    // Accuracy sweep — compare CORDIC against libm as ground truth.
    constexpr int kSamples = 200000;
    core::f64 maxErr = 0.0;
    core::f64 sumSq = 0.0;
    for (int i = 0; i < kSamples; ++i)
    {
        const core::f64 ang = angleAt(static_cast<core::f64>(i) / kSamples);
        const core::f64 ref = std::sin(ang);
        const core::f64 got = math::Cordic::sin(math::Fixed32::fromFloat(static_cast<float>(ang))).toDouble();
        const core::f64 e = std::fabs(got - ref);
        maxErr = std::max(maxErr, e);
        sumSq += e * e;
    }
    const core::f64 rms = std::sqrt(sumSq / kSamples);
    std::printf("    -> CORDIC accuracy vs libm: max abs err %.3e, RMS %.3e (over %d samples in [-π/2, π/2])\n", maxErr,
                rms, kSamples);
}

void benchmarkRegistry()
{
    ecs::Archetype arch;
    arch.add(ecs::ComponentId::Position);

    // create/destroy is net-zero per iteration, so the shared registry stays
    // balanced across repetitions.
    ecs::Registry reg;
    bench::run("Registry create/destroy 10k entities", [&]() {
        for (int i = 0; i < 10000; ++i)
        {
            auto e = reg.createEntity(arch);
            if (e)
            {
                auto r = reg.destroyEntity(e.value());
                bench::doNotOptimize(r);
            }
        }
    });

    // Batch creation accumulates entities, so each repetition builds (and
    // discards) its own registry — the construction cost is dwarfed by 100k
    // creations and this keeps the measurement repeatable.
    bench::run("Registry create 100k entities", [&]() {
        ecs::Registry local;
        for (int i = 0; i < 100000; ++i)
        {
            auto e = local.createEntity(arch);
            bench::doNotOptimize(e);
        }
    });
}

// Builds a physics archetype (Position, Velocity, Mass, AABB) and populates a
// registry with @p count entities, then returns an initialised CPU physics
// backend ready to step. The backend runs the real integrate → collide → sleep
// pipeline over the ECS chunks — unlike WorldPartition::step(), whose CPU path
// is only a GPU-dispatch gateway and does no integration work.
static ecs::Archetype makePhysicsArchetype()
{
    ecs::Archetype arch;
    arch.add(ecs::ComponentId::Position);
    arch.add(ecs::ComponentId::Velocity);
    arch.add(ecs::ComponentId::Mass);
    arch.add(ecs::ComponentId::AABB);
    return arch;
}

// Writes spread-out positions, unit masses, small AABB half-extents and a
// small initial velocity into the freshly-created entities, so the physics
// step exercises a realistic distribution rather than a pathological pile of
// coincident bodies at the origin.
static void seedPhysicsComponents(ecs::Registry &reg, core::u32 count)
{
    core::u32 seed = 2166136261u;
    auto nextRand = [&seed]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xFFFF) / 65535.0f;
    };
    // Cube side scaled so density stays ~constant as count grows.
    const float extent = std::cbrt(static_cast<float>(count)) * 4.0f;

    for (const auto &partition : reg.partitions())
    {
        for (const auto &chunk : partition->chunks())
        {
            const core::u32 n = chunk->count();
            auto *pos = static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *vel = static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto *mass = static_cast<float *>(chunk->writeComponent(ecs::ComponentId::Mass));
            auto *aabb = static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::AABB));
            for (core::u32 i = 0; i < n; ++i)
            {
                if (pos)
                    pos[i] = {(nextRand() - 0.5f) * extent, (nextRand() - 0.5f) * extent,
                              (nextRand() - 0.5f) * extent};
                if (vel)
                    vel[i] = {(nextRand() - 0.5f), (nextRand() - 0.5f), (nextRand() - 0.5f)};
                if (mass)
                    mass[i] = 1.0f;
                if (aabb)
                    aabb[i] = {0.5f, 0.5f, 0.5f};
            }
        }
    }
    (void) count;
}

void benchmarkPhysics()
{
    ecs::Registry reg;
    const ecs::Archetype arch = makePhysicsArchetype();
    for (int i = 0; i < 10000; ++i)
    {
        [[maybe_unused]] auto e = reg.createEntity(arch);
    }
    seedPhysicsComponents(reg, 10000);

    physics::CpuPhysicsBackend backend{reg};
    [[maybe_unused]] auto ok = backend.init();

    // Warm up the pipeline (sleep buffers, caches) before timing.
    for (int i = 0; i < 5; ++i)
        [[maybe_unused]] auto r = backend.step(0.0166f);

    bench::run("Physics step (10k, integrate+collide+sleep)", [&]() {
        [[maybe_unused]] auto r = backend.step(0.0166f);
    });
}

void benchmarkPhysicsScalability()
{
    // The CPU collision pass is O(n²) per chunk, so this sweep is deliberately
    // capped: it exists to *show* the quadratic wall that motivates the octree
    // / GPU broadphase, not to claim a headline number at 100k coincident
    // bodies. Fewer repetitions for the heavier counts keep total runtime sane.
    constexpr core::u32 kCounts[] = {1000, 2000, 5000, 10000, 20000};

    std::printf("\n  --- Physics Scalability Sweep (CPU backend: integrate + O(n²) collide + sleep) ---\n");

    for (core::u32 count : kCounts)
    {
        ecs::Registry reg;
        const ecs::Archetype arch = makePhysicsArchetype();
        for (core::u32 i = 0; i < count; ++i)
        {
            [[maybe_unused]] auto e = reg.createEntity(arch);
        }
        seedPhysicsComponents(reg, count);

        physics::CpuPhysicsBackend backend{reg};
        [[maybe_unused]] auto ok = backend.init();
        for (int i = 0; i < 3; ++i)
            [[maybe_unused]] auto r = backend.step(0.0166f);

        char label[64];
        std::snprintf(label, sizeof(label), "Physics step (%uk entities)", count / 1000);

        bench::Config cfg;
        cfg.minReps = 5;
        cfg.maxReps = 200;
        cfg.targetTotalMs = 400.0;
        const bench::Result r = bench::run(label, [&]() { [[maybe_unused]] auto s = backend.step(0.0166f); }, cfg);

        const core::f64 ms = r.medianNs / 1e6;
        const core::f64 entPerSec = static_cast<core::f64>(count) / (r.medianNs / 1e9);
        std::printf("    -> Throughput: %.2e ent/sec  [%s]\n", entPerSec, bench::frameRateVerdict(ms));
    }
}

void benchmarkWorldPartition()
{
    static constexpr int kEntityCount = 10000;

    auto makeArchetype = []() {
        ecs::Archetype arch;
        arch.add(ecs::ComponentId::Position);
        arch.add(ecs::ComponentId::Velocity);
        arch.add(ecs::ComponentId::Mass);
        arch.add(ecs::ComponentId::AABB);
        arch.add(ecs::ComponentId::Health);
        return arch;
    };

    // Deterministic LCG for reproducible placement.
    auto placeAll = [](ecs::Registry &reg, ecs::WorldPartition &world, const ecs::Archetype &arch) {
        core::u32 seed = 12345;
        auto nextRand = [&seed]() -> float {
            seed = seed * 1103515245u + 12345u;
            return static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
        };
        for (int i = 0; i < kEntityCount; ++i)
        {
            auto entityResult = reg.createEntity(arch);
            if (!entityResult.has_value())
                continue;
            const auto entityId = entityResult.value();
            const float px = (nextRand() - 0.5f) * 10000.0f;
            const float py = nextRand() * 100.0f;
            const float pz = (nextRand() - 0.5f) * 10000.0f;
            const math::Vec3<math::Fixed32> fixedPos{math::Fixed32::fromFloat(px), math::Fixed32::fromFloat(py),
                                                     math::Fixed32::fromFloat(pz)};
            [[maybe_unused]] auto r = world.insertOrUpdate(entityId, fixedPos);
        }
    };

    const ecs::Archetype arch = makeArchetype();

    // Build + populate is self-contained per repetition (state accumulates).
    // This measures the spatial-index insertion path (Morton keying + cell
    // assignment); the actual physics integration is benchmarked separately
    // via CpuPhysicsBackend, since WorldPartition::step()'s CPU path is only a
    // GPU-dispatch gateway and does no integration work.
    bench::run("WorldPartition build+insert 10k entities", [&]() {
        ecs::Registry reg;
        ecs::WorldPartition world(math::Fixed32{10});
        placeAll(reg, world, arch);
        bench::doNotOptimize(world);
    });
}

void benchmarkFlatAtomicHashMap()
{
    static constexpr core::u32 kPoolCap = 16384;

    bench::run("FlatAtomicHashMap insert 16k", []() {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.insert(static_cast<core::u64>(i + 1));
            bench::doNotOptimize(v);
        }
    });

    bench::run("FlatAtomicHashMap get 16k", []() {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
            map.insert(static_cast<core::u64>(i + 1));
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.get(static_cast<core::u64>(i + 1));
            bench::doNotOptimize(v);
        }
    });

    bench::run("FlatAtomicHashMap forEach 16k", []() {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.insert(static_cast<core::u64>(i + 1));
            if (v)
                *v = static_cast<int>(i);
        }
        core::u64 sum = 0;
        map.forEach([&sum](int &val) { sum += static_cast<core::u64>(val); });
        bench::doNotOptimize(sum);
    });

    bench::run("FlatAtomicHashMap forEachParallel 16k", []() {
        container::FlatAtomicHashMap<int> map(kPoolCap);
        for (core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.insert(static_cast<core::u64>(i + 1));
            if (v)
                *v = static_cast<int>(i);
        }
        concurrency::ThreadPool pool{4};
        std::atomic<core::u64> sum{0};
        map.forEachParallel(
            pool, [&sum](int &val) { sum.fetch_add(static_cast<core::u64>(val), std::memory_order_relaxed); });
        bench::doNotOptimize(sum);
    });

    bench::run("FlatAtomicHashMap insert/remove churn 16k", []() {
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

    bench::run("ThreadPool dispatch 10k tasks", [&]() {
        std::atomic<core::u64> counter{0};
        std::vector<std::future<void>> futures;
        futures.reserve(10000);
        for (int i = 0; i < 10000; ++i)
        {
            futures.push_back(pool.enqueue([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); }));
        }
        for (auto &f : futures)
            f.get();
        bench::doNotOptimize(counter);
    });
}

// ---------------------------------------------------------------------------
// Allocator throughput: the engine's custom allocators vs the system malloc.
// Same workload (N fixed-size blocks) through three strategies:
//   * Arena  — bump-pointer allocate, then a single O(1) mass reset;
//   * Pool   — intrusive free-list, per-block acquire then per-block release;
//   * malloc — the libc baseline, per-block malloc then free.
// This is the quantitative justification for shipping custom allocators.
// ---------------------------------------------------------------------------
void benchmarkAllocators()
{
    constexpr int kN = 100000;
    struct alignas(16) Block64 {
        core::u64 data[8];
    };

    std::printf("\n  --- Allocator throughput (%dk x 64-byte blocks) ---\n", kN / 1000);

    std::vector<void *> ptrs(static_cast<core::usize>(kN), nullptr);

    bench::run("Arena  bump-alloc + reset", [&]() {
        memory::ArenaAllocator arena{static_cast<core::usize>(kN) * sizeof(Block64) + 4096};
        for (int i = 0; i < kN; ++i)
        {
            void *p = arena.allocate(sizeof(Block64), 16);
            bench::doNotOptimize(p);
        }
        arena.reset();
    });

    bench::run("Pool   acquire + release", [&]() {
        memory::PoolAllocator<Block64> pool{static_cast<core::usize>(kN)};
        for (int i = 0; i < kN; ++i)
            ptrs[static_cast<core::usize>(i)] = pool.acquire();
        for (int i = 0; i < kN; ++i)
            pool.release(static_cast<Block64 *>(ptrs[static_cast<core::usize>(i)]));
        bench::doNotOptimize(ptrs.data());
    });

    bench::run("malloc + free (libc baseline)", [&]() {
        for (int i = 0; i < kN; ++i)
        {
            ptrs[static_cast<core::usize>(i)] = std::malloc(sizeof(Block64));
            bench::doNotOptimize(ptrs[static_cast<core::usize>(i)]);
        }
        for (int i = 0; i < kN; ++i)
            std::free(ptrs[static_cast<core::usize>(i)]);
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
    struct AoS_Vec3 {
        float x, y, z;
    };
    std::vector<AoS_Vec3> aos(kN);
    for (int i = 0; i < kN; ++i)
        aos[i] = {static_cast<float>(i), 1.0f, 2.0f};

    // SoA: three contiguous float arrays
    std::vector<float> xs(kN), ys(kN), zs(kN);
    for (int i = 0; i < kN; ++i)
    {
        xs[i] = static_cast<float>(i);
        ys[i] = 1.0f;
        zs[i] = 2.0f;
    }

    bench::run("Sum-x AoS (Vec3[]) 1M", [&]() {
        double acc = 0.0;
        for (int i = 0; i < kN; ++i)
            acc += static_cast<double>(aos[i].x);
        bench::doNotOptimize(acc);
    });

    bench::run("Sum-x SoA (float[]) 1M", [&]() {
        double acc = 0.0;
        for (int i = 0; i < kN; ++i)
            acc += static_cast<double>(xs[i]);
        bench::doNotOptimize(acc);
    });
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
        const core::u32 slot = seed & 0x3FFF; // 14-bit slot
        sparseSet.insert(slot, i);
        denseVec.push_back(slot);
    }

    // Look up the LAST 1000 slots to stress worst-case for linear
    constexpr int kLookups = 1000;

    bench::run("SparseSet lookup 1k slots", [&]() {
        core::u32 sink = 0;
        for (int i = 0; i < kLookups; ++i)
        {
            const core::u32 slot = denseVec[kN - 1 - static_cast<core::u32>(i) % kN];
            if (const core::u32 *p = sparseSet.find(slot))
                sink = *p;
        }
        bench::doNotOptimize(sink);
    });

    bench::run("Linear scan lookup 1k slots in 10k vec", [&]() {
        core::u32 sink = 0;
        for (int i = 0; i < kLookups; ++i)
        {
            const core::u32 target = denseVec[kN - 1 - static_cast<core::u32>(i) % kN];
            for (core::u32 j = 0; j < kN; ++j)
            {
                if (denseVec[j] == target)
                {
                    sink = j;
                    break;
                }
            }
        }
        bench::doNotOptimize(sink);
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
    struct AABB {
        float xmin, xmax, ymin, ymax, zmin, zmax;
    };
    std::vector<AABB> boxes(kN);
    core::u32 rng = 99999;
    for (int i = 0; i < kN; ++i)
    {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        const float cx = static_cast<float>(rng & 0xFFFF) / 65535.0f * 1000.0f;
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        const float cy = static_cast<float>(rng & 0xFFFF) / 65535.0f * 1000.0f;
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        const float cz = static_cast<float>(rng & 0xFFFF) / 65535.0f * 1000.0f;
        boxes[i] = {cx - 0.5f, cx + 0.5f, cy - 0.5f, cy + 0.5f, cz - 0.5f, cz + 0.5f};
    }

    bench::run("N\u00b2 AABB collision check (2k)", [&]() {
        int pairs = 0;
        for (int a = 0; a < kN; ++a)
            for (int b = a + 1; b < kN; ++b)
                if (boxes[a].xmax >= boxes[b].xmin && boxes[b].xmax >= boxes[a].xmin &&
                    boxes[a].ymax >= boxes[b].ymin && boxes[b].ymax >= boxes[a].ymin &&
                    boxes[a].zmax >= boxes[b].zmin && boxes[b].zmax >= boxes[a].zmin)
                    ++pairs;
        bench::doNotOptimize(pairs);
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
            math::Vec3<math::Fixed32> pos{math::Fixed32::fromFloat(boxes[i].xmin + 0.5f),
                                          math::Fixed32::fromFloat(boxes[i].ymin + 0.5f),
                                          math::Fixed32::fromFloat(boxes[i].zmin + 0.5f)};
            [[maybe_unused]] auto r = world.insertOrUpdate(e.value(), pos);
        }
    }

    bench::run("Broad-phase queryRadius (2k, r=10)", [&]() {
        core::u32 total = 0;
        for (int i = 0; i < kN; ++i)
        {
            std::vector<ecs::EntityId> hits;
            const math::Vec3<math::Fixed32> center{math::Fixed32::fromFloat(boxes[i].xmin + 0.5f),
                                                   math::Fixed32::fromFloat(boxes[i].ymin + 0.5f),
                                                   math::Fixed32::fromFloat(boxes[i].zmin + 0.5f)};
            world.queryRadius(center, math::Fixed32{10}, hits);
            total += static_cast<core::u32>(hits.size());
        }
        bench::doNotOptimize(total);
    });
}

} // anonymous namespace

int main(int /*argc*/, char * /*argv*/[])
{
    core::Log::info("=== LplPlugin Benchmark ===");
    std::printf("\n");

    bench::printSystemInfo();
    bench::printLegend();

    benchmarkArena();
    benchmarkFixedMath();
    benchmarkMorton();
    benchmarkAllocators();
    benchmarkTrigonometry();
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
