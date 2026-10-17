#include <lpl/bench/Harness.hpp>
#include <lpl/bench/JsonRows.hpp>
#include <lpl/bench/RingBench.hpp>
#include <lpl/bench/SystemInfo.hpp>
#include <lpl/bench/VoxelBench.hpp>

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

#include <lpl/engine/EventQueue.hpp>
#include <lpl/engine/PacketDispatch.hpp>
#include <lpl/engine/systems/AoiBroadcastSystem.hpp>
#include <lpl/engine/systems/BroadcastSystem.hpp>
#include <lpl/net/Endpoint.hpp>
#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/net/session/SessionManager.hpp>
#include <lpl/net/transport/ITransport.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <format>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void benchmarkArena()
{
    lpl::memory::ArenaAllocator arena{4 * 1024 * 1024};
    lpl::bench::run("ArenaAllocator 10k allocs", [&]() {
        for (int i = 0; i < 10000; ++i)
        {
            void *p = arena.allocate(128, 16);
            lpl::bench::doNotOptimize(p);
        }
        arena.reset();
    });
}

void benchmarkFixedMath()
{
    lpl::bench::run("Fixed32 mul 1M ops", []() {
        lpl::math::Fixed32 a = lpl::math::Fixed32::fromFloat(3.14159f);
        lpl::math::Fixed32 b = lpl::math::Fixed32::fromFloat(2.71828f);
        lpl::math::Fixed32 r{0};
        for (int i = 0; i < 1000000; ++i)
        {
            r = a * b;
            a = r;
            // TODO(#361): one raw unit (1/65536), or math::Fixed32::one()?
            b = b + lpl::math::Fixed32::fromRaw(1);
        }
        lpl::bench::doNotOptimize(r);
    });
}

void benchmarkMorton()
{
    lpl::bench::run("Morton encode3D 1M", []() {
        for (lpl::core::u32 i = 0; i < 1000000; ++i)
        {
            auto code = lpl::math::morton::encode3D(static_cast<lpl::core::i32>(i), static_cast<lpl::core::i32>(i + 1),
                                                    static_cast<lpl::core::i32>(i + 2));
            lpl::bench::doNotOptimize(code);
        }
    });
}

/**
 * @brief Trigonometry: CORDIC (fixed-point, shift-add only) vs std::sin (double, libm).
 *
 * Reports throughput for both AND the numerical accuracy of the CORDIC approximation — a speed
 * number for an approximate function is meaningless without its error, so we measure max-absolute
 * and RMS error, against libm as ground truth. Both run over [-π/2, π/2]: `angleAt` maps t∈[0,1]
 * to that domain.
 */
void benchmarkTrigonometry()
{
    constexpr int kN = 1'000'000;
    constexpr lpl::core::f64 kPi = 3.141592653589793;
    const auto angleAt = [](lpl::core::f64 t) { return (t - 0.5) * kPi; };

    std::printf("\n  --- Trigonometry: CORDIC (Fixed32) vs std::sin (double), domain [-π/2, π/2] ---\n");

    lpl::bench::run("CORDIC sin 1M", [&]() {
        lpl::math::Fixed32 acc{0};
        for (int i = 0; i < kN; ++i)
        {
            const float ang = static_cast<float>(angleAt(static_cast<lpl::core::f64>(i) / kN));
            acc = acc + lpl::math::Cordic::sin(lpl::math::Fixed32::fromFloat(ang));
        }
        lpl::bench::doNotOptimize(acc);
    });

    lpl::bench::run("std::sin 1M (double, libm)", [&]() {
        lpl::core::f64 acc = 0.0;
        for (int i = 0; i < kN; ++i)
        {
            acc += std::sin(angleAt(static_cast<lpl::core::f64>(i) / kN));
        }
        lpl::bench::doNotOptimize(acc);
    });

    constexpr int kSamples = 200000;
    lpl::core::f64 maxErr = 0.0;
    lpl::core::f64 sumSq = 0.0;
    for (int i = 0; i < kSamples; ++i)
    {
        const lpl::core::f64 ang = angleAt(static_cast<lpl::core::f64>(i) / kSamples);
        const lpl::core::f64 ref = std::sin(ang);
        const lpl::core::f64 got =
            lpl::math::Cordic::sin(lpl::math::Fixed32::fromFloat(static_cast<float>(ang))).toDouble();
        const lpl::core::f64 e = std::fabs(got - ref);
        maxErr = std::max(maxErr, e);
        sumSq += e * e;
    }
    const lpl::core::f64 rms = std::sqrt(sumSq / kSamples);
    std::printf("    -> CORDIC accuracy vs libm: max abs err %.3e, RMS %.3e (over %d samples in [-π/2, π/2])\n", maxErr,
                rms, kSamples);
}

/**
 * @brief The registry's entity creation and destruction.
 *
 * Create/destroy is net-zero per iteration, so the shared registry stays balanced across
 * repetitions. Batch creation accumulates entities, so each repetition builds (and discards) its
 * own registry — the construction cost is dwarfed by 100k creations and this keeps the measurement
 * repeatable.
 */
void benchmarkRegistry()
{
    lpl::ecs::Archetype arch;
    arch.add(lpl::ecs::ComponentId::Position);

    lpl::ecs::Registry reg;
    lpl::bench::run("Registry create/destroy 10k entities", [&]() {
        for (int i = 0; i < 10000; ++i)
        {
            auto e = reg.createEntity(arch);
            if (e)
            {
                auto r = reg.destroyEntity(e.value());
                lpl::bench::doNotOptimize(r);
            }
        }
    });

    lpl::bench::run("Registry create 100k entities", [&]() {
        lpl::ecs::Registry local;
        for (int i = 0; i < 100000; ++i)
        {
            auto e = local.createEntity(arch);
            lpl::bench::doNotOptimize(e);
        }
    });
}

/**
 * @brief Builds a physics archetype (Position, Velocity, Mass, AABB).
 */
static lpl::ecs::Archetype makePhysicsArchetype()
{
    lpl::ecs::Archetype arch;
    arch.add(lpl::ecs::ComponentId::Position);
    arch.add(lpl::ecs::ComponentId::Velocity);
    arch.add(lpl::ecs::ComponentId::Mass);
    arch.add(lpl::ecs::ComponentId::AABB);
    return arch;
}

/**
 * @brief Writes spread-out positions, unit masses, small AABB half-extents and a
 *        small initial velocity into the freshly-created entities, so the physics
 *        step exercises a realistic distribution rather than a pathological pile of
 *        coincident bodies at the origin.
 *
 * The positions fill a cube whose side scales with the cube root of @p count, so density stays
 * roughly constant as count grows.
 */
static void seedPhysicsComponents(lpl::ecs::Registry &reg, lpl::core::u32 count)
{
    lpl::core::u32 seed = 2166136261u;
    auto nextRand = [&seed]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xFFFF) / 65535.0f;
    };
    const float extent = std::cbrt(static_cast<float>(count)) * 4.0f;

    for (const auto &partition : reg.partitions())
    {
        for (const auto &chunk : partition->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            auto *pos = static_cast<lpl::math::Vec3<float> *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            auto *vel = static_cast<lpl::math::Vec3<float> *>(chunk->writeComponent(lpl::ecs::ComponentId::Velocity));
            auto *mass = static_cast<float *>(chunk->writeComponent(lpl::ecs::ComponentId::Mass));
            auto *aabb = static_cast<lpl::math::Vec3<float> *>(chunk->writeComponent(lpl::ecs::ComponentId::AABB));
            for (lpl::core::u32 i = 0; i < n; ++i)
            {
                if (pos)
                    pos[i] = {(nextRand() - 0.5f) * extent, (nextRand() - 0.5f) * extent, (nextRand() - 0.5f) * extent};
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

/**
 * @brief One step of the CPU physics backend over 10k bodies.
 *
 * The backend runs the real integrate → collide → sleep pipeline over the ECS chunks — unlike
 * WorldPartition::step(), whose CPU path is only a GPU-dispatch gateway and does no integration
 * work. The pipeline (sleep buffers, caches) is warmed up before timing.
 */
void benchmarkPhysics()
{
    lpl::ecs::Registry reg;
    const lpl::ecs::Archetype arch = makePhysicsArchetype();
    for (int i = 0; i < 10000; ++i)
    {
        [[maybe_unused]] auto e = reg.createEntity(arch);
    }
    seedPhysicsComponents(reg, 10000);

    lpl::physics::CpuPhysicsBackend backend{reg};
    [[maybe_unused]] auto ok = backend.init();

    for (int i = 0; i < 5; ++i) [[maybe_unused]]
        auto r = backend.step(0.0166f);

    lpl::bench::run("Physics step (10k, integrate+collide+sleep)",
                    [&]() { [[maybe_unused]] auto r = backend.step(0.0166f); });
}

/**
 * @brief The CPU physics step over 1k to 20k bodies.
 *
 * Fewer repetitions for the heavier counts keep total runtime sane.
 */
void benchmarkPhysicsScalability()
{
    constexpr lpl::core::u32 kCounts[] = {1000, 2000, 5000, 10000, 20000};

    std::printf("\n  --- Physics Scalability Sweep (CPU backend: integrate + O(n²) collide + sleep) ---\n");

    for (lpl::core::u32 count : kCounts)
    {
        lpl::ecs::Registry reg;
        const lpl::ecs::Archetype arch = makePhysicsArchetype();
        for (lpl::core::u32 i = 0; i < count; ++i)
        {
            [[maybe_unused]] auto e = reg.createEntity(arch);
        }
        seedPhysicsComponents(reg, count);

        lpl::physics::CpuPhysicsBackend backend{reg};
        [[maybe_unused]] auto ok = backend.init();
        for (int i = 0; i < 3; ++i) [[maybe_unused]]
            auto r = backend.step(0.0166f);

        char label[64];
        std::snprintf(label, sizeof(label), "Physics step (%uk entities)", count / 1000);

        lpl::bench::Config cfg;
        cfg.minReps = 5;
        cfg.maxReps = 200;
        cfg.targetTotalMs = 400.0;
        const lpl::bench::Result r =
            lpl::bench::run(label, [&]() { [[maybe_unused]] auto s = backend.step(0.0166f); }, cfg);

        const lpl::core::f64 ms = r.medianNs / 1e6;
        const lpl::core::f64 entPerSec = static_cast<lpl::core::f64>(count) / (r.medianNs / 1e9);
        std::printf("    -> Throughput: %.2e ent/sec  [%s]\n", entPerSec, lpl::bench::frameRateVerdict(ms));
    }
}

/**
 * @brief Building and populating a WorldPartition with 10k entities.
 *
 * Placement uses a deterministic LCG, so it is reproducible. Build + populate is self-contained
 * per repetition (state accumulates). This measures the spatial-index insertion path (Morton keying
 * + cell assignment); the actual physics integration is benchmarked separately via
 * CpuPhysicsBackend, since WorldPartition::step()'s CPU path is only a GPU-dispatch gateway and does
 * no integration work.
 */
void benchmarkWorldPartition()
{
    static constexpr int kEntityCount = 10000;

    auto makeArchetype = []() {
        lpl::ecs::Archetype arch;
        arch.add(lpl::ecs::ComponentId::Position);
        arch.add(lpl::ecs::ComponentId::Velocity);
        arch.add(lpl::ecs::ComponentId::Mass);
        arch.add(lpl::ecs::ComponentId::AABB);
        arch.add(lpl::ecs::ComponentId::Health);
        return arch;
    };

    auto placeAll = [](lpl::ecs::Registry &reg, lpl::ecs::WorldPartition &world, const lpl::ecs::Archetype &arch) {
        lpl::core::u32 seed = 12345;
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
            const lpl::math::Vec3<lpl::math::Fixed32> fixedPos{lpl::math::Fixed32::fromFloat(px),
                                                               lpl::math::Fixed32::fromFloat(py),
                                                               lpl::math::Fixed32::fromFloat(pz)};
            [[maybe_unused]] auto r = world.insertOrUpdate(entityId, fixedPos);
        }
    };

    const lpl::ecs::Archetype arch = makeArchetype();

    lpl::bench::run("WorldPartition build+insert 10k entities", [&]() {
        lpl::ecs::Registry reg;
        lpl::ecs::WorldPartition world(lpl::math::Fixed32::fromFloat(10.0f));
        placeAll(reg, world, arch);
        lpl::bench::doNotOptimize(world);
    });
}

/**
 * @brief FlatAtomicHashMap's insert, get, forEach, forEachParallel and churn over 16k keys.
 *
 * The churn run inserts every key, removes half and re-inserts them, which tests pool recycling.
 */
void benchmarkFlatAtomicHashMap()
{
    static constexpr lpl::core::u32 kPoolCap = 16384;

    lpl::bench::run("FlatAtomicHashMap insert 16k", []() {
        lpl::container::FlatAtomicHashMap<int> map(kPoolCap);
        for (lpl::core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.insert(static_cast<lpl::core::u64>(i + 1));
            lpl::bench::doNotOptimize(v);
        }
    });

    lpl::bench::run("FlatAtomicHashMap get 16k", []() {
        lpl::container::FlatAtomicHashMap<int> map(kPoolCap);
        for (lpl::core::u32 i = 0; i < kPoolCap; ++i)
            map.insert(static_cast<lpl::core::u64>(i + 1));
        for (lpl::core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.get(static_cast<lpl::core::u64>(i + 1));
            lpl::bench::doNotOptimize(v);
        }
    });

    lpl::bench::run("FlatAtomicHashMap forEach 16k", []() {
        lpl::container::FlatAtomicHashMap<int> map(kPoolCap);
        for (lpl::core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.insert(static_cast<lpl::core::u64>(i + 1));
            if (v)
                *v = static_cast<int>(i);
        }
        lpl::core::u64 sum = 0;
        map.forEach([&sum](int &val) { sum += static_cast<lpl::core::u64>(val); });
        lpl::bench::doNotOptimize(sum);
    });

    lpl::bench::run("FlatAtomicHashMap forEachParallel 16k", []() {
        lpl::container::FlatAtomicHashMap<int> map(kPoolCap);
        for (lpl::core::u32 i = 0; i < kPoolCap; ++i)
        {
            auto *v = map.insert(static_cast<lpl::core::u64>(i + 1));
            if (v)
                *v = static_cast<int>(i);
        }
        lpl::concurrency::ThreadPool pool{4};
        std::atomic<lpl::core::u64> sum{0};
        map.forEachParallel(
            pool, [&sum](int &val) { sum.fetch_add(static_cast<lpl::core::u64>(val), std::memory_order_relaxed); });
        lpl::bench::doNotOptimize(sum);
    });

    lpl::bench::run("FlatAtomicHashMap insert/remove churn 16k", []() {
        lpl::container::FlatAtomicHashMap<int> map(kPoolCap);
        for (lpl::core::u32 i = 0; i < kPoolCap; ++i)
            map.insert(static_cast<lpl::core::u64>(i + 1));
        for (lpl::core::u32 i = 0; i < kPoolCap / 2; ++i)
            map.remove(static_cast<lpl::core::u64>(i + 1));
        for (lpl::core::u32 i = 0; i < kPoolCap / 2; ++i)
            map.insert(static_cast<lpl::core::u64>(i + 1));
    });
}

void benchmarkThreadPool()
{
    lpl::concurrency::ThreadPool pool{4};

    lpl::bench::run("ThreadPool dispatch 10k tasks", [&]() {
        std::atomic<lpl::core::u64> counter{0};
        std::vector<std::future<void>> futures;
        futures.reserve(10000);
        for (int i = 0; i < 10000; ++i)
        {
            futures.push_back(pool.enqueue([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); }));
        }
        for (auto &f : futures)
            f.get();
        lpl::bench::doNotOptimize(counter);
    });
}

/**
 * @brief Allocator throughput: the engine's custom allocators vs the system malloc.
 *
 * Same workload (N fixed-size blocks) through three strategies:
 *   - Arena  — bump-pointer allocate, then a single O(1) mass reset;
 *   - Pool   — intrusive free-list, per-block acquire then per-block release;
 *   - malloc — the libc baseline, per-block malloc then free.
 *
 * This is the quantitative justification for shipping custom allocators.
 */
void benchmarkAllocators()
{
    constexpr int kN = 100000;
    struct alignas(16) Block64 {
        lpl::core::u64 data[8];
    };

    std::printf("\n  --- Allocator throughput (%dk x 64-byte blocks) ---\n", kN / 1000);

    std::vector<void *> ptrs(static_cast<lpl::core::usize>(kN), nullptr);

    lpl::bench::run("Arena  bump-alloc + reset", [&]() {
        lpl::memory::ArenaAllocator arena{static_cast<lpl::core::usize>(kN) * sizeof(Block64) + 4096};
        for (int i = 0; i < kN; ++i)
        {
            void *p = arena.allocate(sizeof(Block64), 16);
            lpl::bench::doNotOptimize(p);
        }
        arena.reset();
    });

    lpl::bench::run("Pool   acquire + release", [&]() {
        lpl::memory::PoolAllocator<Block64> pool{static_cast<lpl::core::usize>(kN)};
        for (int i = 0; i < kN; ++i)
            ptrs[static_cast<lpl::core::usize>(i)] = pool.acquire();
        for (int i = 0; i < kN; ++i)
            pool.release(static_cast<Block64 *>(ptrs[static_cast<lpl::core::usize>(i)]));
        lpl::bench::doNotOptimize(ptrs.data());
    });

    lpl::bench::run("malloc + free (libc baseline)", [&]() {
        for (int i = 0; i < kN; ++i)
        {
            ptrs[static_cast<lpl::core::usize>(i)] = std::malloc(sizeof(Block64));
            lpl::bench::doNotOptimize(ptrs[static_cast<lpl::core::usize>(i)]);
        }
        for (int i = 0; i < kN; ++i)
            std::free(ptrs[static_cast<lpl::core::usize>(i)]);
    });
}

/**
 * @brief SoA vs AoS memory layout benchmark.
 *
 * Demonstrates the cache-miss penalty of Array-of-Structs vs Structure-of-Arrays when iterating
 * over a single component (x-coordinate only): the AoS element stores (x,y,z) interleaved, the SoA
 * layout keeps three contiguous float arrays.
 */
void benchmarkSoA_vs_AoS()
{
    constexpr int kN = 1'000'000;
    std::printf("\n  --- SoA vs AoS layout (%dM elements, sum-x) ---\n", kN / 1'000'000);

    struct AoS_Vec3 {
        float x, y, z;
    };
    std::vector<AoS_Vec3> aos(kN);
    for (int i = 0; i < kN; ++i)
        aos[i] = {static_cast<float>(i), 1.0f, 2.0f};

    std::vector<float> xs(kN), ys(kN), zs(kN);
    for (int i = 0; i < kN; ++i)
    {
        xs[i] = static_cast<float>(i);
        ys[i] = 1.0f;
        zs[i] = 2.0f;
    }

    lpl::bench::run("Sum-x AoS (Vec3[]) 1M", [&]() {
        double acc = 0.0;
        for (int i = 0; i < kN; ++i)
            acc += static_cast<double>(aos[i].x);
        lpl::bench::doNotOptimize(acc);
    });

    lpl::bench::run("Sum-x SoA (float[]) 1M", [&]() {
        double acc = 0.0;
        for (int i = 0; i < kN; ++i)
            acc += static_cast<double>(xs[i]);
        lpl::bench::doNotOptimize(acc);
    });
}

/**
 * @brief Entity lookup latency: sparse set O(1) vs linear scan O(n).
 *
 * The slots are 14-bit. The lookups ask for the LAST 1000 slots inserted, to stress the worst case
 * of the linear scan.
 */
void benchmarkEntityLookup()
{
    constexpr lpl::core::u32 kN = 10000;
    std::printf("\n  --- Entity lookup: SparseSet O(1) vs linear scan O(n) ---\n");

    lpl::container::SparseSet<lpl::core::u32> sparseSet{1u << 14};
    std::vector<lpl::core::u32> denseVec;
    denseVec.reserve(kN);

    lpl::core::u32 seed = 42;
    for (lpl::core::u32 i = 0; i < kN; ++i)
    {
        seed = seed * 6364136223846793005u + 1442695040888963407u;
        const lpl::core::u32 slot = seed & 0x3FFF;
        sparseSet.insert(slot, i);
        denseVec.push_back(slot);
    }

    constexpr int kLookups = 1000;

    lpl::bench::run("SparseSet lookup 1k slots", [&]() {
        lpl::core::u32 sink = 0;
        for (int i = 0; i < kLookups; ++i)
        {
            const lpl::core::u32 slot = denseVec[kN - 1 - static_cast<lpl::core::u32>(i) % kN];
            if (const lpl::core::u32 *p = sparseSet.find(slot))
                sink = *p;
        }
        lpl::bench::doNotOptimize(sink);
    });

    lpl::bench::run("Linear scan lookup 1k slots in 10k vec", [&]() {
        lpl::core::u32 sink = 0;
        for (int i = 0; i < kLookups; ++i)
        {
            const lpl::core::u32 target = denseVec[kN - 1 - static_cast<lpl::core::u32>(i) % kN];
            for (lpl::core::u32 j = 0; j < kN; ++j)
            {
                if (denseVec[j] == target)
                {
                    sink = j;
                    break;
                }
            }
        }
        lpl::bench::doNotOptimize(sink);
    });
}

/**
 * @brief Collision detection: N² brute-force vs broad-phase (WorldPartition cells).
 *
 * The broad-phase is WorldPartition::queryRadius over the same boxes.
 */
void benchmarkCollisionBroadphase()
{
    constexpr int kN = 2000;
    std::printf("\n  --- Collision N\u00b2 vs broad-phase (%d entities) ---\n", kN);

    struct AABB {
        float xmin, xmax, ymin, ymax, zmin, zmax;
    };
    std::vector<AABB> boxes(kN);
    lpl::core::u32 rng = 99999;
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

    lpl::bench::run("N\u00b2 AABB collision check (2k)", [&]() {
        int pairs = 0;
        for (int a = 0; a < kN; ++a)
            for (int b = a + 1; b < kN; ++b)
                if (boxes[a].xmax >= boxes[b].xmin && boxes[b].xmax >= boxes[a].xmin &&
                    boxes[a].ymax >= boxes[b].ymin && boxes[b].ymax >= boxes[a].ymin &&
                    boxes[a].zmax >= boxes[b].zmin && boxes[b].zmax >= boxes[a].zmin)
                    ++pairs;
        lpl::bench::doNotOptimize(pairs);
    });

    lpl::ecs::WorldPartition world(lpl::math::Fixed32::fromFloat(10.0f));
    lpl::ecs::Registry reg;
    lpl::ecs::Archetype arch;
    arch.add(lpl::ecs::ComponentId::Position);
    arch.add(lpl::ecs::ComponentId::AABB);

    for (int i = 0; i < kN; ++i)
    {
        if (auto e = reg.createEntity(arch))
        {
            lpl::math::Vec3<lpl::math::Fixed32> pos{lpl::math::Fixed32::fromFloat(boxes[i].xmin + 0.5f),
                                                    lpl::math::Fixed32::fromFloat(boxes[i].ymin + 0.5f),
                                                    lpl::math::Fixed32::fromFloat(boxes[i].zmin + 0.5f)};
            [[maybe_unused]] auto r = world.insertOrUpdate(e.value(), pos);
        }
    }

    lpl::bench::run("Broad-phase queryRadius (2k, r=10)", [&]() {
        lpl::core::u32 total = 0;
        for (int i = 0; i < kN; ++i)
        {
            std::vector<lpl::ecs::EntityId> hits;
            const lpl::math::Vec3<lpl::math::Fixed32> center{lpl::math::Fixed32::fromFloat(boxes[i].xmin + 0.5f),
                                                             lpl::math::Fixed32::fromFloat(boxes[i].ymin + 0.5f),
                                                             lpl::math::Fixed32::fromFloat(boxes[i].zmin + 0.5f)};
            world.queryRadius(center, lpl::math::Fixed32::fromFloat(10.0f), hits);
            total += static_cast<lpl::core::u32>(hits.size());
        }
        lpl::bench::doNotOptimize(total);
    });
}

class CountingTransport final : public lpl::net::transport::ITransport {
public:
    std::atomic<lpl::core::u64> bytes{0};
    std::atomic<lpl::core::u64> packets{0};

    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "CountingTransport"; }
    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }
    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> d, const lpl::net::Endpoint *) override
    {
        bytes.fetch_add(d.size(), std::memory_order_relaxed);
        packets.fetch_add(1, std::memory_order_relaxed);
        return static_cast<lpl::core::u32>(d.size());
    }
    void reset()
    {
        bytes.store(0, std::memory_order_relaxed);
        packets.store(0, std::memory_order_relaxed);
    }
};

struct NetWorld {
    lpl::ecs::Registry registry;
    lpl::ecs::WorldPartition world;
    lpl::net::session::SessionManager sessions;
    std::vector<lpl::ecs::EntityId> ids;

    explicit NetWorld(lpl::math::Fixed32 cell) : world{cell, 1u << 20} {}
};

/**
 * @brief Places @p n entities on a square grid and records their ids in @p nw; the first @p
 *        clients of them are also bound as session avatars, each with its own endpoint.
 *
 * The avatars are spread across the whole grid so each client's radius sees a different local
 * neighbourhood, not all the same cluster.
 */
void seedNetWorld(NetWorld &nw, lpl::core::u32 n, lpl::core::u32 clients, float spacing)
{
    const lpl::ecs::ComponentId comps[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::AABB,
                                           lpl::ecs::ComponentId::Health};
    const lpl::ecs::Archetype arch{comps};
    const auto side = static_cast<lpl::core::u32>(std::sqrt(static_cast<double>(n))) + 1;

    for (lpl::core::u32 i = 0; i < n; ++i)
    {
        auto e = nw.registry.createEntity(arch);
        if (!e)
            continue;
        const auto id = e.value();
        nw.ids.push_back(id);

        const float x = static_cast<float>(i % side) * spacing;
        const float z = static_cast<float>(i / side) * spacing;
        using Vec = lpl::math::Vec3<lpl::math::Fixed32>;
        const Vec pos{lpl::math::Fixed32::fromFloat(x), lpl::math::Fixed32::zero(), lpl::math::Fixed32::fromFloat(z)};
        const Vec one{lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one()};

        auto ref = nw.registry.resolve(id);
        auto &part = nw.registry.getOrCreatePartition(arch);
        auto &chunk = *part.chunks()[ref.value().chunkIndex];
        const lpl::core::u32 li = ref.value().localIndex;
        for (auto which : {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::AABB})
        {
            const Vec v = (which == lpl::ecs::ComponentId::Position) ? pos : one;
            if (auto *w = static_cast<Vec *>(chunk.writeComponent(which)))
                w[li] = v;
            if (auto *r = const_cast<Vec *>(static_cast<const Vec *>(chunk.readComponent(which))))
                r[li] = v;
        }
        [[maybe_unused]] auto s = nw.world.insertOrUpdate(id, pos);
    }

    for (lpl::core::u32 c = 0; c < clients && c < nw.ids.size(); ++c)
    {
        auto joined = nw.sessions.connect(c + 1);
        if (!joined)
            continue;
        auto *sess = joined.value();
        sess->setAddress(lpl::net::Endpoint::fromOctets(10, 0, static_cast<lpl::core::u8>(c >> 8),
                                                        static_cast<lpl::core::u8>(c & 0xFF),
                                                        static_cast<lpl::core::u16>(40000 + (c & 0x3FFF))));
        sess->bindEntity(nw.ids[(static_cast<lpl::core::u64>(c) * nw.ids.size() / clients) % nw.ids.size()]);
    }
}

/**
 * @brief Networking: the two load walls of chapter 6.2.10, measured on this machine.
 *
 * The point is not a headline number but to *show* the O(N²) send wall of a full broadcast, and
 * how interest management turns it into O(N·k). A byte-counting transport stands in for the
 * socket, so this measures the serialiser and the bandwidth it produces, independent of loopback
 * noise.
 *
 * A sparse map: 1000 entities on a ~790-unit grid (spacing 25), interest radius 40 → each client
 * sees only a handful of neighbours (k ≪ N), the regime interest management is actually for. A
 * dense map where the radius covers half the world would make AOI both marginal on bandwidth and
 * far slower than a flat broadcast — the parameters matter, and are reported.
 *
 * The interest-managed pass runs one warmup tick so per-client interest sets exist, then measures
 * a tick where every entity is a fresh full send (a keyframe interval of 1 → worst-case AOI). At
 * ~32 bytes per entity on the wire, the bytes a tick sends give the average neighbours each client
 * received.
 *
 * Receive wall: how fast one thread parses + dispatches inbound INPUT packets — what a server
 * actually decodes from clients (a small WASD + neural payload), not the heavy state snapshots it
 * sends. The §6.1 policy caps a single thread at 256 pkt/tick x 144 Hz = 36,864 pkt/s; this is the
 * CPU ceiling underneath that cap. The payload is the entity id (u32), 4 key events (u16 count,
 * then a u16 key and a bool each: WASD), 0 axes (u8), the neural alpha, beta and concentration
 * (three floats) and the blink (bool).
 */
void benchmarkNetworking()
{
    std::printf("\n  --- Networking: broadcast send wall vs interest management (§6.2.10) ---\n");
    constexpr lpl::core::u32 kTickRate = 144;
    constexpr lpl::core::u32 kEntities = 1000;
    constexpr lpl::core::u32 kClientCounts[] = {10, 50, 100, 500, 1000};
    const float dt = 1.0f / static_cast<float>(kTickRate);

    constexpr float kSpacing = 25.0f;

    std::printf("    %6s | %10s | %12s | %10s\n", "clients", "bytes/tick", "out Gbit/s", "us/tick");
    std::printf("    Full broadcast (every entity to every client — O(clients x N)):\n");
    for (lpl::core::u32 c : kClientCounts)
    {
        NetWorld nw{lpl::math::Fixed32::fromFloat(10.0f)};
        seedNetWorld(nw, kEntities, c, kSpacing);
        auto tr = std::make_shared<CountingTransport>();
        lpl::engine::systems::BroadcastSystem full{nw.sessions, tr, nw.world, nw.registry};

        tr->reset();
        full.execute(dt);
        const lpl::core::u64 bytesPerTick = tr->bytes.load();
        const double gbit = static_cast<double>(bytesPerTick) * 8.0 * kTickRate / 1e9;

        lpl::bench::Config cfg;
        cfg.minReps = 3;
        cfg.maxReps = 100;
        cfg.targetTotalMs = 300.0;
        const std::string label = std::format("full-broadcast {} clients", c);
        const lpl::bench::Result r = lpl::bench::run(label.c_str(), [&]() { full.execute(dt); }, cfg);
        std::printf("    %6u | %10llu | %11.2f | %10.1f\n", c, static_cast<unsigned long long>(bytesPerTick), gbit,
                    r.medianNs / 1e3);
    }

    std::printf("    Interest-managed (each client only its neighbours — O(clients x k)):\n");
    std::printf("    %6s | %10s | %12s | %10s | %8s\n", "clients", "bytes/tick", "out Gbit/s", "us/tick", "k (ent/cl)");
    const auto radius = lpl::math::Fixed32::fromFloat(40.0f);
    for (lpl::core::u32 c : kClientCounts)
    {
        NetWorld nw{lpl::math::Fixed32::fromFloat(10.0f)};
        seedNetWorld(nw, kEntities, c, kSpacing);
        auto tr = std::make_shared<CountingTransport>();
        lpl::engine::systems::AoiBroadcastSystem aoi{nw.sessions, tr, nw.world, nw.registry, radius, 1};

        aoi.execute(dt);
        tr->reset();
        aoi.execute(dt);
        const lpl::core::u64 bytesPerTick = tr->bytes.load();
        const double gbit = static_cast<double>(bytesPerTick) * 8.0 * kTickRate / 1e9;
        const double k = static_cast<double>(bytesPerTick) / static_cast<double>(c) / 32.0;

        lpl::bench::Config cfg;
        cfg.minReps = 3;
        cfg.maxReps = 100;
        cfg.targetTotalMs = 300.0;
        const std::string label = std::format("aoi-broadcast {} clients", c);
        const lpl::bench::Result r = lpl::bench::run(label.c_str(), [&]() { aoi.execute(dt); }, cfg);
        std::printf("    %6u | %10llu | %11.2f | %10.1f | %8.1f\n", c, static_cast<unsigned long long>(bytesPerTick),
                    gbit, r.medianNs / 1e3, k);
    }

    std::printf("    Receive/decode throughput (single thread, small InputPayload packets):\n");
    {
        lpl::net::protocol::Bitstream stream;
        stream.writeU32(1);
        stream.writeU16(4);
        for (int k = 0; k < 4; ++k)
        {
            stream.writeU16(static_cast<lpl::core::u16>('W' + k));
            stream.writeBool(true);
        }
        stream.writeU8(0);
        stream.writeFloat(0.3f);
        stream.writeFloat(0.5f);
        stream.writeFloat(0.7f);
        stream.writeBool(false);

        std::vector<lpl::core::byte> datagram(sizeof(lpl::net::protocol::PacketHeader) + stream.data().size());
        auto &h = *reinterpret_cast<lpl::net::protocol::PacketHeader *>(datagram.data());
        h.magic = lpl::net::protocol::kProtocolMagic;
        h.version = lpl::net::protocol::kProtocolVersion;
        h.type = lpl::net::protocol::PacketType::InputPayload;
        h.flags = 0;
        h.padding = 0;
        h.sequence = 0;
        h.payloadSize = static_cast<lpl::core::u32>(stream.data().size());
        std::memcpy(datagram.data() + sizeof(h), stream.data().data(), stream.data().size());

        const lpl::net::Endpoint from = lpl::net::Endpoint::fromOctets(10, 0, 0, 1, 40000);
        lpl::bench::Config cfg;
        cfg.minReps = 5;
        cfg.maxReps = 500;
        cfg.targetTotalMs = 300.0;
        const lpl::bench::Result r = lpl::bench::run(
            "decode-10k-input-packets",
            [&]() {
                lpl::engine::EventQueues q;
                for (int p = 0; p < 10000; ++p)
                {
                    lpl::net::protocol::PacketHeader ph{};
                    std::span<const lpl::core::byte> payload;
                    if (lpl::engine::detail::parsePacket(datagram, ph, payload))
                        lpl::engine::detail::dispatchPacket(ph, payload, from, q);
                }
            },
            cfg);
        const double pktPerSec = 10000.0 / (r.medianNs / 1e9);
        std::printf("    -> %.2e input packets/sec decoded on one thread (policy cap %u pkt/s at 256/tick x %u Hz)\n",
                    pktPerSec, 256u * kTickRate, kTickRate);
    }
}

struct Section {
    std::string_view name;
    void (*run)();
};

constexpr std::array kSections{
    Section{"arena",       benchmarkArena                },
    Section{"fixed",       benchmarkFixedMath            },
    Section{"morton",      benchmarkMorton               },
    Section{"allocators",  benchmarkAllocators           },
    Section{"trig",        benchmarkTrigonometry         },
    Section{"registry",    benchmarkRegistry             },
    Section{"physics",     benchmarkPhysics              },
    Section{"scalability", benchmarkPhysicsScalability   },
    Section{"partition",   benchmarkWorldPartition       },
    Section{"hashmap",     benchmarkFlatAtomicHashMap    },
    Section{"threadpool",  benchmarkThreadPool           },
    Section{"ring",        lpl::bench::runRingBenchmarks },
    Section{"soa",         benchmarkSoA_vs_AoS           },
    Section{"lookup",      benchmarkEntityLookup         },
    Section{"broadphase",  benchmarkCollisionBroadphase  },
    Section{"net",         benchmarkNetworking           },
    Section{"voxel",       lpl::bench::runVoxelBenchmarks},
};

[[nodiscard]] bool isSectionName(std::string_view name)
{
    return std::ranges::any_of(kSections, [name](const Section &section) { return section.name == name; });
}

/** Width the help wraps its descriptions at. */
constexpr std::size_t kHelpWidth = 100;

void printWrapped(std::FILE *stream, std::size_t indent, std::string_view text)
{
    std::size_t column = indent;
    while (!text.empty())
    {
        const std::size_t wordEnd = std::min(text.find(' '), text.size());
        const std::string_view word = text.substr(0, wordEnd);
        if (column > indent && column + 1 + word.size() > kHelpWidth)
        {
            std::fprintf(stream, "\n%*s", static_cast<int>(indent), "");
            column = indent;
        }
        else if (column > indent)
        {
            std::fputc(' ', stream);
            ++column;
        }
        std::fprintf(stream, "%.*s", static_cast<int>(word.size()), word.data());
        column += word.size();
        text.remove_prefix(std::min(wordEnd + 1, text.size()));
    }
    std::fputc('\n', stream);
}

consteval std::size_t longestRowFieldKey()
{
    std::size_t longest = 0;
    for (const lpl::bench::JsonRowField &field : lpl::bench::kJsonRowFields)
        longest = std::max(longest, field.key.size());
    return longest;
}

constexpr char kSynopsis[] = "usage: lpl-benchmark [-o|--only SECTION] [--json PATH] [-h|--help]\n";

void printUsage(std::FILE *stream)
{
    std::fputs(kSynopsis, stream);
    std::fputs("  Measures the engine's building blocks and prints the results on stdout.\n"
               "\n"
               "  -o, --only SECTION  run that section alone; default: every section\n"
               "  --json PATH         also write one JSON row per measurement to PATH, which is created or\n"
               "                      overwritten; default: no file\n"
               "  -h, --help          print this help\n"
               "\n"
               "Sections, in the order they run:\n",
               stream);
    for (const Section &section : kSections)
        std::fprintf(stream, "  %.*s\n", static_cast<int>(section.name.size()), section.name.data());

    constexpr std::size_t kKeyWidth = longestRowFieldKey();
    std::fputs("\nFields of a JSON row, one object per line, in this order:\n", stream);
    for (const lpl::bench::JsonRowField &field : lpl::bench::kJsonRowFields)
    {
        std::fprintf(stream, "  %-*.*s  ", static_cast<int>(kKeyWidth), static_cast<int>(field.key.size()),
                     field.key.data());
        printWrapped(stream, kKeyWidth + 4u, field.meaning);
    }

    const std::string_view exampleSection = kSections.front().name;
    std::fprintf(stream,
                 "\nExit status: 0 when every measurement ran and, with --json, every row is in PATH; 1 when a\n"
                 "row is missing from PATH (a write failed, or two measurements share a label); 2 on a usage\n"
                 "error, or when PATH cannot be opened.\n"
                 "\n"
                 "Examples:\n"
                 "  lpl-benchmark --only %.*s\n"
                 "  lpl-benchmark --json before.jsonl\n"
                 "  tools/bench-diff.py before.jsonl after.jsonl    compares two runs row by row\n",
                 static_cast<int>(exampleSection.size()), exampleSection.data());
}

struct Arguments {
    std::optional<std::string_view> onlySection;
    std::optional<std::string_view> jsonPath;
    bool helpRequested{false};
};

[[nodiscard]] std::expected<Arguments, std::string> parseArguments(int argc, char *argv[])
{
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument = argv[i];
        if (argument == "--help" || argument == "-h")
            return Arguments{.onlySection = std::nullopt, .jsonPath = std::nullopt, .helpRequested = true};
        const bool namesSection = argument == "--only" || argument == "-o";
        const bool namesPath = argument == "--json";
        if (!namesSection && !namesPath)
            return std::unexpected(std::format("unknown argument '{}'", argument));
        if (i + 1 == argc)
            return std::unexpected(std::format("'{}' needs a {}", argument, namesSection ? "SECTION" : "PATH"));
        const std::string_view value = argv[++i];
        if (namesPath)
        {
            arguments.jsonPath = value;
            continue;
        }
        if (!isSectionName(value))
            return std::unexpected(std::format("unknown section '{}'", value));
        arguments.onlySection = value;
    }
    return arguments;
}

[[nodiscard]] std::expected<std::optional<lpl::bench::JsonRowFile>, std::string>
openRowsFile(std::optional<std::string_view> path, const lpl::bench::SystemInfo &system)
{
    if (!path)
        return std::optional<lpl::bench::JsonRowFile>{};
    std::expected<lpl::bench::JsonRowFile, std::string> file =
        lpl::bench::JsonRowFile::create(std::string{*path}, system);
    if (!file)
        return std::unexpected(std::move(file.error()));
    return std::optional<lpl::bench::JsonRowFile>{*std::move(file)};
}

[[nodiscard]] int reportRows(const std::optional<lpl::bench::JsonRowFile> &rowsFile)
{
    if (!rowsFile)
    {
        std::printf("\nDone.\n");
        return 0;
    }
    std::printf("\nDone. Rows written to %s: %zu\n", rowsFile->path().c_str(), rowsFile->rowCount());
    if (!rowsFile->failure())
        return 0;
    std::fprintf(stderr, "lpl-benchmark: --json: %s\n", rowsFile->failure()->c_str());
    return 1;
}

} // namespace

/**
 * @brief LplPlugin benchmark entry-point: defines the individual benchmark
 *        cases and runs them. The measurement machinery (timing, statistics,
 *        DCE barriers) and host introspection live in the lpl::bench module.
 */
int main(int argc, char *argv[])
{
    const std::expected<Arguments, std::string> arguments = parseArguments(argc, argv);
    if (!arguments)
    {
        std::fprintf(stderr,
                     "lpl-benchmark: %s\n%sRun 'lpl-benchmark --help' for the sections, the fields of a JSON row "
                     "and the exit status.\n",
                     arguments.error().c_str(), kSynopsis);
        return 2;
    }
    if (arguments->helpRequested)
    {
        printUsage(stdout);
        return 0;
    }

    const lpl::bench::SystemInfo system = lpl::bench::collectSystemInfo();
    std::expected<std::optional<lpl::bench::JsonRowFile>, std::string> rowsFile =
        openRowsFile(arguments->jsonPath, system);
    if (!rowsFile)
    {
        std::fprintf(stderr, "lpl-benchmark: --json: %s\n", rowsFile.error().c_str());
        return 2;
    }
    if (*rowsFile)
        lpl::bench::writeRowsTo(&**rowsFile);

    lpl::core::Log::info("=== LplPlugin Benchmark ===");
    std::printf("\n");

    lpl::bench::printSystemInfo(system);
    lpl::bench::printLegend();

    for (const Section &section : kSections)
        if (!arguments->onlySection || *arguments->onlySection == section.name)
            section.run();

    lpl::bench::writeRowsTo(nullptr);
    return reportRows(*rowsFile);
}
