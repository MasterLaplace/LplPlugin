/**
 * @file main.cpp
 * @brief LplPlugin benchmark entry-point.
 *
 * @author MasterLaplace
 * @version 0.1.0
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

using namespace lpl;

namespace {

template <typename Fn>
core::f64 benchmarkMs(const char* label, Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const core::f64 ms = std::chrono::duration<core::f64, std::milli>(end - start).count();
    std::printf("  %-36s %10.3f ms\n", label, ms);
    return ms;
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

    std::printf("\nDone.\n");
    return 0;
}
