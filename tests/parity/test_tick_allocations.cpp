/**
 * @file test_tick_allocations.cpp
 * @brief Locates heap allocations performed by the authoritative fixed step.
 *
 * The kernel's real-time guard proves an allocating tick exists (kmalloc refuses
 * inside the section and the count moves) but cannot say WHERE: ring 0 has no
 * backtrace. This host harness runs the very same code — the World's scheduler
 * driving the engine's built-in systems::PhysicsSystem over the World's
 * registry, exactly as Engine::init wires it when Config::enablePhysics is on —
 * with a global operator new that counts, and dumps the call stack of every
 * distinct site that allocates once the loop is warm. Same call sites, same
 * order, so a site fixed here is a site fixed in ring 0.
 *
 * Not a parity test: it folds nothing and must never gate determinism.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-21
 * @copyright MIT License
 */

#include <lpl/engine/Engine.hpp>
#include <lpl/engine/World.hpp>
#include <lpl/engine/systems/PhysicsSystem.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/samples/CubePileWorld.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <new>

namespace {

bool gCounting = false;
unsigned long gAllocations = 0;

/// Distinct backtraces already reported, so a site is printed once, not 1024 times.
constexpr int kFrames = 12;
constexpr int kMaxSites = 64;
void *gSites[kMaxSites][kFrames];
int gSiteDepth[kMaxSites];
unsigned long gSiteHits[kMaxSites];
int gSiteCount = 0;

void record()
{
    void *frames[kFrames];
    const int depth = ::backtrace(frames, kFrames);

    for (int i = 0; i < gSiteCount; ++i)
    {
        if (gSiteDepth[i] == depth && std::memcmp(gSites[i], frames, sizeof(void *) * static_cast<size_t>(depth)) == 0)
        {
            ++gSiteHits[i];
            return;
        }
    }
    if (gSiteCount == kMaxSites)
        return;

    std::memcpy(gSites[gSiteCount], frames, sizeof(void *) * static_cast<size_t>(depth));
    gSiteDepth[gSiteCount] = depth;
    gSiteHits[gSiteCount] = 1;
    ++gSiteCount;
}

void report()
{
    for (int i = 0; i < gSiteCount; ++i)
    {
        std::printf("\n--- site %d (%lu allocations) ---\n", i, gSiteHits[i]);
        char **symbols = ::backtrace_symbols(gSites[i], gSiteDepth[i]);
        // Frames 0..1 are this recorder and operator new itself.
        for (int f = 2; f < gSiteDepth[i]; ++f)
            std::printf("  %s\n", symbols[f]);
        std::free(symbols);
    }
}

} // namespace

void *operator new(std::size_t size)
{
    if (gCounting)
    {
        // Suspend counting while backtrace()/malloc do their own work.
        gCounting = false;
        ++gAllocations;
        record();
        gCounting = true;
    }
    void *p = std::malloc(size ? size : 1);
    // The engine builds -fno-exceptions, so a failed new aborts rather than throws.
    if (p == nullptr)
        std::abort();
    return p;
}

void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

namespace {

/// The cube pile without the display half: same registry, scheduler and
/// built-in physics wiring as samples::CubePileWorld under Config::enablePhysics
/// (see Engine::init), minus the surface query onInit would need.
class HeadlessCubePileWorld final : public lpl::engine::World {
public:
    HeadlessCubePileWorld() : _cube{registry()} {}

    void setUp()
    {
        _cube.init();

        enableSpatialPartition(lpl::math::Fixed32::fromFloat(10.0f), 1024u);
        _physicsBackend = lpl::pmr::make_unique<lpl::physics::CpuPhysicsBackend>(registry());
        (void) _physicsBackend->init();
        (void) scheduler().registerSystem(lpl::pmr::make_unique<lpl::engine::systems::PhysicsSystem>(
            *spatialPartition(), *_physicsBackend, registry()));
    }

private:
    lpl::samples::CubePile _cube;
    lpl::pmr::unique_ptr<lpl::physics::CpuPhysicsBackend> _physicsBackend;
};

/// 1024 entities live here, not on the stack (the kernel path has the same rule).
HeadlessCubePileWorld gWorld;

} // namespace

int main()
{
    gWorld.setUp();
    if (!gWorld.build())
    {
        std::puts("scheduler graph build failed");
        return 1;
    }

    constexpr lpl::core::f32 kDt = 1.0f / 60.0f;

    // Warm-up: the first steps legitimately grow amortised buffers once.
    for (int i = 0; i < 30; ++i)
        gWorld.onFixedStep(kDt);

    gCounting = true;
    constexpr int kMeasuredSteps = 30;
    for (int i = 0; i < kMeasuredSteps; ++i)
        gWorld.onFixedStep(kDt);
    gCounting = false;

    std::printf("allocations over %d warm steps: %lu (%.1f per step), %d distinct sites\n", kMeasuredSteps,
                gAllocations, static_cast<double>(gAllocations) / kMeasuredSteps, gSiteCount);
    report();
    return gAllocations == 0 ? 0 : 2;
}
