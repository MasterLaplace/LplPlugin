#include <lpl/engine/World.hpp>
#include <lpl/engine/systems/PhysicsSystem.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/samples/CubePile.hpp>
#include <lpl/testing/Test.hpp>

#include <new>

LPL_TEST_SUITE(bytes_moved);

namespace {

/**
 * @brief Most bytes the server's tick may copy from back to front buffers.
 *
 * The 1024 bodies of the cube pile carry a Position, a Velocity and an AABB of 12 bytes and a Mass
 * of 4. The physics writes the first two and only reads the others, so a swap copies 24 bytes a
 * body, 24,576 a tick, where it copied all 40 before. A ratchet, as the zero-allocation gate is:
 * a change that moves fewer bytes lowers it in the same pull request, and one that moves more
 * fails here first.
 */
constexpr lpl::core::u32 kSwapBytesPerTickBudget = 24576u;

/** @brief Ticks run before the count starts, and ticks counted. */
constexpr lpl::core::u32 kWarmTicks = 1u;
constexpr lpl::core::u32 kCountedTicks = 2u;

/** @brief The step the server runs at, 60 Hz. */
constexpr lpl::core::f32 kStep = 1.0f / 60.0f;

/**
 * @class ServerWorld
 * @brief The world the server profile runs, wired as Engine::run wires it: the cube pile on the
 *        World's registry, the engine's physics system, and the buffer swap after the physics
 *        phase.
 */
class ServerWorld final : public lpl::engine::World {
public:
    ServerWorld() : _pile{registry()} {}

    /** @brief Seeds the pile and registers what a tick runs; false when the system graph does not build. */
    [[nodiscard]] bool setUp()
    {
        _pile.init();
        enableSpatialPartition(lpl::math::Fixed32::fromInt(10), lpl::samples::CubePile::count());
        _physicsBackend = lpl::pmr::make_unique<lpl::physics::CpuPhysicsBackend>(registry());
        (void) _physicsBackend->init();
        (void) scheduler().registerSystem(lpl::pmr::make_unique<lpl::engine::systems::PhysicsSystem>(
            *spatialPartition(), *_physicsBackend, registry()));
        scheduler().setPhaseCallback(lpl::ecs::SchedulePhase::Physics, [this]() { registry().swapAllBuffers(); });
        return build().has_value();
    }

private:
    lpl::samples::CubePile _pile;
    lpl::pmr::unique_ptr<lpl::physics::CpuPhysicsBackend> _physicsBackend;
};

/** @brief Storage of the world, in BSS: a kernel stack cannot hold 1024 bodies. */
alignas(ServerWorld) unsigned char gWorldStorage[sizeof(ServerWorld)];

} // namespace

/**
 * @brief The server's tick moves no more bytes through the double buffer than its budget, and says
 *        how many, per component.
 *
 * @details Counted, not timed: the figure is exact and the same under TCG, KVM and real hardware,
 *          which no clock can be.
 */
LPL_TEST(the_server_tick_stays_within_its_swap_budget)
{
    ServerWorld *world = ::new (static_cast<void *>(gWorldStorage)) ServerWorld();

    if (test.check(world->setUp(), "the server world builds its system graph"))
    {
        for (lpl::core::u32 tick = 0u; tick < kWarmTicks; ++tick)
            world->onFixedStep(kStep);

        const lpl::ecs::Registry &registry = world->registry();
        const lpl::core::u64 before = registry.swappedBytes();
        const lpl::core::u64 positionBefore = registry.swappedBytes(lpl::ecs::ComponentId::Position);
        for (lpl::core::u32 tick = 0u; tick < kCountedTicks; ++tick)
            world->onFixedStep(kStep);
        const lpl::core::u64 perTick = (registry.swappedBytes() - before) / kCountedTicks;
        const lpl::core::u64 positionPerTick =
            (registry.swappedBytes(lpl::ecs::ComponentId::Position) - positionBefore) / kCountedTicks;

        test.check(perTick > 0u, "a tick swaps its buffers");
        test.check(perTick <= kSwapBytesPerTickBudget, "and copies no more bytes than its budget");
        test.measure("swap_bytes_per_tick", static_cast<lpl::core::u32>(perTick));
        test.measure("swap_bytes_budget", kSwapBytesPerTickBudget);
        test.measure("swap_bytes_position_per_tick", static_cast<lpl::core::u32>(positionPerTick));
    }
    world->~ServerWorld();
}
