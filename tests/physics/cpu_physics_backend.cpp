#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/testing/Test.hpp>

#include <span>

LPL_TEST_SUITE(cpu_physics_backend);

namespace {

using PartitionSpan = std::span<const lpl::pmr::unique_ptr<lpl::ecs::Partition>>;
using FixedVector = lpl::math::Vec3<lpl::math::Fixed32>;

/**
 * @brief What one step of gravity left: how many bodies it moved, the first one's height and
 *        vertical speed, and whether every body fell.
 */
struct Fall {
    lpl::core::u32 stepped = 0u;
    lpl::math::Fixed32 height{};
    lpl::math::Fixed32 speed{};
    bool everyBodyFell = true;
};

/**
 * @brief Puts every body at rest at a height of 100 with a unit mass.
 *
 * @details Position and velocity go into the buffer the integrator writes; the mass goes into both
 *          buffers, because the integrator reads it from the other one.
 *
 * @return Bodies seeded.
 */
lpl::core::u32 seedAtRest(PartitionSpan partitions)
{
    lpl::core::u32 seeded = 0u;

    for (const auto &partition : partitions)
    {
        for (const auto &chunk : partition->chunks())
        {
            auto *positions = static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            auto *velocities = static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Velocity));
            auto *masses = static_cast<lpl::math::Fixed32 *>(chunk->writeComponent(lpl::ecs::ComponentId::Mass));
            auto *readMasses = static_cast<lpl::math::Fixed32 *>(
                const_cast<void *>(chunk->readComponent(lpl::ecs::ComponentId::Mass)));

            if (positions == nullptr || velocities == nullptr)
                continue;
            for (lpl::core::u32 index = 0u; index < chunk->count(); ++index)
            {
                positions[index] = FixedVector{lpl::math::Fixed32::fromInt(0), lpl::math::Fixed32::fromInt(100),
                                               lpl::math::Fixed32::fromInt(0)};
                velocities[index] = FixedVector{lpl::math::Fixed32::fromInt(0), lpl::math::Fixed32::fromInt(0),
                                                lpl::math::Fixed32::fromInt(0)};
                if (masses != nullptr)
                    masses[index] = lpl::math::Fixed32::one();
                if (readMasses != nullptr)
                    readMasses[index] = lpl::math::Fixed32::one();
                ++seeded;
            }
        }
    }
    return seeded;
}

[[nodiscard]] Fall readFall(PartitionSpan partitions)
{
    Fall fall;

    for (const auto &partition : partitions)
    {
        for (const auto &chunk : partition->chunks())
        {
            auto *positions = static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            auto *velocities = static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Velocity));

            if (positions == nullptr || velocities == nullptr)
                continue;
            for (lpl::core::u32 index = 0u; index < chunk->count(); ++index)
            {
                if (fall.stepped == 0u)
                {
                    fall.height = positions[index].y;
                    fall.speed = velocities[index].y;
                }
                fall.everyBodyFell = fall.everyBodyFell && positions[index].y < lpl::math::Fixed32::fromInt(100) &&
                                     velocities[index].y < lpl::math::Fixed32::fromInt(0);
                ++fall.stepped;
            }
        }
    }
    return fall;
}

} // namespace

/**
 * @brief Gate P1 physics: one 1/60 s step of gravity over three bodies at rest gives the same
 *        Fixed32 height and speed on both targets.
 */
LPL_TEST(one_step_of_gravity_gives_the_same_words)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                lpl::ecs::ComponentId::Mass};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;

    for (lpl::core::u32 body = 0u; body < 3u; ++body)
        (void) registry.createEntity(archetype);

    const PartitionSpan partitions = registry.partitions();

    if (!test.check(seedAtRest(partitions) == 3u, "three bodies are put at rest"))
        return;

    lpl::physics::CpuPhysicsBackend backend{registry};

    (void) backend.init();
    test.check(backend.step(1.0f / 60.0f).has_value(), "the backend steps");

    const Fall fall = readFall(partitions);

    test.check(fall.stepped == 3u, "the step moves the three bodies");
    test.check(fall.everyBodyFell, "and every one falls, with a downward speed");

    test.measureHexadecimal("height", static_cast<lpl::core::u32>(fall.height.raw()));
    test.measureHexadecimal("speed", static_cast<lpl::core::u32>(fall.speed.raw()));
}
