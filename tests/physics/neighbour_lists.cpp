#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/testing/Test.hpp>

#include <span>

LPL_TEST_SUITE(neighbour_lists);

namespace {

using PartitionSpan = std::span<const lpl::pmr::unique_ptr<lpl::ecs::Partition>>;
using FixedVector = lpl::math::Vec3<lpl::math::Fixed32>;

/** Bodies in the row: past the 32 that the resolver pairs all together, so the lists serve. */
constexpr lpl::core::u32 kBodies = 40u;

/**
 * @brief Where body @p index starts along x, in raw Fixed32.
 *
 * Body 0 at 0. Body 1 three margins of 1/64 past touching it, so outside its list. Body 2 a
 * quarter unit into body 1, which it pushes into body 0 during the first iteration. The others
 * three units apart from x = 100, touching nothing.
 */
[[nodiscard]] lpl::core::i32 startAlongX(lpl::core::u32 index) noexcept
{
    constexpr lpl::core::i32 kUnit = 65536;
    constexpr lpl::core::i32 kMargin = 1024;
    if (index == 0u)
        return 0;
    if (index == 1u)
        return kUnit + 3 * kMargin;
    if (index == 2u)
        return 2 * kUnit + 3 * kMargin - kUnit / 4;
    return (100 + 3 * static_cast<lpl::core::i32>(index)) * kUnit;
}

/**
 * @brief Puts the row of unit boxes of unit mass at rest at a height of 100.
 *
 * @details Position and velocity go into the buffer the integrator writes; the mass and the box go
 *          into both buffers, because the resolver reads them from the other one.
 *
 * @return Bodies seeded.
 */
lpl::core::u32 seedRow(PartitionSpan partitions)
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
            auto *boxes = static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::AABB));
            auto *readBoxes =
                static_cast<FixedVector *>(const_cast<void *>(chunk->readComponent(lpl::ecs::ComponentId::AABB)));

            if (positions == nullptr || velocities == nullptr || masses == nullptr || readMasses == nullptr ||
                boxes == nullptr || readBoxes == nullptr)
                continue;
            for (lpl::core::u32 index = 0u; index < chunk->count(); ++index)
            {
                const FixedVector box{lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one()};

                positions[index] = FixedVector{lpl::math::Fixed32::fromRaw(startAlongX(seeded)),
                                               lpl::math::Fixed32::fromInt(100), lpl::math::Fixed32::zero()};
                velocities[index] =
                    FixedVector{lpl::math::Fixed32::zero(), lpl::math::Fixed32::zero(), lpl::math::Fixed32::zero()};
                masses[index] = lpl::math::Fixed32::one();
                readMasses[index] = lpl::math::Fixed32::one();
                boxes[index] = box;
                readBoxes[index] = box;
                ++seeded;
            }
        }
    }
    return seeded;
}

/**
 * @brief Where the first body of the row stands along x, in raw Fixed32.
 */
[[nodiscard]] lpl::core::i32 firstAlongX(PartitionSpan partitions) noexcept
{
    for (const auto &partition : partitions)
        for (const auto &chunk : partition->chunks())
            if (chunk->count() != 0u)
                return static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Position))->x.raw();
    return 0;
}

} // namespace

/**
 * @brief A body pushed further than the margin during a step still meets the bodies before it.
 *
 * The lists are made at the step's start, and body 1 is not in body 0's list. During the first
 * iteration body 2 pushes body 1 an eighth of a unit into body 0. The next iteration finds body 1
 * beyond the margin and searches the tree for every body, so body 0 meets body 1 and moves; lists
 * trusted past the first iteration would leave body 0 where it stood.
 */
LPL_TEST(a_body_pushed_beyond_the_margin_meets_the_bodies_before_it)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;

    for (lpl::core::u32 body = 0u; body < kBodies; ++body)
        (void) registry.createEntity(archetype);

    const PartitionSpan partitions = registry.partitions();

    if (!test.check(seedRow(partitions) == kBodies, "forty unit boxes are put in a row"))
        return;

    lpl::physics::CpuPhysicsBackend backend{registry};

    (void) backend.init();
    test.check(backend.step(1.0f / 60.0f).has_value(), "the backend steps");

    const lpl::core::i32 first = firstAlongX(partitions);

    test.check(first < 0, "body 0 is pushed back by body 1");
    test.measureHexadecimal("first_along_x", static_cast<lpl::core::u32>(first));
}
