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

constexpr lpl::core::i32 kUnit = 65536;  /**< One unit, in raw Fixed32. */
constexpr lpl::core::i32 kMargin = 1024; /**< The speculative distance, 1/64, in raw Fixed32. */

/**
 * @struct Row
 * @brief A row of unit boxes of unit mass at a height of 100: where each starts along x, in raw
 *        Fixed32, and how fast it moves along x, in units a second.
 */
struct Row {
    lpl::core::i32 x[kBodies];     /**< Where each box starts along x, in raw Fixed32. */
    lpl::core::i32 speed[kBodies]; /**< How fast each box moves along x, in units a second. */
};

/**
 * @brief Gives every box the place and the speed @p row holds for it.
 *
 * @details Position and velocity go into the buffer the integrator writes; the mass and the box go
 *          into both buffers, because the resolver reads them from the other one.
 *
 * @return Bodies seeded.
 */
lpl::core::u32 seedRow(PartitionSpan partitions, const Row &row)
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
            for (lpl::core::u32 index = 0u; index < chunk->count() && seeded < kBodies; ++index)
            {
                const FixedVector box{lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one()};

                positions[index] = FixedVector{lpl::math::Fixed32::fromRaw(row.x[seeded]),
                                               lpl::math::Fixed32::fromInt(100), lpl::math::Fixed32::zero()};
                velocities[index] = FixedVector{lpl::math::Fixed32::fromInt(row.speed[seeded]),
                                                lpl::math::Fixed32::zero(), lpl::math::Fixed32::zero()};
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
 * @brief The row's boxes at rest, three units apart from x = 100.
 */
[[nodiscard]] Row spreadRow() noexcept
{
    Row row{};
    for (lpl::core::u32 body = 0u; body < kBodies; ++body)
    {
        row.x[body] = (100 + 3 * static_cast<lpl::core::i32>(body)) * kUnit;
        row.speed[body] = 0;
    }
    return row;
}

/**
 * @brief Where every box of the row stands along x, in raw Fixed32.
 */
void readAlongX(PartitionSpan partitions, lpl::core::i32 (&x)[kBodies]) noexcept
{
    lpl::core::u32 read = 0u;
    for (const auto &partition : partitions)
        for (const auto &chunk : partition->chunks())
        {
            const auto *positions =
                static_cast<const FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            for (lpl::core::u32 index = 0u; index < chunk->count() && read < kBodies; ++index)
                x[read++] = positions[index].x.raw();
        }
}

/**
 * @brief Moves box @p body of the row to @p x, in raw Fixed32, between two steps.
 */
void placeAlongX(PartitionSpan partitions, lpl::core::u32 body, lpl::core::i32 x) noexcept
{
    lpl::core::u32 seen = 0u;
    for (const auto &partition : partitions)
        for (const auto &chunk : partition->chunks())
        {
            auto *positions = static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            for (lpl::core::u32 index = 0u; index < chunk->count(); ++index, ++seen)
                if (seen == body)
                    positions[index].x = lpl::math::Fixed32::fromRaw(x);
        }
}

/**
 * @struct World
 * @brief A registry of the row's boxes, and a backend over it.
 */
struct World {
    lpl::ecs::Registry registry;                       /**< The boxes. */
    lpl::physics::CpuPhysicsBackend backend{registry}; /**< Steps them. */

    /**
     * @brief Makes the row's entities, and starts the backend.
     */
    World()
    {
        const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                    lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
        lpl::ecs::Archetype archetype{components};
        for (lpl::core::u32 body = 0u; body < kBodies; ++body)
            (void) registry.createEntity(archetype);
        (void) backend.init();
    }
};

} // namespace

/**
 * @brief A box closing on another across a gap of 1/64 closes the gap within the step and no
 *        further; across twice that gap, it is not a contact.
 *
 * @details Boxes 1 and 3 close on boxes 0 and 2 at two units a second, a thirtieth of a unit per
 *          step. Box 1 starts 1/64 away from box 0: a speculative contact, which brings the two
 *          together and pushes box 0. Box 3 starts 2/64 away from box 2: no contact, so box 2 stays
 *          where it was and box 3 ends the step inside it.
 */
LPL_TEST(a_gap_of_the_speculative_distance_closes_and_no_further)
{
    World world;
    Row row = spreadRow();
    row.x[0] = 0;
    row.x[1] = kUnit + kMargin;
    row.speed[1] = -2;
    row.x[2] = 10 * kUnit;
    row.x[3] = 11 * kUnit + 2 * kMargin;
    row.speed[3] = -2;
    const PartitionSpan partitions = world.registry.partitions();

    if (!test.check(seedRow(partitions, row) == kBodies, "forty unit boxes are put in a row"))
        return;
    test.check(world.backend.step(1.0f / 60.0f).has_value(), "the backend steps");

    lpl::core::i32 x[kBodies];
    readAlongX(partitions, x);
    const lpl::core::i32 gapAfter = x[1] - x[0] - kUnit;
    const lpl::core::i32 overlapAfter = x[2] + kUnit - x[3];

    test.check(x[0] < 0, "box 0 is pushed by the box closing on it");
    test.check(gapAfter >= -2 && gapAfter <= 2, "and the two end the step touching");
    test.check(x[2] == 10 * kUnit, "box 2, twice as far, is not touched");
    test.check(overlapAfter > 0, "and the box closing on it ends the step inside it");
    test.measure("gap_after_raw", static_cast<lpl::core::u32>(gapAfter));
}

/**
 * @brief A box moved beyond the margin between two steps meets the boxes where it now stands.
 *
 * @details The first step makes the lists, with box 39 at the far end of the row. It is then moved a
 *          quarter unit into box 0, so the lists of the first step pair it with nothing there. The
 *          second step finds a box beyond the margin of where the lists found it and makes them
 *          again, so box 0 is pushed: lists kept past such a move would leave it where it stood.
 */
LPL_TEST(lists_are_made_again_when_a_box_moves_beyond_the_margin)
{
    World world;
    const PartitionSpan partitions = world.registry.partitions();

    if (!test.check(seedRow(partitions, spreadRow()) == kBodies, "forty unit boxes are put in a row"))
        return;
    test.check(world.backend.step(1.0f / 60.0f).has_value(), "the backend steps once");

    lpl::core::i32 before[kBodies];
    readAlongX(partitions, before);
    placeAlongX(partitions, kBodies - 1u, before[0] + kUnit - kUnit / 4);
    test.check(world.backend.step(1.0f / 60.0f).has_value(), "and once more");

    lpl::core::i32 after[kBodies];
    readAlongX(partitions, after);

    test.check(after[0] < before[0], "box 0 is pushed by the box moved into it");
    test.measureHexadecimal("first_along_x", static_cast<lpl::core::u32>(after[0]));
}
