#include <lpl/core/EventBus.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/physics/CollisionEvent.hpp>
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

/**
 * @brief What a subscriber to CollisionEvent saw over one step.
 */
struct Contacts {
    lpl::core::u32 published = 0u;   /**< Events received. */
    lpl::ecs::EntityId first{};      /**< The first body of the last event. */
    lpl::ecs::EntityId second{};     /**< The second body of the last event. */
    lpl::math::Vec3<float> normal{}; /**< The normal of the last event. */
    float impulse = 0.0f;            /**< The impulse magnitude of the last event. */
};

/**
 * @brief Puts two unit-mass boxes of side 2 at a height of 100, one unit apart along x, closing on
 *        each other at 2 units a second each.
 *
 * @details Position and velocity go into the buffer the integrator writes; the mass and the box go
 *          into both buffers, because the resolver reads them from the other one.
 *
 * @return Bodies seeded.
 */
lpl::core::u32 seedClosing(PartitionSpan partitions)
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
            for (lpl::core::u32 index = 0u; index < chunk->count() && seeded < 2u; ++index)
            {
                const lpl::math::Fixed32 side = seeded == 0u ? -lpl::math::Fixed32::one() : lpl::math::Fixed32::one();
                const lpl::math::Fixed32 half = lpl::math::Fixed32::fromInt(1) / lpl::math::Fixed32::fromInt(2);
                const FixedVector box{lpl::math::Fixed32::fromInt(2), lpl::math::Fixed32::fromInt(2),
                                      lpl::math::Fixed32::fromInt(2)};

                positions[index] =
                    FixedVector{side * half, lpl::math::Fixed32::fromInt(100), lpl::math::Fixed32::zero()};
                velocities[index] = FixedVector{-side * lpl::math::Fixed32::fromInt(2), lpl::math::Fixed32::zero(),
                                                lpl::math::Fixed32::zero()};
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

/** @brief Boxes in each stack of stackedSpeeds. */
constexpr lpl::core::u32 kStackHeight = 4u;

/**
 * @brief What the bodies of a world of stacks did by the end of a run.
 */
struct StackRest {
    lpl::core::u32 bodies = 0u;   /**< Bodies read. */
    lpl::core::u32 moving = 0u;   /**< Bodies whose squared speed is at or over the sleep threshold, 0.01. */
    lpl::core::u32 sunk = 0u;     /**< Bodies more than 1/64 below the height their stack puts them at. */
    lpl::math::Fixed32 fastest{}; /**< The largest squared speed. */
};

/**
 * @brief Puts @p stacks columns of kStackHeight unit boxes of unit mass on the ground, each box
 *        touching the one under it, the columns three units apart along x.
 *
 * @details Position and velocity go into the buffer the integrator writes; the mass and the box go
 *          into both buffers, because the resolver reads them from the other one.
 *
 * @return Bodies seeded.
 */
lpl::core::u32 seedStacks(PartitionSpan partitions, lpl::core::u32 stacks)
{
    lpl::core::u32 seeded = 0u;
    const FixedVector box{lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one()};

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
            for (lpl::core::u32 index = 0u; index < chunk->count() && seeded < stacks * kStackHeight; ++index)
            {
                const lpl::core::i32 column = static_cast<lpl::core::i32>(seeded / kStackHeight);
                const lpl::core::i32 level = static_cast<lpl::core::i32>(seeded % kStackHeight);
                positions[index] =
                    FixedVector{lpl::math::Fixed32::fromInt(3 * column),
                                lpl::math::Fixed32::fromRaw(32768 + 65536 * level), lpl::math::Fixed32::zero()};
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
 * @brief Reads how still the stacks of seedStacks stand.
 */
[[nodiscard]] StackRest readStacks(PartitionSpan partitions)
{
    StackRest rest;
    const lpl::math::Fixed32 threshold = lpl::math::Fixed32::fromFloat(0.01f);
    const lpl::math::Fixed32 sinking = lpl::math::Fixed32::fromRaw(1024);

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
                const lpl::math::Fixed32 speed = velocities[index].lengthSquared();
                const lpl::math::Fixed32 standing = lpl::math::Fixed32::fromRaw(
                    32768 + 65536 * static_cast<lpl::core::i32>(rest.bodies % kStackHeight));
                rest.moving += speed >= threshold ? 1u : 0u;
                rest.sunk += positions[index].y < standing - sinking ? 1u : 0u;
                rest.fastest = speed > rest.fastest ? speed : rest.fastest;
                ++rest.bodies;
            }
        }
    }
    return rest;
}

/**
 * @brief Runs @p stacks stacks of seedStacks for two seconds of 1/60 s steps and reads how still they
 *        stand.
 */
[[nodiscard]] StackRest runStacks(lpl::core::u32 stacks)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;
    for (lpl::core::u32 body = 0u; body < stacks * kStackHeight; ++body)
        (void) registry.createEntity(archetype);
    if (seedStacks(registry.partitions(), stacks) != stacks * kStackHeight)
        return StackRest{};

    lpl::physics::CpuPhysicsBackend backend{registry};
    (void) backend.init();
    for (lpl::core::u32 step = 0u; step < 120u; ++step)
        (void) backend.step(1.0f / 60.0f);
    return readStacks(registry.partitions());
}

/**
 * @brief Puts the body of index @p body, counted across the chunks, at @p position with @p velocity,
 *        a unit box of unit mass.
 *
 * @details Position and velocity go into the buffer the integrator writes; the mass and the box go
 *          into both buffers, because the resolver reads them from the other one.
 */
void placeBox(PartitionSpan partitions, lpl::core::u32 body, const FixedVector &position, const FixedVector &velocity)
{
    const FixedVector box{lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one()};
    lpl::core::u32 seen = 0u;

    for (const auto &partition : partitions)
    {
        for (const auto &chunk : partition->chunks())
        {
            for (lpl::core::u32 index = 0u; index < chunk->count(); ++index, ++seen)
            {
                if (seen != body)
                    continue;
                static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Position))[index] = position;
                static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Velocity))[index] = velocity;
                static_cast<lpl::math::Fixed32 *>(chunk->writeComponent(lpl::ecs::ComponentId::Mass))[index] =
                    lpl::math::Fixed32::one();
                static_cast<lpl::math::Fixed32 *>(const_cast<void *>(
                    chunk->readComponent(lpl::ecs::ComponentId::Mass)))[index] = lpl::math::Fixed32::one();
                static_cast<FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::AABB))[index] = box;
                static_cast<FixedVector *>(
                    const_cast<void *>(chunk->readComponent(lpl::ecs::ComponentId::AABB)))[index] = box;
            }
        }
    }
}

/**
 * @brief An FNV-1a fold of the positions of the first @p count bodies, counted across the chunks.
 */
[[nodiscard]] lpl::core::u32 foldPositions(PartitionSpan partitions, lpl::core::u32 count)
{
    lpl::core::u32 fold = 0x811C9DC5u;
    lpl::core::u32 seen = 0u;

    for (const auto &partition : partitions)
    {
        for (const auto &chunk : partition->chunks())
        {
            const auto *positions =
                static_cast<const FixedVector *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            for (lpl::core::u32 index = 0u; index < chunk->count() && seen < count; ++index, ++seen)
            {
                const lpl::core::i32 words[3] = {positions[index].x.raw(), positions[index].y.raw(),
                                                 positions[index].z.raw()};
                for (const lpl::core::i32 word : words)
                    fold = (fold ^ static_cast<lpl::core::u32>(word)) * 0x01000193u;
            }
        }
    }
    return fold;
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

/**
 * @brief Two boxes closing on each other touch in one step, and a subscriber to CollisionEvent
 *        receives that contact: the two bodies, a normal along x, and a positive impulse.
 */
LPL_TEST(a_contact_reaches_a_subscriber)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;
    const auto left = registry.createEntity(archetype);
    const auto right = registry.createEntity(archetype);

    if (!test.check(left.has_value() && right.has_value(), "two bodies are made"))
        return;
    if (!test.check(seedClosing(registry.partitions()) == 2u, "two boxes are put one unit apart, closing"))
        return;

    lpl::physics::CpuPhysicsBackend backend{registry};
    Contacts contacts;
    lpl::core::EventBus &bus = lpl::core::EventBus::instance();
    const lpl::core::u32 subscription =
        bus.subscribe<lpl::physics::CollisionEvent>([&contacts](const lpl::physics::CollisionEvent &event) {
            ++contacts.published;
            contacts.first = event.entityA;
            contacts.second = event.entityB;
            contacts.normal = event.normal;
            contacts.impulse = event.impulseMagnitude;
        });

    (void) backend.init();
    test.check(backend.step(1.0f / 60.0f).has_value(), "the backend steps");
    bus.unsubscribe<lpl::physics::CollisionEvent>(subscription);

    const bool samePair = (contacts.first == left.value() && contacts.second == right.value()) ||
                          (contacts.first == right.value() && contacts.second == left.value());

    test.check(contacts.published == 1u, "the subscriber receives one contact");
    test.check(samePair, "between the two boxes");
    test.check((contacts.normal.x == 1.0f || contacts.normal.x == -1.0f) && contacts.normal.y == 0.0f &&
                   contacts.normal.z == 0.0f,
               "along x, the axis they close on");
    test.check(contacts.impulse > 0.0f, "with a positive impulse");

    test.measure("contacts_published", contacts.published);
}

/**
 * @brief A stack of four boxes on the ground stands still: every box ends each step under the sleep
 *        threshold, at the height the stack puts it.
 *
 * @details One stack is solved pair by pair, and nine stacks, 36 bodies, take their pairs from the
 *          neighbour lists. The velocities were corrected only while two boxes overlapped, and the
 *          position correction separated them in the first iteration: every box above the bottom one
 *          kept one step of gravity, a sixth of a unit a second, and could never sleep.
 */
LPL_TEST(a_stack_at_rest_keeps_no_speed)
{
    const StackRest alone = runStacks(1u);
    const StackRest crowd = runStacks(9u);

    test.check(alone.bodies == kStackHeight && crowd.bodies == 9u * kStackHeight, "the stacks are built");
    test.check(alone.moving == 0u, "every box of a lone stack is under the sleep threshold");
    test.check(crowd.moving == 0u, "and so is every box of nine stacks, paired through the lists");
    test.check(alone.sunk == 0u && crowd.sunk == 0u, "and no box sank into the one under it");
    test.measureHexadecimal("fastest_alone", static_cast<lpl::core::u32>(alone.fastest.raw()));
    test.measureHexadecimal("fastest_crowd", static_cast<lpl::core::u32>(crowd.fastest.raw()));
}

/**
 * @brief Two stacks fall asleep, and a box dropped on the first wakes all of it, its bottom box
 *        included, and nothing of the second.
 *
 * @details Bodies that touch form an island, which sleeps and wakes whole, as in Box2D. A box asleep
 *          on its own under a woken one would hold still while the box above it moved: the drop
 *          woke only the box it hit.
 */
LPL_TEST(a_dropped_box_wakes_the_island_it_hits_and_only_that_one)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;
    lpl::ecs::EntityId stacked[2u * kStackHeight];
    for (lpl::ecs::EntityId &body : stacked)
    {
        const auto created = registry.createEntity(archetype);
        if (!test.check(created.has_value(), "a stacked box is made"))
            return;
        body = created.value();
    }
    const lpl::math::Fixed32 zero = lpl::math::Fixed32::zero();

    if (!test.check(seedStacks(registry.partitions(), 2u) == 2u * kStackHeight, "two stacks are built"))
        return;

    lpl::physics::CpuPhysicsBackend backend{registry};
    (void) backend.init();
    for (lpl::core::u32 step = 0u; step < 120u; ++step)
        (void) backend.step(1.0f / 60.0f);

    lpl::core::u32 asleep = 0u;
    for (const lpl::ecs::EntityId body : stacked)
        asleep += backend.isAwake(body) ? 0u : 1u;
    test.check(asleep == 2u * kStackHeight, "every box of both stacks sleeps once it has been still");

    const auto dropped = registry.createEntity(archetype);
    if (!test.check(dropped.has_value(), "a box to drop is made"))
        return;
    placeBox(registry.partitions(), 2u * kStackHeight, FixedVector{zero, lpl::math::Fixed32::fromInt(5), zero},
             FixedVector{zero, lpl::math::Fixed32::fromInt(-3), zero});
    for (lpl::core::u32 step = 0u; step < 20u; ++step)
        (void) backend.step(1.0f / 60.0f);

    lpl::core::u32 firstAwake = 0u;
    lpl::core::u32 secondAwake = 0u;
    for (lpl::core::u32 level = 0u; level < kStackHeight; ++level)
    {
        firstAwake += backend.isAwake(stacked[level]) ? 1u : 0u;
        secondAwake += backend.isAwake(stacked[kStackHeight + level]) ? 1u : 0u;
    }
    test.check(backend.isAwake(dropped.value()), "the dropped box is awake");
    test.check(firstAwake == kStackHeight, "every box of the stack it landed on woke, the bottom one too");
    test.check(secondAwake == 0u, "and the other stack still sleeps");
    test.measure("first_stack_awake", firstAwake);
}

/**
 * @brief A body asleep through a step reports that it did not move, and it did not; a falling one
 *        reports that it moved.
 *
 * @details The spatial index skips the bodies the backend says did not move, so the claim has to be
 *          true to the word: the sleepers' positions are read before and after the step.
 */
LPL_TEST(a_body_asleep_through_a_step_reports_that_it_did_not_move)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;
    lpl::ecs::EntityId stacked[kStackHeight];
    for (lpl::ecs::EntityId &body : stacked)
    {
        const auto created = registry.createEntity(archetype);
        if (!test.check(created.has_value(), "a stacked box is made"))
            return;
        body = created.value();
    }
    const lpl::math::Fixed32 zero = lpl::math::Fixed32::zero();

    if (!test.check(seedStacks(registry.partitions(), 1u) == kStackHeight, "a stack is built"))
        return;

    lpl::physics::CpuPhysicsBackend backend{registry};
    (void) backend.init();
    for (lpl::core::u32 step = 0u; step < 120u; ++step)
        (void) backend.step(1.0f / 60.0f);

    const auto falling = registry.createEntity(archetype);
    if (!test.check(falling.has_value(), "a box to drop is made"))
        return;
    placeBox(registry.partitions(), kStackHeight,
             FixedVector{lpl::math::Fixed32::fromInt(10), lpl::math::Fixed32::fromInt(5), zero},
             FixedVector{zero, zero, zero});

    const lpl::core::u32 before = foldPositions(registry.partitions(), kStackHeight);
    (void) backend.step(1.0f / 60.0f);
    const lpl::core::u32 after = foldPositions(registry.partitions(), kStackHeight);

    lpl::core::u32 movedSleepers = 0u;
    for (const lpl::ecs::EntityId body : stacked)
        movedSleepers += backend.movedLastStep(body) ? 1u : 0u;
    test.check(movedSleepers == 0u, "every box of the sleeping stack says it did not move");
    test.check(before == after, "and none of them moved");
    test.check(backend.movedLastStep(falling.value()), "the falling box says it moved");
}
