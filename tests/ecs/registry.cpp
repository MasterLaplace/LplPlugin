#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Entity.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(registry);

namespace {

[[nodiscard]] lpl::core::u32 countPartitionEntities(lpl::ecs::Registry &registry)
{
    lpl::core::u32 entities = 0u;

    for (const auto &partition : registry.partitions())
        entities += partition->entityCount();
    return entities;
}

/** @brief Hands out the write buffer of @p id in every chunk, as a system that writes it does. */
void handOutForWriting(lpl::ecs::Registry &registry, lpl::ecs::ComponentId id)
{
    for (const auto &partition : registry.partitions())
        for (const auto &chunk : partition->chunks())
            (void) chunk->writeComponent(id);
}

} // namespace

/**
 * @brief Gate P1 ECS: ten entities, three destroyed, and the freed slots reused last in, first out
 *        with their generation bumped, so the stale identifier reads dead.
 */
LPL_TEST(freed_slots_are_reused_last_in_first_out)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                                lpl::ecs::ComponentId::Mass};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;
    constexpr lpl::core::u32 kCount = 10u;
    lpl::ecs::EntityId created[kCount];
    lpl::core::u32 createdCount = 0u;

    for (; createdCount < kCount; ++createdCount)
    {
        auto entity = registry.createEntity(archetype);

        if (!entity.has_value())
            break;
        created[createdCount] = entity.value();
    }
    if (!test.check(createdCount == kCount, "ten entities are created"))
        return;
    test.check(registry.liveCount() == kCount, "and all ten are alive");

    bool destroyed = true;

    for (const lpl::core::u32 slot : {2u, 4u, 6u})
        destroyed = registry.destroyEntity(created[slot]).has_value() && destroyed;
    test.check(destroyed, "slots 2, 4 and 6 are destroyed");
    test.check(registry.liveCount() == kCount - 3u, "leaving seven alive");

    const auto recycled = registry.createEntity(archetype);

    if (test.check(recycled.has_value(), "a new entity is created"))
    {
        test.check(recycled.value().slot() == 6u, "in the slot freed last");
        test.check(recycled.value().generation() == 1u, "with its generation bumped");
    }
    test.check(!registry.isAlive(created[6]), "the old identifier of that slot reads dead");

    (void) registry.createEntity(archetype);
    (void) registry.createEntity(archetype);
    test.check(registry.liveCount() == kCount, "two more refill the other freed slots");
    test.check(countPartitionEntities(registry) == kCount, "and the partitions hold every live entity");

    test.measureHexadecimal("first_entity", created[0].raw());
}

/**
 * @brief A swap counts exactly the bytes it copies from back to front buffers, per component and
 *        in total: the data the double buffer moves, the same on both targets.
 */
LPL_TEST(a_swap_counts_the_bytes_it_copies)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Mass};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;
    constexpr lpl::core::u32 kCount = 3u;

    for (lpl::core::u32 created = 0u; created < kCount; ++created)
        if (!test.check(static_cast<bool>(registry.createEntity(archetype)), "an entity is created"))
            return;
    test.check(registry.swappedBytes() == 0u, "nothing is counted before the first swap");

    handOutForWriting(registry, lpl::ecs::ComponentId::Position);
    handOutForWriting(registry, lpl::ecs::ComponentId::Mass);
    registry.swapAllBuffers();

    const lpl::core::u64 position =
        static_cast<lpl::core::u64>(kCount) * lpl::ecs::defaultLayout(lpl::ecs::ComponentId::Position).size;
    const lpl::core::u64 mass =
        static_cast<lpl::core::u64>(kCount) * lpl::ecs::defaultLayout(lpl::ecs::ComponentId::Mass).size;
    test.check(registry.swappedBytes(lpl::ecs::ComponentId::Position) == position &&
                   registry.swappedBytes(lpl::ecs::ComponentId::Mass) == mass,
               "each component counts the bytes of its own buffers");
    test.check(registry.swappedBytes(lpl::ecs::ComponentId::Velocity) == 0u,
               "a component no entity carries counts nothing");

    handOutForWriting(registry, lpl::ecs::ComponentId::Position);
    handOutForWriting(registry, lpl::ecs::ComponentId::Mass);
    registry.swapAllBuffers();

    test.check(registry.swappedBytes() == 2u * (position + mass), "and every swap adds what it copied");
    test.measure("swapped_bytes", static_cast<lpl::core::u32>(registry.swappedBytes()));
}

/**
 * @brief A swap copies only the components handed out for writing since the last one: a component
 *        only read through @ref lpl::ecs::Chunk::latestComponent keeps its two buffers, which
 *        already agree, and the written one is published.
 */
LPL_TEST(a_swap_copies_only_what_was_handed_out_for_writing)
{
    const lpl::ecs::ComponentId components[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Mass};
    lpl::ecs::Archetype archetype{components};
    lpl::ecs::Registry registry;
    const auto entity = registry.createEntity(archetype);
    lpl::core::u32 row = 0u;

    if (!test.check(static_cast<bool>(entity), "an entity is created"))
        return;
    lpl::ecs::Chunk *chunk = registry.chunkOf(entity.value(), row);
    if (!test.check(chunk != nullptr, "its chunk is found"))
        return;

    const lpl::core::u64 massBefore = registry.swappedBytes(lpl::ecs::ComponentId::Mass);
    auto *positions =
        static_cast<lpl::math::Vec3<lpl::math::Fixed32> *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
    const auto *masses = static_cast<const lpl::math::Fixed32 *>(chunk->latestComponent(lpl::ecs::ComponentId::Mass));
    positions[row].x = lpl::math::Fixed32::fromInt(7);
    const lpl::math::Fixed32 mass = masses[row];

    registry.swapAllBuffers();

    const auto *published =
        static_cast<const lpl::math::Vec3<lpl::math::Fixed32> *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
    const auto *publishedMass =
        static_cast<const lpl::math::Fixed32 *>(chunk->readComponent(lpl::ecs::ComponentId::Mass));
    test.check(registry.swappedBytes(lpl::ecs::ComponentId::Position) ==
                   lpl::ecs::defaultLayout(lpl::ecs::ComponentId::Position).size,
               "the written component is copied");
    test.check(registry.swappedBytes(lpl::ecs::ComponentId::Mass) == massBefore, "the one only read is not");
    test.check(published[row].x == lpl::math::Fixed32::fromInt(7), "the written value is published");
    test.check(publishedMass[row] == mass && static_cast<const lpl::math::Fixed32 *>(
                                                 chunk->latestComponent(lpl::ecs::ComponentId::Mass))[row] == mass,
               "and both buffers of the one only read still hold its value");
}
