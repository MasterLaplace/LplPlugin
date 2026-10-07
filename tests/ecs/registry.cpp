#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Entity.hpp>
#include <lpl/ecs/Registry.hpp>
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
