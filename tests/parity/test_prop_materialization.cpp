#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

#include <cstdio>

namespace {

using FVec3 = lpl::math::Vec3<lpl::math::Fixed32>;

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    if (!condition)
    {
        ++gFailures;
        std::printf("  FAIL %s\n", what);
    }
}

/**
 * @brief Reads one entity's mass through its identity, never through a row number.
 */
[[nodiscard]] lpl::math::Fixed32 massOf(const lpl::ecs::Registry &registry, lpl::ecs::EntityId id)
{
    const auto ref = registry.resolve(id);
    if (!ref)
        return lpl::math::Fixed32::fromInt(-1);
    for (const auto &partition : registry.partitions())
    {
        if (!partition || !partition->archetype().has(lpl::ecs::ComponentId::Mass))
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const auto ids = chunk->entities();
            const auto *mass =
                static_cast<const lpl::math::Fixed32 *>(chunk->readComponent(lpl::ecs::ComponentId::Mass));
            if (ids.empty() || mass == nullptr)
                continue;
            for (lpl::core::u32 i = 0u; i < chunk->count(); ++i)
                if (ids[i] == id)
                    return mass[i];
        }
    }
    return lpl::math::Fixed32::fromInt(-1);
}

[[nodiscard]] FVec3 positionOf(const lpl::ecs::Registry &registry, lpl::ecs::EntityId id)
{
    for (const auto &partition : registry.partitions())
    {
        if (!partition)
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const auto ids = chunk->entities();
            const auto *position = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
            if (ids.empty() || position == nullptr)
                continue;
            for (lpl::core::u32 i = 0u; i < chunk->count(); ++i)
                if (ids[i] == id)
                    return position[i];
        }
    }
    return FVec3{};
}

/**
 * @brief Scattering props does not overwrite bodies that were already there.
 *
 * Bodies come first, exactly as a host does: create them once, give every one a real mass and a distinctive height,
 * then let the world be (re)generated around them. The world is generated with collidable scatter into the SAME
 * registry: a collidable prop lands in the body archetype, which is the whole trap.
 *
 * The claim is measured through identity rather than through row order: every body still weighs what it weighed. A
 * zero here is a boulder that has quietly become a tree, and on screen it is a boulder hanging in mid-air.
 *
 * And the props themselves must have landed: a fix that protects the bodies by dropping the props on the floor would
 * pass the two checks on the bodies.
 */
void testScatterLeavesExistingBodiesAlone()
{
    std::printf("scattering props does not overwrite bodies that were already there\n");

    lpl::ecs::Registry registry;

    const lpl::ecs::ComponentId bodyIds[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                             lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
    const lpl::ecs::Archetype bodyArchetype{bodyIds};

    lpl::pmr::vector<lpl::ecs::EntityId> bodies;
    for (lpl::core::u32 i = 0u; i < 24u; ++i)
    {
        auto created = registry.createEntity(bodyArchetype);
        check(static_cast<bool>(created), "the body was created");
        if (created)
            bodies.push_back(*created);
    }

    const lpl::math::Fixed32 weight = lpl::math::Fixed32::fromFloat(4.0f);
    const lpl::math::Fixed32 altitude = lpl::math::Fixed32::fromFloat(40.0f);
    for (const auto &partition : registry.partitions())
    {
        if (!partition || !(partition->archetype() == bodyArchetype))
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            auto *mass = static_cast<lpl::math::Fixed32 *>(chunk->writeComponent(lpl::ecs::ComponentId::Mass));
            auto *massRead = static_cast<lpl::math::Fixed32 *>(
                const_cast<void *>(chunk->readComponent(lpl::ecs::ComponentId::Mass)));
            auto *position = static_cast<FVec3 *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
            auto *positionRead =
                static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(lpl::ecs::ComponentId::Position)));
            for (lpl::core::u32 i = 0u; i < chunk->count(); ++i)
            {
                if (mass != nullptr)
                    mass[i] = weight;
                if (massRead != nullptr)
                    massRead[i] = weight;
                if (position != nullptr)
                    position[i] = FVec3{lpl::math::Fixed32::zero(), altitude, lpl::math::Fixed32::zero()};
                if (positionRead != nullptr)
                    positionRead[i] = FVec3{lpl::math::Fixed32::zero(), altitude, lpl::math::Fixed32::zero()};
            }
        }
    }

    lpl::procgen::WorldBuilder builder{4242u};
    lpl::procgen::ScatterRule forest;
    forest.biome = lpl::procgen::BiomeId::Forest;
    forest.density = 0.25f;
    forest.halfExtent = 0.5f;
    forest.collidable = true;
    builder.terrain(48u, 48u).normalize(0.5f, 14.0f).biomes().scatter(forest);

    lpl::pmr::vector<lpl::ecs::EntityId> props;
    const lpl::procgen::BuiltWorldStats stats = builder.materializeProps(registry, &props);
    std::printf("    %u props scattered alongside %u bodies\n", stats.propEntities,
                static_cast<lpl::core::u32>(bodies.size()));
    check(stats.propEntities != 0u, "the scatter placed something (otherwise this proves nothing)");

    lpl::core::u32 stolen = 0u;
    lpl::core::u32 moved = 0u;
    for (lpl::core::u32 i = 0u; i < bodies.size(); ++i)
    {
        if (massOf(registry, bodies[i]).raw() != weight.raw())
            ++stolen;
        if (positionOf(registry, bodies[i]).y.raw() != altitude.raw())
            ++moved;
    }
    std::printf("    %u of %u bodies lost their mass, %u were moved\n", stolen,
                static_cast<lpl::core::u32>(bodies.size()), moved);
    check(stolen == 0u, "no body had its mass overwritten by a prop");
    check(moved == 0u, "no body was teleported onto a prop's position");

    lpl::core::u32 placed = 0u;
    for (lpl::core::u32 i = 0u; i < props.size(); ++i)
        if (massOf(registry, props[i]).raw() == 0)
            ++placed;
    std::printf("    %u of %u props carry a prop's zero mass\n", placed, static_cast<lpl::core::u32>(props.size()));
    check(placed == props.size(), "every prop was written where it belongs");
}

/**
 * @brief A body survives many regenerations of the world around it.
 *
 * The viewer's actual loop: retire the previous world's props, generate the next one. The destroy/create churn is what
 * reshuffles the rows, so one pass would not have caught this.
 */
void testRegenerationIsStable()
{
    std::printf("a body survives many regenerations of the world around it\n");

    lpl::ecs::Registry registry;
    const lpl::ecs::ComponentId bodyIds[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                             lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB};
    const lpl::ecs::Archetype bodyArchetype{bodyIds};

    lpl::pmr::vector<lpl::ecs::EntityId> bodies;
    for (lpl::core::u32 i = 0u; i < 8u; ++i)
        if (auto created = registry.createEntity(bodyArchetype))
            bodies.push_back(*created);

    const lpl::math::Fixed32 weight = lpl::math::Fixed32::fromFloat(2.5f);
    for (const auto &partition : registry.partitions())
    {
        if (!partition || !(partition->archetype() == bodyArchetype))
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            auto *mass = static_cast<lpl::math::Fixed32 *>(chunk->writeComponent(lpl::ecs::ComponentId::Mass));
            auto *massRead = static_cast<lpl::math::Fixed32 *>(
                const_cast<void *>(chunk->readComponent(lpl::ecs::ComponentId::Mass)));
            for (lpl::core::u32 i = 0u; i < chunk->count(); ++i)
            {
                if (mass != nullptr)
                    mass[i] = weight;
                if (massRead != nullptr)
                    massRead[i] = weight;
            }
        }
    }

    lpl::pmr::vector<lpl::ecs::EntityId> props;
    for (lpl::core::u32 round = 0u; round < 5u; ++round)
    {
        for (lpl::core::u32 i = 0u; i < props.size(); ++i)
            (void) registry.destroyEntity(props[i]);
        props.clear();

        lpl::procgen::WorldBuilder builder{1000u + round * 77u};
        lpl::procgen::ScatterRule rule;
        rule.biome = lpl::procgen::BiomeId::Grassland;
        rule.density = 0.2f;
        rule.collidable = true;
        builder.terrain(40u, 40u).normalize(0.5f, 12.0f).biomes().scatter(rule);
        (void) builder.materializeProps(registry, &props);
    }

    lpl::core::u32 stolen = 0u;
    for (lpl::core::u32 i = 0u; i < bodies.size(); ++i)
        if (massOf(registry, bodies[i]).raw() != weight.raw())
            ++stolen;
    std::printf("    after 5 regenerations, %u of %u bodies still weigh what they did\n",
                static_cast<lpl::core::u32>(bodies.size()) - stolen, static_cast<lpl::core::u32>(bodies.size()));
    check(stolen == 0u, "no body was lost to five rounds of world churn");
}

} // namespace

/**
 * @brief Scattering props must not touch entities somebody else owns.
 *
 * A collidable prop needs Position, Velocity, Mass and AABB for the solver to see
 * it, which is EXACTLY the archetype a loose body has. The two therefore share a
 * partition — and a writer that addresses rows by their position in the chunk
 * rather than by the entity that owns them will write over whatever is in the
 * first rows, whoever put it there.
 *
 * The visible symptom is specific and was reported from the viewer before it was
 * ever measured here: regenerate a world and some boulders hang motionless in the
 * air. They are not stuck — they stopped being boulders. A prop's mass is zero
 * (that is how the solver is told "immovable"), so a body overwritten by a prop
 * keeps a position high above the ground and loses the one property that made
 * gravity apply to it.
 */
int main()
{
    std::printf("== prop materialisation ==\n");
    testScatterLeavesExistingBodiesAlone();
    testRegenerationIsStable();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
