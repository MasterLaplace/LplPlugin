#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/engine/systems/HeightfieldCollisionSystem.hpp>

#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

namespace {

using FVec3 = lpl::math::Vec3<lpl::math::Fixed32>;

constexpr lpl::core::u32 kSize = 32u;

/**
 * @brief A west-high ramp: heights fall as x grows, so downhill is +x.
 */
lpl::procgen::Heightfield ramp(lpl::core::u32 size)
{
    lpl::procgen::Heightfield field{size, size, lpl::math::Fixed32{}};
    for (lpl::core::u32 z = 0u; z < size; ++z)
        for (lpl::core::u32 x = 0u; x < size; ++x)
            field.at(x, z) = lpl::math::Fixed32::fromFloat(8.0f - static_cast<float>(x) * 0.5f);
    return field;
}

/**
 * @brief One body with Position, Velocity, AABB and Mass.
 *
 * @details Its values are written to the WRITE side: that is where a system writes and therefore where a caller
 *          reads in a world whose systems never trigger a phase swap.
 *
 * @return Its id.
 */
lpl::ecs::EntityId spawnBody(lpl::ecs::Registry &registry, float x, float y, float z, float mass)
{
    lpl::ecs::Archetype archetype;
    archetype.add(lpl::ecs::ComponentId::Position);
    archetype.add(lpl::ecs::ComponentId::Velocity);
    archetype.add(lpl::ecs::ComponentId::AABB);
    archetype.add(lpl::ecs::ComponentId::Mass);

    const auto created = registry.createEntity(archetype);
    if (!created)
        return lpl::ecs::EntityId{};
    const lpl::ecs::EntityId id = created.value();

    lpl::core::u32 row = 0u;
    lpl::ecs::Chunk *held = registry.chunkOf(id, row);
    if (held == nullptr)
        return id;
    lpl::ecs::Chunk &chunk = *held;

    auto *positions = static_cast<FVec3 *>(chunk.writeComponent(lpl::ecs::ComponentId::Position));
    auto *velocities = static_cast<FVec3 *>(chunk.writeComponent(lpl::ecs::ComponentId::Velocity));
    auto *aabbs = static_cast<FVec3 *>(chunk.writeComponent(lpl::ecs::ComponentId::AABB));
    auto *masses = static_cast<lpl::math::Fixed32 *>(chunk.writeComponent(lpl::ecs::ComponentId::Mass));
    positions[row] =
        FVec3{lpl::math::Fixed32::fromFloat(x), lpl::math::Fixed32::fromFloat(y), lpl::math::Fixed32::fromFloat(z)};
    velocities[row] = FVec3{lpl::math::Fixed32{}, lpl::math::Fixed32::fromFloat(-2.0f), lpl::math::Fixed32{}};
    aabbs[row] = FVec3{lpl::math::Fixed32::fromFloat(0.5f), lpl::math::Fixed32::fromFloat(0.5f),
                       lpl::math::Fixed32::fromFloat(0.5f)};
    masses[row] = lpl::math::Fixed32::fromFloat(mass);
    return id;
}

/**
 * @brief Reads one vector component of @p id from the WRITE side.
 *
 * @details That is where a system writes, and this world's systems never trigger a phase swap, so the front buffer
 *          still holds what the entity was born with.
 */
FVec3 readVec(lpl::ecs::Registry &registry, lpl::ecs::EntityId id, lpl::ecs::ComponentId component)
{
    lpl::core::u32 row = 0u;
    lpl::ecs::Chunk *chunk = registry.chunkOf(id, row);
    if (chunk == nullptr)
        return FVec3{};
    auto *values = static_cast<FVec3 *>(chunk->writeComponent(component));
    return values == nullptr ? FVec3{} : values[row];
}

FVec3 positionOf(lpl::ecs::Registry &registry, lpl::ecs::EntityId id)
{
    return readVec(registry, id, lpl::ecs::ComponentId::Position);
}

FVec3 velocityOf(lpl::ecs::Registry &registry, lpl::ecs::EntityId id)
{
    return readVec(registry, id, lpl::ecs::ComponentId::Velocity);
}

/**
 * @brief A falling body lands and stops falling.
 *
 * @details Cell 16 of the ramp stands at 8 - 8 = 0, so a body dropped from above it must come to rest at 0 + half
 *          its height. Put under the surface and stepped again, it must be lifted onto it.
 */
void checkAFallingBodyLandsAndStopsFalling(const lpl::procgen::Heightfield &field)
{
    std::printf("-- a falling body lands and stops falling --\n");
    lpl::ecs::Registry registry;
    const lpl::ecs::EntityId body = spawnBody(registry, 0.0f, 20.0f, 0.0f, 1.0f);
    lpl::engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
    collision.execute(1.0f / 60.0f);
    check(collision.resting() == 0u, "a body above the ground is not in contact");

    {
        lpl::core::u32 row = 0u;
        lpl::ecs::Chunk *chunk = registry.chunkOf(body, row);
        auto *positions = static_cast<FVec3 *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
        positions[row].y = lpl::math::Fixed32::fromFloat(-4.0f);
    }
    collision.execute(1.0f / 60.0f);
    check(collision.resting() == 1u, "a body below the ground is in contact after one step");
    const FVec3 at = positionOf(registry, body);
    check(at.y.toFloat() > -1.0f, "and it was lifted onto the surface");
    check(velocityOf(registry, body).y.toFloat() >= 0.0f, "its downward velocity was cancelled");
    std::printf("     landed at y=%.3f\n", at.y.toFloat());
}

/**
 * @brief Zero mass is immovable.
 */
void checkZeroMassMeansImmovable(const lpl::procgen::Heightfield &field)
{
    std::printf("\n-- zero mass means immovable --\n");
    lpl::ecs::Registry registry;
    const lpl::ecs::EntityId tree = spawnBody(registry, 0.0f, -4.0f, 0.0f, 0.0f);
    lpl::engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
    for (lpl::core::u32 i = 0u; i < 30u; ++i)
        collision.execute(1.0f / 60.0f);
    const FVec3 at = positionOf(registry, tree);
    const FVec3 velocity = velocityOf(registry, tree);
    check(at.y.toFloat() == -4.0f, "a massless body is never corrected");
    check(velocity.x.raw() == 0 && velocity.z.raw() == 0, "and never handed a downhill slide");
    check(collision.resting() == 0u, "so it never counts as resting either");
}

/**
 * @brief The slide follows the gradient: a resting body slides downhill and loses speed.
 *
 * @details The ramp falls toward +x, so the push must be toward +x.
 *
 * Friction: with no new slope contribution the horizontal speed must decay. A flat field is the clean way to ask,
 * since the gradient term is then zero.
 */
void checkARestingBodySlidesDownhillAndLosesSpeed(const lpl::procgen::Heightfield &field)
{
    std::printf("\n-- a resting body slides downhill and loses speed --\n");
    lpl::ecs::Registry registry;
    const lpl::ecs::EntityId rock = spawnBody(registry, -8.0f, -20.0f, 0.0f, 1.0f);
    lpl::engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
    const float startX = positionOf(registry, rock).x.toFloat();
    collision.execute(1.0f / 60.0f);
    const float pushed = velocityOf(registry, rock).x.toFloat();
    check(pushed > 0.0f, "the push is downhill, not uphill");
    std::printf("     one step of slide gave vx=%+.4f from x=%+.1f\n", pushed, startX);

    const lpl::procgen::Heightfield flat{kSize, kSize, lpl::math::Fixed32{}};
    lpl::ecs::Registry level;
    const lpl::ecs::EntityId sliding = spawnBody(level, 0.0f, -4.0f, 0.0f, 1.0f);
    {
        lpl::core::u32 row = 0u;
        lpl::ecs::Chunk *chunk = level.chunkOf(sliding, row);
        auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(lpl::ecs::ComponentId::Velocity));
        velocities[row].x = lpl::math::Fixed32::fromFloat(4.0f);
    }
    lpl::engine::systems::HeightfieldCollisionSystem onFlat{level, flat, 1.0f};
    onFlat.execute(1.0f / 60.0f);
    const float after = velocityOf(level, sliding).x.toFloat();
    std::printf("     4.000 became %.4f on flat ground\n", after);
    check(after < 4.0f && after > 0.0f, "friction bleeds horizontal speed without reversing it");
}

/**
 * @brief Bodies stay over the map.
 *
 * @details The body starts well past the eastern edge: half the map is 16 units, so 40 is outside.
 */
void checkABodyNeverLeavesTheMap(const lpl::procgen::Heightfield &field)
{
    std::printf("\n-- a body never leaves the map --\n");
    lpl::ecs::Registry registry;
    const lpl::ecs::EntityId strayBody = spawnBody(registry, 40.0f, -4.0f, 40.0f, 1.0f);
    lpl::engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
    collision.execute(1.0f / 60.0f);
    const FVec3 at = positionOf(registry, strayBody);
    std::printf("     (40.0, 40.0) became (%.1f, %.1f)\n", at.x.toFloat(), at.z.toFloat());
    check(at.x.toFloat() <= 15.0f && at.z.toFloat() <= 15.0f, "a body outside the map is pulled back over it");
    const FVec3 velocity = velocityOf(registry, strayBody);
    check(velocity.x.raw() == 0 || velocity.z.raw() == 0, "and its outward velocity is cancelled");
}

/**
 * @brief An empty field is survivable.
 */
void checkAnEmptyWorldDoesNotCrash()
{
    std::printf("\n-- an empty world does not crash --\n");
    lpl::ecs::Registry registry;
    (void) spawnBody(registry, 0.0f, -4.0f, 0.0f, 1.0f);
    const lpl::procgen::Heightfield nothing;
    lpl::engine::systems::HeightfieldCollisionSystem collision{registry, nothing, 1.0f};
    collision.execute(1.0f / 60.0f);
    check(collision.resting() == 0u, "nothing to stand on means nothing resting");
}

} // namespace

/**
 * @brief The three rules of landing on a heightfield, asserted.
 *
 * This system spent its life inside `apps/mapview/main.cpp`, arguing in its own
 * comment that it belonged there because "a heightfield is content, not a host
 * service". A heightfield is content; colliding a rigid body against one is not, and
 * the argument is exactly the one that keeps engine knowledge trapped where no test
 * can reach it.
 *
 * Each check below is a rule that was learned from looking at a picture, and none of
 * them needs a picture now:
 *
 *  1. **A falling body lands and stops falling.** The count that means "at rest" is
 *     the CONTACT count, not the velocity: a body held up by position correction
 *     still takes a tick of gravity every tick, so its velocity oscillates around
 *     zero forever while it has not moved.
 *  2. **Zero mass means immovable.** Correcting one to the ground is harmless; giving
 *     it the downhill slide walks a tree off its own footing.
 *  3. **Bodies stay over the map.** One nudged past the edge sails off and hangs in
 *     empty space, which looks exactly like the collision having failed.
 *  4. **The slide follows the downhill gradient**, and loses speed doing it — without
 *     it a boulder stops where it landed and the terrain might as well be a floor.
 */
int main()
{
    std::printf("=== heightfield collision ===\n\n");

    const lpl::procgen::Heightfield field = ramp(kSize);

    checkAFallingBodyLandsAndStopsFalling(field);

    checkZeroMassMeansImmovable(field);

    checkARestingBodySlidesDownhillAndLosesSpeed(field);

    checkABodyNeverLeavesTheMap(field);

    checkAnEmptyWorldDoesNotCrash();

    std::printf("\n%s\n", failures == 0 ? "ALL PASS (0 failures)" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
