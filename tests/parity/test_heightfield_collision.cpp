/**
 * @file test_heightfield_collision.cpp
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
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/engine/systems/HeightfieldCollisionSystem.hpp>

#include <cstdio>
#include <string>

using namespace lpl;

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

namespace {

using FVec3 = math::Vec3<math::Fixed32>;

/// A west-high ramp: heights fall as x grows, so downhill is +x.
procgen::Heightfield ramp(core::u32 size)
{
    procgen::Heightfield field{size, size, math::Fixed32{}};
    for (core::u32 z = 0u; z < size; ++z)
        for (core::u32 x = 0u; x < size; ++x)
            field.at(x, z) = math::Fixed32::fromFloat(8.0f - static_cast<float>(x) * 0.5f);
    return field;
}

/// One body with Position, Velocity, AABB and Mass. Returns its id.
ecs::EntityId spawnBody(ecs::Registry &registry, float x, float y, float z, float mass)
{
    ecs::Archetype archetype;
    archetype.add(ecs::ComponentId::Position);
    archetype.add(ecs::ComponentId::Velocity);
    archetype.add(ecs::ComponentId::AABB);
    archetype.add(ecs::ComponentId::Mass);

    const auto created = registry.createEntity(archetype);
    if (!created)
        return ecs::EntityId{};
    const ecs::EntityId id = created.value();

    core::u32 row = 0u;
    ecs::Chunk *held = registry.chunkOf(id, row);
    if (held == nullptr)
        return id;
    ecs::Chunk &chunk = *held;

    // The WRITE side: that is where a system writes and therefore where a caller
    // reads in a world whose systems never trigger a phase swap.
    auto *positions = static_cast<FVec3 *>(chunk.writeComponent(ecs::ComponentId::Position));
    auto *velocities = static_cast<FVec3 *>(chunk.writeComponent(ecs::ComponentId::Velocity));
    auto *aabbs = static_cast<FVec3 *>(chunk.writeComponent(ecs::ComponentId::AABB));
    auto *masses = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Mass));
    positions[row] = FVec3{math::Fixed32::fromFloat(x), math::Fixed32::fromFloat(y), math::Fixed32::fromFloat(z)};
    velocities[row] = FVec3{math::Fixed32{}, math::Fixed32::fromFloat(-2.0f), math::Fixed32{}};
    aabbs[row] = FVec3{math::Fixed32::fromFloat(0.5f), math::Fixed32::fromFloat(0.5f), math::Fixed32::fromFloat(0.5f)};
    masses[row] = math::Fixed32::fromFloat(mass);
    return id;
}

FVec3 readVec(ecs::Registry &registry, ecs::EntityId id, ecs::ComponentId component)
{
    core::u32 row = 0u;
    ecs::Chunk *chunk = registry.chunkOf(id, row);
    if (chunk == nullptr)
        return FVec3{};
    // The WRITE side: that is where a system writes, and this world's systems never
    // trigger a phase swap, so the front buffer still holds what the entity was born
    // with.
    auto *values = static_cast<FVec3 *>(chunk->writeComponent(component));
    return values == nullptr ? FVec3{} : values[row];
}

FVec3 positionOf(ecs::Registry &registry, ecs::EntityId id)
{
    return readVec(registry, id, ecs::ComponentId::Position);
}

FVec3 velocityOf(ecs::Registry &registry, ecs::EntityId id)
{
    return readVec(registry, id, ecs::ComponentId::Velocity);
}

} // namespace

int main()
{
    std::printf("=== heightfield collision ===\n\n");

    constexpr core::u32 kSize = 32u;
    const procgen::Heightfield field = ramp(kSize);

    // ── 1. A falling body lands ──────────────────────────────────────────────
    std::printf("-- a falling body lands and stops falling --\n");
    {
        ecs::Registry registry;
        // Cell 16 of the ramp stands at 8 - 8 = 0, so a body dropped from above it
        // must come to rest at 0 + half its height.
        const ecs::EntityId body = spawnBody(registry, 0.0f, 20.0f, 0.0f, 1.0f);
        engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
        collision.execute(1.0f / 60.0f);
        check(collision.resting() == 0u, "a body above the ground is not in contact");

        // Put it under the surface and step again: it must be lifted onto it.
        {
            core::u32 row = 0u;
            ecs::Chunk *chunk = registry.chunkOf(body, row);
            auto *positions = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            positions[row].y = math::Fixed32::fromFloat(-4.0f);
        }
        collision.execute(1.0f / 60.0f);
        check(collision.resting() == 1u, "a body below the ground is in contact after one step");
        const FVec3 at = positionOf(registry, body);
        check(at.y.toFloat() > -1.0f, "and it was lifted onto the surface");
        check(velocityOf(registry, body).y.toFloat() >= 0.0f, "its downward velocity was cancelled");
        std::printf("     landed at y=%.3f\n", at.y.toFloat());
    }

    // ── 2. Zero mass is immovable ────────────────────────────────────────────
    std::printf("\n-- zero mass means immovable --\n");
    {
        ecs::Registry registry;
        const ecs::EntityId tree = spawnBody(registry, 0.0f, -4.0f, 0.0f, 0.0f);
        engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
        for (core::u32 i = 0u; i < 30u; ++i)
            collision.execute(1.0f / 60.0f);
        const FVec3 at = positionOf(registry, tree);
        const FVec3 velocity = velocityOf(registry, tree);
        check(at.y.toFloat() == -4.0f, "a massless body is never corrected");
        check(velocity.x.raw() == 0 && velocity.z.raw() == 0, "and never handed a downhill slide");
        check(collision.resting() == 0u, "so it never counts as resting either");
    }

    // ── 3. The slide follows the gradient ────────────────────────────────────
    std::printf("\n-- a resting body slides downhill and loses speed --\n");
    {
        ecs::Registry registry;
        const ecs::EntityId rock = spawnBody(registry, -8.0f, -20.0f, 0.0f, 1.0f);
        engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
        const float startX = positionOf(registry, rock).x.toFloat();
        collision.execute(1.0f / 60.0f);
        const float pushed = velocityOf(registry, rock).x.toFloat();
        // The ramp falls toward +x, so the push must be toward +x.
        check(pushed > 0.0f, "the push is downhill, not uphill");
        std::printf("     one step of slide gave vx=%+.4f from x=%+.1f\n", pushed, startX);

        // Friction: with no new slope contribution the horizontal speed must decay.
        // A flat field is the clean way to ask, since the gradient term is then zero.
        const procgen::Heightfield flat{kSize, kSize, math::Fixed32{}};
        ecs::Registry level;
        const ecs::EntityId sliding = spawnBody(level, 0.0f, -4.0f, 0.0f, 1.0f);
        {
            core::u32 row = 0u;
            ecs::Chunk *chunk = level.chunkOf(sliding, row);
            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            velocities[row].x = math::Fixed32::fromFloat(4.0f);
        }
        engine::systems::HeightfieldCollisionSystem onFlat{level, flat, 1.0f};
        onFlat.execute(1.0f / 60.0f);
        const float after = velocityOf(level, sliding).x.toFloat();
        std::printf("     4.000 became %.4f on flat ground\n", after);
        check(after < 4.0f && after > 0.0f, "friction bleeds horizontal speed without reversing it");
    }

    // ── 4. Bodies stay over the map ──────────────────────────────────────────
    std::printf("\n-- a body never leaves the map --\n");
    {
        ecs::Registry registry;
        // Well past the eastern edge: half the map is 16 units, so 40 is outside.
        const ecs::EntityId strayBody = spawnBody(registry, 40.0f, -4.0f, 40.0f, 1.0f);
        engine::systems::HeightfieldCollisionSystem collision{registry, field, 1.0f};
        collision.execute(1.0f / 60.0f);
        const FVec3 at = positionOf(registry, strayBody);
        std::printf("     (40.0, 40.0) became (%.1f, %.1f)\n", at.x.toFloat(), at.z.toFloat());
        check(at.x.toFloat() <= 15.0f && at.z.toFloat() <= 15.0f, "a body outside the map is pulled back over it");
        const FVec3 velocity = velocityOf(registry, strayBody);
        check(velocity.x.raw() == 0 || velocity.z.raw() == 0, "and its outward velocity is cancelled");
    }

    // ── 5. An empty field is survivable ──────────────────────────────────────
    std::printf("\n-- an empty world does not crash --\n");
    {
        ecs::Registry registry;
        (void) spawnBody(registry, 0.0f, -4.0f, 0.0f, 1.0f);
        const procgen::Heightfield nothing;
        engine::systems::HeightfieldCollisionSystem collision{registry, nothing, 1.0f};
        collision.execute(1.0f / 60.0f);
        check(collision.resting() == 0u, "nothing to stand on means nothing resting");
    }

    std::printf("\n%s\n", failures == 0 ? "ALL PASS (0 failures)" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
