/*
** LplPlugin — client state reconciliation + id adoption test
**
** Proves two things that were broken:
**   1. Registry::createEntityWithId adopts a caller-supplied (server) id, so
**      resolve() finds it, it is idempotent, and a later local createEntity never
**      collides with the adopted slot. This is the modern form of the legacy
**      EntityRegistry::registerEntity(publicId) sparse-set.
**   2. StateReconciliationSystem no longer spawns a fresh ghost entity for every
**      snapshot: re-sending the same server id updates the SAME entity in place
**      (legacy findEntity(publicId) semantics), instead of leaking a duplicate
**      each tick. A despawn removes it.
*/

#include <lpl/engine/systems/StateReconciliationSystem.hpp>

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/engine/EventQueue.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

#include <cstdio>

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

const ecs::ComponentId kIds[] = {ecs::ComponentId::Position, ecs::ComponentId::Velocity, ecs::ComponentId::Mass,
                                 ecs::ComponentId::AABB, ecs::ComponentId::Health};
const ecs::Archetype kArch{kIds};

// Reads back an entity's write-buffer Position (the buffer reconciliation wrote).
[[nodiscard]] bool positionOf(ecs::Registry &reg, ecs::EntityId id, math::Vec3<math::Fixed32> &out)
{
    auto ref = reg.resolve(id);
    if (!ref.has_value())
        return false;
    auto &partition = reg.getOrCreatePartition(kArch);
    const auto &chunks = partition.chunks();
    if (ref.value().chunkIndex >= static_cast<core::u32>(chunks.size()))
        return false;
    auto &chunk = *chunks[ref.value().chunkIndex];
    const auto *positions =
        static_cast<const math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position));
    if (!positions)
        return false;
    out = positions[ref.value().localIndex];
    return true;
}

engine::StateEntity makeSnap(core::u32 id, float x, float y, float z)
{
    engine::StateEntity e{};
    e.id = id;
    e.pos = {x, y, z};
    e.size = {1.0f, 1.0f, 1.0f};
    e.hp = 100;
    return e;
}

} // namespace

int main()
{
    std::printf("== reconciliation & id adoption ==\n");

    // ── Registry::createEntityWithId semantics ──────────────────────────────── //
    {
        ecs::Registry reg;
        const ecs::EntityId serverId{3u, 7u}; // generation 3, slot 7

        auto created = reg.createEntityWithId(serverId, kArch);
        check(created.has_value() && created.value() == serverId, "createEntityWithId returns the adopted id");
        check(reg.resolve(serverId).has_value(), "the adopted id resolves");
        check(reg.isAlive(serverId), "the adopted entity is alive");
        check(reg.liveCount() == 1, "exactly one live entity after adoption");

        // Idempotent: re-adopting the same id changes nothing.
        auto again = reg.createEntityWithId(serverId, kArch);
        check(again.has_value() && again.value() == serverId, "re-adopting the same id succeeds idempotently");
        check(reg.liveCount() == 1, "re-adoption does not create a duplicate");

        // A local allocation must not hand out the reserved slot.
        auto local = reg.createEntity(kArch);
        check(local.has_value() && local.value() != serverId, "a local createEntity never mints the adopted id");
        check(local.value().slot() != serverId.slot(), "and never reuses the adopted slot");
        check(reg.liveCount() == 2, "two live entities now");

        check(reg.destroyEntity(serverId).has_value(), "the adopted entity destroys");
        check(!reg.resolve(serverId).has_value(), "and no longer resolves");
        check(reg.liveCount() == 1, "one live entity remains");
    }

    // ── Reconciliation no longer duplicates on repeated snapshots ───────────── //
    {
        ecs::Registry reg;
        ecs::WorldPartition world{math::Fixed32::fromFloat(10.0f), 4096};
        engine::EventQueues queues;
        engine::systems::StateReconciliationSystem reconcile{queues, world, reg};

        const auto idA = ecs::EntityId{0u, 10u};
        const auto idB = ecs::EntityId{0u, 11u};

        // First snapshot: two server entities the client has never seen.
        engine::StateUpdateEvent first{};
        first.entities.push_back(makeSnap(idA.raw(), 1.0f, 2.0f, 3.0f));
        first.entities.push_back(makeSnap(idB.raw(), 4.0f, 5.0f, 6.0f));
        queues.states.push(std::move(first));
        reconcile.execute(1.0f / 60.0f);

        check(reg.liveCount() == 2, "a first snapshot creates one entity per server id");
        check(reg.resolve(idA).has_value() && reg.resolve(idB).has_value(),
              "both are stored under their SERVER id (resolve by server id works)");

        math::Vec3<math::Fixed32> pos{};
        check(positionOf(reg, idA, pos) && pos.x == math::Fixed32::fromFloat(1.0f),
              "the snapshot position was actually written (not a hollow ghost)");

        // Second snapshot for the SAME ids, new positions. The bug spawned two
        // more ghosts here (liveCount 4) and never populated them.
        engine::StateUpdateEvent second{};
        second.entities.push_back(makeSnap(idA.raw(), 7.0f, 8.0f, 9.0f));
        second.entities.push_back(makeSnap(idB.raw(), 10.0f, 11.0f, 12.0f));
        queues.states.push(std::move(second));
        reconcile.execute(1.0f / 60.0f);

        check(reg.liveCount() == 2, "a repeated snapshot updates in place — no duplicate entities");
        check(positionOf(reg, idA, pos) && pos.x == math::Fixed32::fromFloat(7.0f),
              "the same entity's position was updated by the second snapshot");

        // A despawn (AOI leave) removes the entity locally.
        engine::EntityDestroyEvent destroy{};
        destroy.ids.push_back(idA.raw());
        queues.destroys.push(std::move(destroy));
        reconcile.execute(1.0f / 60.0f);

        check(reg.liveCount() == 1, "a despawn removes exactly one entity");
        check(!reg.resolve(idA).has_value(), "the despawned entity no longer resolves");
        check(reg.resolve(idB).has_value(), "the other entity is untouched");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
