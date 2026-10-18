#include <lpl/engine/systems/StateReconciliationSystem.hpp>

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/engine/EventQueue.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/net/protocol/EntityDelta.hpp>

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

const lpl::ecs::ComponentId kIds[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                      lpl::ecs::ComponentId::Mass, lpl::ecs::ComponentId::AABB,
                                      lpl::ecs::ComponentId::Health};
const lpl::ecs::Archetype kArch{kIds};

/**
 * @brief Reads back an entity's write-buffer Position (the buffer reconciliation wrote).
 */
[[nodiscard]] bool positionOf(lpl::ecs::Registry &reg, lpl::ecs::EntityId id, lpl::math::Vec3<lpl::math::Fixed32> &out)
{
    auto ref = reg.resolve(id);
    if (!ref.has_value())
        return false;
    auto &partition = reg.getOrCreatePartition(kArch);
    const auto &chunks = partition.chunks();
    if (ref.value().chunkIndex >= static_cast<lpl::core::u32>(chunks.size()))
        return false;
    auto &chunk = *chunks[ref.value().chunkIndex];
    const auto *positions = static_cast<const lpl::math::Vec3<lpl::math::Fixed32> *>(
        chunk.latestComponent(lpl::ecs::ComponentId::Position));
    if (!positions)
        return false;
    out = positions[ref.value().localIndex];
    return true;
}

lpl::engine::StateEntity makeSnap(lpl::core::u32 id, float x, float y, float z)
{
    lpl::engine::StateEntity e{};
    e.id = id;
    e.pos = {x, y, z};
    e.size = {1.0f, 1.0f, 1.0f};
    e.hp = 100;
    return e;
}

/**
 * @brief Registry::createEntityWithId semantics.
 *
 * @details The adopted server id is generation 3, slot 7. Adoption is idempotent: re-adopting the same id
 *          changes nothing. A local allocation must not hand out the reserved slot.
 */
void checkCreateEntityWithIdAdoptsTheServerId()
{
    lpl::ecs::Registry reg;
    const lpl::ecs::EntityId serverId{3u, 7u};

    auto created = reg.createEntityWithId(serverId, kArch);
    check(created.has_value() && created.value() == serverId, "createEntityWithId returns the adopted id");
    check(reg.resolve(serverId).has_value(), "the adopted id resolves");
    check(reg.isAlive(serverId), "the adopted entity is alive");
    check(reg.liveCount() == 1, "exactly one live entity after adoption");

    auto again = reg.createEntityWithId(serverId, kArch);
    check(again.has_value() && again.value() == serverId, "re-adopting the same id succeeds idempotently");
    check(reg.liveCount() == 1, "re-adoption does not create a duplicate");

    auto local = reg.createEntity(kArch);
    check(local.has_value() && local.value() != serverId, "a local createEntity never mints the adopted id");
    check(local.value().slot() != serverId.slot(), "and never reuses the adopted slot");
    check(reg.liveCount() == 2, "two live entities now");

    check(reg.destroyEntity(serverId).has_value(), "the adopted entity destroys");
    check(!reg.resolve(serverId).has_value(), "and no longer resolves");
    check(reg.liveCount() == 1, "one live entity remains");
}

/**
 * @brief Reconciliation no longer duplicates on repeated snapshots.
 *
 * @details The first snapshot carries two server entities the client has never seen. The second carries the
 *          SAME ids at new positions: the bug spawned two more ghosts there (liveCount 4) and never populated
 *          them. A despawn (AOI leave) then removes one entity locally.
 */
void checkRepeatedSnapshotsUpdateInPlace()
{
    lpl::ecs::Registry reg;
    lpl::ecs::WorldPartition world{lpl::math::Fixed32::fromFloat(10.0f), 4096};
    lpl::engine::EventQueues queues;
    lpl::engine::systems::StateReconciliationSystem reconcile{queues, world, reg};

    const auto idA = lpl::ecs::EntityId{0u, 10u};
    const auto idB = lpl::ecs::EntityId{0u, 11u};

    lpl::engine::StateUpdateEvent first{};
    first.entities.push_back(makeSnap(idA.raw(), 1.0f, 2.0f, 3.0f));
    first.entities.push_back(makeSnap(idB.raw(), 4.0f, 5.0f, 6.0f));
    queues.states.push(std::move(first));
    reconcile.execute(1.0f / 60.0f);

    check(reg.liveCount() == 2, "a first snapshot creates one entity per server id");
    check(reg.resolve(idA).has_value() && reg.resolve(idB).has_value(),
          "both are stored under their SERVER id (resolve by server id works)");

    lpl::math::Vec3<lpl::math::Fixed32> pos{};
    check(positionOf(reg, idA, pos) && pos.x == lpl::math::Fixed32::fromFloat(1.0f),
          "the snapshot position was actually written (not a hollow ghost)");

    lpl::engine::StateUpdateEvent second{};
    second.entities.push_back(makeSnap(idA.raw(), 7.0f, 8.0f, 9.0f));
    second.entities.push_back(makeSnap(idB.raw(), 10.0f, 11.0f, 12.0f));
    queues.states.push(std::move(second));
    reconcile.execute(1.0f / 60.0f);

    check(reg.liveCount() == 2, "a repeated snapshot updates in place — no duplicate entities");
    check(positionOf(reg, idA, pos) && pos.x == lpl::math::Fixed32::fromFloat(7.0f),
          "the same entity's position was updated by the second snapshot");

    lpl::engine::EntityDestroyEvent destroy{};
    destroy.ids.push_back(idA.raw());
    queues.destroys.push(std::move(destroy));
    reconcile.execute(1.0f / 60.0f);

    check(reg.liveCount() == 1, "a despawn removes exactly one entity");
    check(!reg.resolve(idA).has_value(), "the despawned entity no longer resolves");
    check(reg.resolve(idB).has_value(), "the other entity is untouched");
}

/**
 * @brief A field delta (§6.2.5) merges only its present fields.
 *
 * @details A full spawn, whose field mask defaults to all fields, seeds the client's baseline at (3, 4, 5).
 *          The delta then carries ONLY posX (the entity slid along X): posY/posZ in its payload are
 *          meaningless and must be ignored.
 */
void checkFieldDeltaMergesOnlyItsPresentFields()
{
    lpl::ecs::Registry reg;
    lpl::ecs::WorldPartition world{lpl::math::Fixed32::fromFloat(10.0f), 4096};
    lpl::engine::EventQueues queues;
    lpl::engine::systems::StateReconciliationSystem reconcile{queues, world, reg};

    const auto id = lpl::ecs::EntityId{0u, 20u};

    lpl::engine::EntitySpawnEvent spawn{};
    spawn.entities.push_back(makeSnap(id.raw(), 3.0f, 4.0f, 5.0f));
    queues.spawns.push(std::move(spawn));
    reconcile.execute(1.0f / 60.0f);

    lpl::math::Vec3<lpl::math::Fixed32> pos{};
    check(positionOf(reg, id, pos) && pos.x == lpl::math::Fixed32::fromFloat(3.0f) &&
              pos.z == lpl::math::Fixed32::fromFloat(5.0f),
          "a full spawn seeds the whole position");

    lpl::engine::StateDeltaEvent delta{};
    lpl::engine::StateEntity d = makeSnap(id.raw(), 9.0f, -999.0f, -999.0f);
    d.fieldMask = lpl::net::protocol::FieldPosX;
    delta.entities.push_back(d);
    queues.deltas.push(std::move(delta));
    reconcile.execute(1.0f / 60.0f);

    check(positionOf(reg, id, pos) && pos.x == lpl::math::Fixed32::fromFloat(9.0f), "the delta's X is applied");
    check(pos.y == lpl::math::Fixed32::fromFloat(4.0f) && pos.z == lpl::math::Fixed32::fromFloat(5.0f),
          "the absent Y and Z keep their held value (not overwritten by garbage)");
}

} // namespace

/**
 * @brief Client state reconciliation + id adoption test.
 *
 * @details Proves two things that were broken:
 *            1. Registry::createEntityWithId adopts a caller-supplied (server) id, so resolve() finds it, it is
 *               idempotent, and a later local createEntity never collides with the adopted slot. This is the
 *               modern form of the legacy EntityRegistry::registerEntity(publicId) sparse-set.
 *            2. StateReconciliationSystem no longer spawns a fresh ghost entity for every snapshot: re-sending
 *               the same server id updates the SAME entity in place (legacy findEntity(publicId) semantics),
 *               instead of leaking a duplicate each tick. A despawn removes it.
 */
int main()
{
    std::printf("== reconciliation & id adoption ==\n");

    checkCreateEntityWithIdAdoptsTheServerId();
    checkRepeatedSnapshotsUpdateInPlace();
    checkFieldDeltaMergesOnlyItsPresentFields();

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
