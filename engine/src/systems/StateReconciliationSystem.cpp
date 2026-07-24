/**
 * @file StateReconciliationSystem.cpp
 * @brief Client-side: applies authoritative state snapshots from server.
 *
 * Drains StateUpdateEvent queue and reconciles each received entity's
 * position/size/health with the local world.  Missing entities are created;
 * existing entities are overwritten (authoritative model).
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/engine/systems/StateReconciliationSystem.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/net/protocol/EntityDelta.hpp>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::ComponentAccess kReconcileAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::AABB,     ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Health,   ecs::AccessMode::ReadWrite},
};

static const ecs::SystemDescriptor kReconcileDesc{"StateReconciliation", ecs::SchedulePhase::PrePhysics,
                                                  std::span<const ecs::ComponentAccess>{kReconcileAccesses}};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct StateReconciliationSystem::Impl {
    EventQueues &queues;
    ecs::WorldPartition &world;
    ecs::Registry &registry;
    core::u64 lastAppliedSeq{0}; ///< Highest snapshot sequence applied (§6.2.5 ack source).

    Impl(EventQueues &q, ecs::WorldPartition &w, ecs::Registry &reg) : queues{q}, world{w}, registry{reg} {}

    /// Create-or-overwrite one authoritative entity snapshot into the local world.
    /// Shared by the full snapshot (states) and by the AOI spawn/delta streams,
    /// which carry the same StateEntity payload and reconcile identically.
    void applyEntity(const StateEntity &ent)
    {
        using namespace net::protocol;
        auto entityId = ecs::EntityId{ent.id};
        const core::u8 mask = ent.fieldMask;

        // Write into SoA chunk buffers (Position, AABB, Health)
        auto refResult = registry.resolve(entityId);
        if (!refResult.has_value())
        {
            // Entity doesn't exist locally yet — create it UNDER THE SERVER'S ID.
            // Minting a fresh local id here was the bug: the entity then lived
            // under an id that never matched the server id the next snapshot
            // carried, so resolve() failed forever and every tick spawned a new
            // ghost that the write loop below (which matches on the server id)
            // never populated. Adopting the id makes the same entity resolve and
            // update in place — the legacy findEntity(publicId) semantics.
            ecs::Archetype arch;
            arch.add(ecs::ComponentId::Position);
            arch.add(ecs::ComponentId::Velocity);
            arch.add(ecs::ComponentId::Mass);
            arch.add(ecs::ComponentId::AABB);
            arch.add(ecs::ComponentId::Health);

            auto createRes = registry.createEntityWithId(entityId, arch);
            if (!createRes.has_value())
                return;

            refResult = registry.resolve(entityId);
            if (!refResult.has_value())
                return;
        }

        auto ref = refResult.value();
        const auto &partitions = registry.partitions();
        // Find the partition that owns this entity
        for (const auto &part : partitions)
        {
            if (!part)
                continue;

            const auto &chunks = part->chunks();
            if (ref.chunkIndex >= static_cast<core::u32>(chunks.size()))
                continue;

            auto &chunk = *chunks[ref.chunkIndex];

            // Verify this chunk actually holds the entity
            auto entityIds = chunk.entities();
            bool found = false;
            for (core::u32 i = 0; i < chunk.count(); ++i)
            {
                if (entityIds[i].raw() == entityId.raw())
                {
                    found = true;

                    // Merge only the fields the snapshot actually carries onto
                    // what the client already holds (§6.2.5): a full snapshot /
                    // spawn has every bit set and overwrites everything; a field
                    // delta touches only the changed components and leaves the
                    // rest as they stood. Position needs the MERGED value in full,
                    // both for the SoA write and for the spatial index, so read
                    // the current one first and replace only the masked axes.
                    auto *wpos =
                        static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position));
                    math::Vec3<math::Fixed32> merged =
                        wpos ? wpos[i] :
                               math::Vec3<math::Fixed32>{math::Fixed32::zero(), math::Fixed32::zero(),
                                                         math::Fixed32::zero()};
                    if (mask & FieldPosX)
                        merged.x = math::Fixed32::fromFloat(ent.pos.x);
                    if (mask & FieldPosY)
                        merged.y = math::Fixed32::fromFloat(ent.pos.y);
                    if (mask & FieldPosZ)
                        merged.z = math::Fixed32::fromFloat(ent.pos.z);
                    if (wpos)
                        wpos[i] = merged;

                    if (auto *wsize =
                            static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
                    {
                        if (mask & FieldSizeX)
                            wsize[i].x = math::Fixed32::fromFloat(ent.size.x);
                        if (mask & FieldSizeY)
                            wsize[i].y = math::Fixed32::fromFloat(ent.size.y);
                        if (mask & FieldSizeZ)
                            wsize[i].z = math::Fixed32::fromFloat(ent.size.z);
                    }

                    if (mask & FieldHp)
                        if (auto *whp = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
                            whp[i] = ent.hp;

                    // Spatial index tracks the merged authoritative position.
                    [[maybe_unused]] auto res = world.insertOrUpdate(entityId, merged);

                    break;
                }
            }
            if (found)
                break;
        }
    }

    /// Remove one entity the server said left the client's interest radius.
    void removeEntity(core::u32 rawId)
    {
        auto entityId = ecs::EntityId{rawId};
        [[maybe_unused]] auto spatialRes = world.remove(entityId);
        [[maybe_unused]] auto destroyRes = registry.destroyEntity(entityId);
    }
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

StateReconciliationSystem::StateReconciliationSystem(EventQueues &queues, ecs::WorldPartition &world,
                                                     ecs::Registry &registry)
    : _impl{std::make_unique<Impl>(queues, world, registry)}
{
}

StateReconciliationSystem::~StateReconciliationSystem() = default;

const ecs::SystemDescriptor &StateReconciliationSystem::descriptor() const noexcept { return kReconcileDesc; }

core::u64 StateReconciliationSystem::lastAppliedSnapshotSeq() const noexcept { return _impl->lastAppliedSeq; }

void StateReconciliationSystem::execute(core::f32 /*dt*/)
{
    // Full snapshots (non-AOI server) and AOI spawn/delta all reconcile the same
    // way: create-or-overwrite each entity from its authoritative snapshot.
    const auto noteSeq = [&](core::u64 seq) {
        if (seq > _impl->lastAppliedSeq)
            _impl->lastAppliedSeq = seq;
    };
    for (const auto &ev : _impl->queues.states.drain())
    {
        for (const auto &ent : ev.entities)
            _impl->applyEntity(ent);
        noteSeq(ev.seq);
    }
    for (const auto &ev : _impl->queues.spawns.drain())
    {
        for (const auto &ent : ev.entities)
            _impl->applyEntity(ent);
        noteSeq(ev.seq);
    }
    for (const auto &ev : _impl->queues.deltas.drain())
    {
        for (const auto &ent : ev.entities)
            _impl->applyEntity(ent);
        noteSeq(ev.seq);
    }
    // AOI despawn: entities that left the interest radius are removed locally.
    for (const auto &ev : _impl->queues.destroys.drain())
    {
        for (const core::u32 id : ev.ids)
            _impl->removeEntity(id);
    }
}

} // namespace lpl::engine::systems
