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

    Impl(EventQueues &q, ecs::WorldPartition &w, ecs::Registry &reg) : queues{q}, world{w}, registry{reg} {}
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

void StateReconciliationSystem::execute(core::f32 /*dt*/)
{
    auto events = _impl->queues.states.drain();

    for (const auto &ev : events)
    {
        for (const auto &ent : ev.entities)
        {
            auto entityId = ecs::EntityId{ent.id};

            // Convert float position to Fixed32 for spatial index
            auto fixedPos =
                math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(ent.pos.x), math::Fixed32::fromFloat(ent.pos.y),
                                          math::Fixed32::fromFloat(ent.pos.z)};

            // Update spatial index
            [[maybe_unused]] auto res = _impl->world.insertOrUpdate(entityId, fixedPos);

            // Write into SoA chunk buffers (Position, AABB, Health)
            auto refResult = _impl->registry.resolve(entityId);
            if (!refResult.has_value())
            {
                // Entity doesn't exist locally yet — create it
                ecs::Archetype arch;
                arch.add(ecs::ComponentId::Position);
                arch.add(ecs::ComponentId::Velocity);
                arch.add(ecs::ComponentId::Mass);
                arch.add(ecs::ComponentId::AABB);
                arch.add(ecs::ComponentId::Health);

                auto createRes = _impl->registry.createEntity(arch);
                if (!createRes.has_value())
                    continue;

                refResult = _impl->registry.resolve(createRes.value());
                if (!refResult.has_value())
                    continue;
            }

            auto ref = refResult.value();
            const auto &partitions = _impl->registry.partitions();
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

                        // Write Position
                        if (auto *wpos =
                                static_cast<math::Vec3<float> *>(chunk.writeComponent(ecs::ComponentId::Position)))
                            wpos[i] = {ent.pos.x, ent.pos.y, ent.pos.z};

                        // Write AABB
                        if (auto *wsize =
                                static_cast<math::Vec3<float> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
                            wsize[i] = {ent.size.x, ent.size.y, ent.size.z};

                        // Write Health
                        if (auto *whp = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
                            whp[i] = ent.hp;

                        break;
                    }
                }
                if (found)
                    break;
            }
        }
    }
}

} // namespace lpl::engine::systems
