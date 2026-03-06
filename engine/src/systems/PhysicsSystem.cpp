/**
 * @file PhysicsSystem.cpp
 * @brief Dispatches physics step through IPhysicsBackend + WorldPartition.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/engine/systems/PhysicsSystem.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/math/FixedPoint.hpp>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::ComponentAccess kPhysicsAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Mass,     ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::AABB,     ecs::AccessMode::ReadOnly},
};

static const ecs::SystemDescriptor kPhysicsDesc{
    "Physics",
    ecs::SchedulePhase::Physics,
    std::span<const ecs::ComponentAccess>{kPhysicsAccesses}
};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct PhysicsSystem::Impl
{
    ecs::WorldPartition&      world;
    physics::IPhysicsBackend& backend;
    ecs::Registry&            registry;

    Impl(ecs::WorldPartition& w, physics::IPhysicsBackend& b, ecs::Registry& r)
        : world{w}, backend{b}, registry{r}
    {
    }
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

PhysicsSystem::PhysicsSystem(ecs::WorldPartition& world,
                             physics::IPhysicsBackend& backend,
                             ecs::Registry& registry)
    : _impl{std::make_unique<Impl>(world, backend, registry)}
{
}

PhysicsSystem::~PhysicsSystem() = default;

const ecs::SystemDescriptor& PhysicsSystem::descriptor() const noexcept
{
    return kPhysicsDesc;
}

void PhysicsSystem::execute(core::f32 dt)
{
    // 1. Run physics integration, collision, sleeping on all chunks
    [[maybe_unused]] auto res = _impl->backend.step(dt);

    // 2. Spatial migration: re-assign entities to correct cells after physics
    //    Iterate all partitions/chunks that have Position and update the
    //    spatial index. insertOrUpdate is a no-op if the entity's Morton
    //    code hasn't changed (fast early-exit).
    const auto& partitions = _impl->registry.partitions();
    for (const auto& partition : partitions)
    {
        if (!partition->archetype().has(ecs::ComponentId::Position))
            continue;

        for (const auto& chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
                continue;

            // Read from the write buffer (post-physics positions)
            auto* positions = static_cast<const math::Vec3<float>*>(
                chunk->writeComponent(ecs::ComponentId::Position));
            if (!positions)
                continue;

            auto entityIds = chunk->entities();

            for (core::u32 i = 0; i < count; ++i)
            {
                // Convert float position to Fixed32 for WorldPartition
                math::Vec3<math::Fixed32> fixedPos{
                    math::Fixed32::fromFloat(positions[i].x),
                    math::Fixed32::fromFloat(positions[i].y),
                    math::Fixed32::fromFloat(positions[i].z)
                };
                [[maybe_unused]] auto migRes = _impl->world.insertOrUpdate(entityIds[i], fixedPos);
            }
        }
    }

    // 3. WorldPartition::step() for GPU dispatch (future)
    _impl->world.step(dt);
}

} // namespace lpl::engine::systems
