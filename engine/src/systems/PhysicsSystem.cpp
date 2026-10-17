#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/engine/systems/PhysicsSystem.hpp>
#include <lpl/math/FixedPoint.hpp>

namespace lpl::engine::systems {

static const ecs::ComponentAccess kPhysicsAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Mass,     ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::AABB,     ecs::AccessMode::ReadOnly },
};

static const ecs::SystemDescriptor kPhysicsDesc{"Physics", ecs::SchedulePhase::Physics,
                                                std::span<const ecs::ComponentAccess>{kPhysicsAccesses}};

struct PhysicsSystem::Impl {
    ecs::WorldPartition &world;
    physics::IPhysicsBackend &backend;
    ecs::Registry &registry;

    Impl(ecs::WorldPartition &w, physics::IPhysicsBackend &b, ecs::Registry &r) : world{w}, backend{b}, registry{r} {}
};

PhysicsSystem::PhysicsSystem(ecs::WorldPartition &world, physics::IPhysicsBackend &backend, ecs::Registry &registry)
    : _impl{pmr::make_unique<Impl>(world, backend, registry)}
{
}

PhysicsSystem::~PhysicsSystem() = default;

const ecs::SystemDescriptor &PhysicsSystem::descriptor() const noexcept { return kPhysicsDesc; }

void PhysicsSystem::execute(core::f32 dt)
{
    [[maybe_unused]] auto res = _impl->backend.step(dt);

    const auto &partitions = _impl->registry.partitions();
    for (const auto &partition : partitions)
    {
        if (!partition->archetype().has(ecs::ComponentId::Position))
            continue;

        for (const auto &chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
                continue;

            auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->latestComponent(ecs::ComponentId::Position));
            if (!positions)
                continue;

            auto entityIds = chunk->entities();

            for (core::u32 i = 0; i < count; ++i)
            {
                [[maybe_unused]] auto migRes = _impl->world.insertOrUpdate(entityIds[i], positions[i]);
            }
        }
    }

    _impl->world.step(dt);
}

} // namespace lpl::engine::systems
