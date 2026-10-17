#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/engine/systems/MovementSystem.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::engine::systems {

static const ecs::ComponentAccess kMovementAccesses[] = {
    {ecs::ComponentId::Velocity,      ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::InputSnapshot, ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::BciInput,      ecs::AccessMode::ReadOnly },
};

static const ecs::SystemDescriptor kMovementDesc{"Movement", ecs::SchedulePhase::PrePhysics,
                                                 std::span<const ecs::ComponentAccess>{kMovementAccesses}};

struct MovementSystem::Impl {
    input::InputManager &inputManager;
    ecs::Registry &registry;

    Impl(input::InputManager &im, ecs::Registry &reg) : inputManager{im}, registry{reg} {}
};

MovementSystem::MovementSystem(input::InputManager &inputManager, ecs::Registry &registry)
    : _impl{pmr::make_unique<Impl>(inputManager, registry)}
{
}

MovementSystem::~MovementSystem() = default;

const ecs::SystemDescriptor &MovementSystem::descriptor() const noexcept { return kMovementDesc; }

void MovementSystem::execute(core::f32 /*dt*/)
{
    const auto &partitions = _impl->registry.partitions();
    for (const auto &part : partitions)
    {
        if (!part)
            continue;

        if (!part->archetype().has(ecs::ComponentId::Velocity))
            continue;

        for (const auto &chunk : part->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
                continue;

            auto *velocities =
                static_cast<math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            if (!velocities)
                continue;

            auto entityIds = chunk->entities();

            auto *sleepStates = part->archetype().has(ecs::ComponentId::SleepState) ?
                                    static_cast<core::u8 *>(chunk->writeComponent(ecs::ComponentId::SleepState)) :
                                    nullptr;

            for (core::u32 i = 0; i < count; ++i)
            {
                const core::u32 eid = entityIds[i].raw();

                if (!_impl->inputManager.hasEntity(eid))
                    continue;

                const math::Vec3<math::Fixed32> vel = _impl->inputManager.computeMovementVelocity(eid, velocities[i]);
                velocities[i] = vel;

                const math::Fixed32 speedSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
                if (speedSq > math::Fixed32::fromFloat(0.0001f) && sleepStates && sleepStates[i] != 0)
                {
                    sleepStates[i] = 0;
                }
            }
        }
    }
}

} // namespace lpl::engine::systems
