/**
 * @file MovementSystem.cpp
 * @brief Computes per-entity velocity from input state (WASD + neural BCI).
 *
 * Mirrors the legacy InputManager::computeMovementVelocity() pipeline:
 *   1. Neural concentration scales speed [0.70x .. 1.30x]
 *   2. WASD produces horizontal velocity
 *   3. Blink detection triggers jump (rising-edge, grounded-gated)
 *
 * The result is written back into the entity's velocity component via
 * WorldPartition::insertOrUpdate so the Physics system picks it up.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/engine/systems/MovementSystem.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::ComponentAccess kMovementAccesses[] = {
    {ecs::ComponentId::Velocity,      ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::InputSnapshot, ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::BciInput,      ecs::AccessMode::ReadOnly },
};

static const ecs::SystemDescriptor kMovementDesc{"Movement", ecs::SchedulePhase::PrePhysics,
                                                 std::span<const ecs::ComponentAccess>{kMovementAccesses}};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct MovementSystem::Impl {
    input::InputManager &inputManager;
    ecs::Registry &registry;

    Impl(input::InputManager &im, ecs::Registry &reg) : inputManager{im}, registry{reg} {}
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

MovementSystem::MovementSystem(input::InputManager &inputManager, ecs::Registry &registry)
    : _impl{std::make_unique<Impl>(inputManager, registry)}
{
}

MovementSystem::~MovementSystem() = default;

const ecs::SystemDescriptor &MovementSystem::descriptor() const noexcept { return kMovementDesc; }

void MovementSystem::execute(core::f32 /*dt*/)
{
    // Iterate all chunks that have Velocity + entities known to InputManager.
    // For each entity with input registered, compute movement velocity from
    // its current input state (WASD + neural BCI), write to the back buffer.
    //
    // Legacy equivalent: Systems.hpp "movement" lambda calling
    //   computeMovementVelocity() per entity and writing into _velocities[writeIdx].

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

            auto *velocities = static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            if (!velocities)
                continue;

            // Get entity IDs from the chunk
            auto entityIds = chunk->entities();

            // Optional: SleepState for wake-on-input
            auto *sleepStates = part->archetype().has(ecs::ComponentId::SleepState) ?
                                    static_cast<core::u8 *>(chunk->writeComponent(ecs::ComponentId::SleepState)) :
                                    nullptr;

            for (core::u32 i = 0; i < count; ++i)
            {
                const core::u32 eid = entityIds[i].raw();

                // Only process entities that have input registered
                if (!_impl->inputManager.hasEntity(eid))
                    continue;

                // Read current velocity from the front (read) buffer
                auto currentVel = velocities[i]; // back buffer was synced from front in swapBuffers

                // Compute movement velocity from current input state (WASD + BCI)
                auto vel = _impl->inputManager.computeMovementVelocity(eid, currentVel);

                velocities[i] = vel;

                // Wake entity if velocity is non-zero and it was sleeping
                float speedSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
                if (speedSq > 0.0001f && sleepStates && sleepStates[i] != 0)
                {
                    sleepStates[i] = 0;
                }
            }
        }
    }
}

} // namespace lpl::engine::systems
