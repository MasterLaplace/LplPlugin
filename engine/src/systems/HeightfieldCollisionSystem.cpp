/**
 * @file HeightfieldCollisionSystem.cpp
 * @brief Implementation of the bounded-heightfield collision pass.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/systems/HeightfieldCollisionSystem.hpp>

#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>

namespace lpl::engine::systems {

namespace {
using FVec3 = math::Vec3<math::Fixed32>;
}

HeightfieldCollisionSystem::HeightfieldCollisionSystem(ecs::Registry &registry, const procgen::Heightfield &terrain,
                                                       core::f32 cellSize) noexcept
    : _registry(registry), _terrain(terrain), _cellSize(cellSize)
{
}

void HeightfieldCollisionSystem::execute(core::f32 /*dt*/)
{
    ++_executions;
    // Reset per tick: what matters is how many bodies are in contact NOW, not how
    // many contacts have ever happened.
    _resting = 0u;
    if (_terrain.empty())
    {
        if (_executions == 1u)
            core::Log::warn("HeightfieldCollision: the terrain is empty, nothing to stand on");
        return;
    }

    const core::f32 halfWidth = static_cast<core::f32>(_terrain.width()) * _cellSize * 0.5f;
    const core::f32 halfDepth = static_cast<core::f32>(_terrain.depth()) * _cellSize * 0.5f;

    for (const auto &partition : _registry.partitions())
    {
        if (!partition)
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            // ALL FOUR from the write side. Mixing sides is a defect this system
            // carried invisibly for as long as it lived in an app: AABB and Mass were
            // read from the front buffer, which holds nothing until the first phase
            // swap publishes it — so on tick one every body read mass zero, was taken
            // for immovable, and nothing landed. One frame of nothing is not something
            // an eye catches; it is the first thing a test does.
            auto *positions = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            const auto *aabb = static_cast<const FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB));
            const auto *mass = static_cast<const math::Fixed32 *>(chunk->writeComponent(ecs::ComponentId::Mass));
            // A chunk without Velocity is scenery: it was never going to move, so
            // there is nothing to correct.
            if (positions == nullptr || velocities == nullptr)
                continue;

            const core::u32 count = chunk->count();
            for (core::u32 i = 0u; i < count; ++i)
            {
                // Zero mass means immovable. Correcting one to the ground would be
                // harmless, but the downhill slide below would hand it a velocity and
                // walk a tree off its own footing.
                if (mass != nullptr && mass[i].raw() == 0)
                    continue;
                const core::f32 half = aabb != nullptr ? aabb[i].y.toFloat() * 0.5f : 0.25f;

                const core::f32 worldX = positions[i].x.toFloat() + halfWidth;
                const core::f32 worldZ = positions[i].z.toFloat() + halfDepth;
                const core::i32 cellX = static_cast<core::i32>(worldX / _cellSize);
                const core::i32 cellZ = static_cast<core::i32>(worldZ / _cellSize);

                // Keep bodies over the map. A body nudged past the edge by a collision
                // would otherwise sail off and hang in empty space, which looks exactly
                // like the collision having failed.
                const core::f32 limitX = halfWidth - _cellSize;
                const core::f32 limitZ = halfDepth - _cellSize;
                if (positions[i].x.toFloat() < -limitX)
                {
                    positions[i].x = math::Fixed32::fromFloat(-limitX);
                    velocities[i].x = math::Fixed32::zero();
                }
                else if (positions[i].x.toFloat() > limitX)
                {
                    positions[i].x = math::Fixed32::fromFloat(limitX);
                    velocities[i].x = math::Fixed32::zero();
                }
                if (positions[i].z.toFloat() < -limitZ)
                {
                    positions[i].z = math::Fixed32::fromFloat(-limitZ);
                    velocities[i].z = math::Fixed32::zero();
                }
                else if (positions[i].z.toFloat() > limitZ)
                {
                    positions[i].z = math::Fixed32::fromFloat(limitZ);
                    velocities[i].z = math::Fixed32::zero();
                }

                const core::f32 ground = _terrain.clamped(cellX, cellZ).toFloat() + half;
                if (positions[i].y.toFloat() >= ground)
                    continue;

                // Land: sit on the surface and stop falling.
                ++_resting;
                positions[i].y = math::Fixed32::fromFloat(ground);
                if (velocities[i].y.toFloat() < 0.0f)
                    velocities[i].y = math::Fixed32::zero();

                // Slide along the downhill gradient, and lose speed doing it. Without
                // the slide a boulder simply stops where it landed and the terrain
                // might as well be a floor; with it, the map's drainage pattern becomes
                // visible in where things collect.
                const core::f32 left = _terrain.clamped(cellX - 1, cellZ).toFloat();
                const core::f32 right = _terrain.clamped(cellX + 1, cellZ).toFloat();
                const core::f32 back = _terrain.clamped(cellX, cellZ - 1).toFloat();
                const core::f32 front = _terrain.clamped(cellX, cellZ + 1).toFloat();

                velocities[i].x =
                    math::Fixed32::fromFloat(velocities[i].x.toFloat() * _friction + (left - right) * _slide);
                velocities[i].z =
                    math::Fixed32::fromFloat(velocities[i].z.toFloat() * _friction + (back - front) * _slide);
            }
        }
    }
}

} // namespace lpl::engine::systems
