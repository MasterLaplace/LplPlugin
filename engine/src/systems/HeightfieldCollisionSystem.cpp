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
            auto *positions = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            const auto *aabb = static_cast<const FVec3 *>(chunk->latestComponent(ecs::ComponentId::AABB));
            const auto *mass = static_cast<const math::Fixed32 *>(chunk->latestComponent(ecs::ComponentId::Mass));
            if (positions == nullptr || velocities == nullptr)
                continue;

            const core::u32 count = chunk->count();
            for (core::u32 i = 0u; i < count; ++i)
            {
                if (mass != nullptr && mass[i].raw() == 0)
                    continue;
                const core::f32 half = aabb != nullptr ? aabb[i].y.toFloat() * 0.5f : 0.25f;

                const core::f32 worldX = positions[i].x.toFloat() + halfWidth;
                const core::f32 worldZ = positions[i].z.toFloat() + halfDepth;
                const core::i32 cellX = static_cast<core::i32>(worldX / _cellSize);
                const core::i32 cellZ = static_cast<core::i32>(worldZ / _cellSize);

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

                ++_resting;
                positions[i].y = math::Fixed32::fromFloat(ground);
                if (velocities[i].y.toFloat() < 0.0f)
                    velocities[i].y = math::Fixed32::zero();

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
