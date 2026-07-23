/**
 * @file World.inl
 * @brief Out-of-line definitions for engine::World.
 *
 * Included at the end of World.hpp. World is used header-only, in the
 * freestanding kernel as well as on the host, so its non-trivial definitions
 * go here instead of a translation unit the kernel build paths do not list.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_WORLD_INL
#    define LPL_ENGINE_WORLD_INL

namespace lpl::engine {

inline core::u64 World::stateHash() const noexcept
{
    core::u64 accumulator = 0;

    for (const auto &partition : _registry.partitions())
    {
        if (!partition)
            continue;

        const auto &archetype = partition->archetype();
        const bool hasPosition = archetype.has(ecs::ComponentId::Position);
        const bool hasVelocity = archetype.has(ecs::ComponentId::Velocity);
        if (!hasPosition && !hasVelocity)
            continue;

        for (const auto &chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
                continue;

            // The WRITE buffer, not the read one: chunks are double
            // buffered, the integrator advances the write side, and the read
            // side is last frame's snapshot kept for concurrent readers.
            // Folding the read side would digest stale state — and would sit
            // still for a whole tick, which is exactly how this was caught.
            // samples::CubePile's parity fold reads the same side.
            const auto *positions =
                hasPosition ?
                    static_cast<const math::Vec3<math::Fixed32> *>(
                        chunk->writeComponent(ecs::ComponentId::Position)) :
                    nullptr;
            const auto *velocities =
                hasVelocity ?
                    static_cast<const math::Vec3<math::Fixed32> *>(
                        chunk->writeComponent(ecs::ComponentId::Velocity)) :
                    nullptr;

            const auto entityIds = chunk->entities();

            for (core::u32 i = 0; i < count; ++i)
            {
                math::StateHash entityHash;
                entityHash.combine(entityIds[i].raw());

                if (positions != nullptr)
                {
                    entityHash.combine(positions[i].x.raw());
                    entityHash.combine(positions[i].y.raw());
                    entityHash.combine(positions[i].z.raw());
                }
                if (velocities != nullptr)
                {
                    entityHash.combine(velocities[i].x.raw());
                    entityHash.combine(velocities[i].y.raw());
                    entityHash.combine(velocities[i].z.raw());
                }

                accumulator += entityHash.digest();
            }
        }
    }

    return accumulator;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_WORLD_INL
