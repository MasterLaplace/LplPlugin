/**
 * @file PhysicsSystem.hpp
 * @brief ECS system that advances the physics simulation by one tick.
 *
 * Pipeline per tick:
 *   1. IPhysicsBackend::step()  — integrate, collide, sleep
 *   2. spatial migration        — each entity re-filed in its cell
 *   3. WorldPartition::step()   — GPU dispatch (future)
 *
 * Used by both server and client.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_PHYSICSSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_PHYSICSSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/physics/IPhysicsBackend.hpp>

#    include <lpl/std/memory.hpp>

namespace lpl::engine::systems {

/**
 * @class PhysicsSystem
 * @brief Advances physics via IPhysicsBackend::step() + WorldPartition::step().
 */
class PhysicsSystem final : public ecs::ISystem {
public:
    /**
     * @param world    World partition for spatial migration/GC.
     * @param backend  Physics backend (CPU or GPU).
     * @param registry ECS registry for post-physics entity migration.
     */
    PhysicsSystem(ecs::WorldPartition &world, physics::IPhysicsBackend &backend, ecs::Registry &registry);
    ~PhysicsSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief Steps the backend, then re-files in its spatial cell every entity that can have moved.
     *
     * The backend runs integration, collision and sleeping on all chunks. The spatial
     * migration then walks every partition that has a Position and updates the spatial
     * index from the write buffer — the post-physics positions, already authoritative
     * Fixed32, with no float conversion. An entity the backend says did not move, asleep
     * through the step, is skipped: once a pile sleeps, filing it again was the largest
     * cost left in the server's tick. insertOrUpdate is a no-op when the entity's
     * cell has not changed (fast early-exit). WorldPartition::step() comes last, for the
     * GPU dispatch (future).
     */
    void execute(core::f32 dt) override;

private:
    struct Impl;
    pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_PHYSICSSYSTEM_HPP
