/**
 * @file PhysicsSystem.hpp
 * @brief ECS system that advances the physics simulation by one tick.
 *
 * Pipeline per tick:
 *   1. IPhysicsBackend::step()  — integrate, collide, sleep
 *   2. WorldPartition::step()   — migration + GC (future)
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

#    include <memory>

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
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_PHYSICSSYSTEM_HPP
