/**
 * @file CpuPhysicsBackend.hpp
 * @brief CPU-only reference physics backend.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_CPUPHYSICSBACKEND_HPP
#    define LPL_PHYSICS_CPUPHYSICSBACKEND_HPP

#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/ecs/Entity.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/physics/IPhysicsBackend.hpp>

#    include <lpl/std/memory.hpp>

namespace lpl::ecs {
class Registry;
}

namespace lpl::physics {

/**
 * @class CpuPhysicsBackend
 * @brief Full CPU physics backend ported from legacy engine/Partition.hpp.
 *
 * Pipeline (per chunk): integrate → collide (N² or octree) → sleep.
 * Constants: gravity -9.81, damping 0.995, restitution 0.5,
 *            sleep threshold² 0.01, sleep frames 30, solver iterations 4.
 */
class CpuPhysicsBackend final : public IPhysicsBackend, public core::NonCopyable<CpuPhysicsBackend> {
public:
    /**
     * @brief Constructs with a reference to the ECS registry.
     * @param registry Registry containing Position, Velocity, Mass, AABB, etc.
     */
    explicit CpuPhysicsBackend(ecs::Registry &registry);
    ~CpuPhysicsBackend() override;

    [[nodiscard]] core::Expected<void> init() override;
    [[nodiscard]] core::Expected<void> step(core::f32 dt) override;
    void shutdown() override;
    [[nodiscard]] const char *name() const noexcept override;

private:
    /**
     * @brief Semi-implicit Euler integration + gravity + damping + ground.
     * @param entities List of entity IDs.
     * @param positions List of positions.
     * @param velocities List of velocities.
     * @param masses List of masses.
     * @param count Number of entities.
     * @param dt Delta time.
     */
    void integrateChunk(const ecs::EntityId *entities, math::Vec3<math::Fixed32> *positions,
                        math::Vec3<math::Fixed32> *velocities, const math::Fixed32 *masses, core::u32 count,
                        math::Fixed32 dt) const noexcept;

    /**
     * @brief AABB collision detection + impulse resolution (4 iterations), over
     *        every physics entity in the world (not scoped to one chunk).
     *
     * Chunks group entities by archetype and cap out at 256 (@ref
     * ecs::Chunk::kChunkCapacity), not by spatial region — two entities that
     * happen to sit in different chunks (different archetype, or the same
     * archetype split across chunks once it exceeds 256) still occupy the same
     * physical space and must be able to collide.
     */
    void resolveCollisionsWorld() const noexcept;

    /**
     * @brief Sleeping detection: sleep after 30 frames below threshold.
     * @param entities List of entity IDs.
     * @param velocities List of velocities.
     * @param count Number of entities.
     */
    void updateSleepingChunk(const ecs::EntityId *entities, math::Vec3<math::Fixed32> *velocities,
                             core::u32 count) const noexcept;

    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_CPUPHYSICSBACKEND_HPP
