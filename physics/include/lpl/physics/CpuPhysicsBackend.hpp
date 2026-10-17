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
 * A step integrates each chunk, resolves the collisions of every body of the world together, then
 * puts to sleep the bodies that stopped. Constants: gravity -9.81, damping 0.995, restitution 0.5,
 * sleep threshold² 0.01, sleep frames 30, solver iterations 4.
 *
 * Every authoritative value it reads or writes, position, velocity, box and mass, is a Fixed32, and so
 * is all its arithmetic: a step gives the same words on the Linux oracle and in the i686 kernel.
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

    /**
     * @brief Advances every body by @p dt: integration, then collisions, then sleep.
     *
     * Integration and sleep walk each chunk on its own, since neither reads another body.
     * Collisions gather the bodies of every chunk first: see @ref resolveCollisionsWorld.
     *
     * @param dt The step, in seconds.
     * @return Always a success.
     */
    [[nodiscard]] core::Expected<void> step(core::f32 dt) override;

    void shutdown() override;
    [[nodiscard]] const char *name() const noexcept override;

private:
    /**
     * @brief Semi-implicit Euler integration + gravity + damping + ground.
     *
     * In this order, the same for every body: gravity on a body with mass, damping of the velocity,
     * the move by velocity × dt, then, below the ground's half height, the return to it and the
     * bounce of a downward velocity. Blocks of four awake bodies damp and move through SimdFixed4,
     * bit-identical to the scalar path (test_simd_fixed_parity), so the result does not depend on the
     * width; gravity and the bounce, a branch per body, stay scalar. A block that holds a sleeping
     * body, and the bodies after the last block, go one at a time.
     *
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
     * A contact's normal is one axis, so the resolver moves the pair and its
     * velocities along that axis alone: along the other two, the products of
     * the whole normal are zero and leave every word as it was.
     *
     * An iteration in which no pair overlaps changes nothing, so neither would
     * the ones after it: the resolver stops there.
     *
     * A pair of bodies that both sleep is skipped. A pair that overlaps wakes both and moves apart
     * along the axis of least penetration by the whole penetration, shared by inverse mass; unless
     * the two already separate, an impulse with restitution 0.5 then sets their velocities along it.
     * A pair of bodies both too heavy to move stays as it is. Each impulse publishes a
     * CollisionEvent: a gameplay signal and not authoritative state, so its normal and its
     * magnitude are floats.
     *
     * Up to 32 bodies, every pair is tested. Past 32 bodies the pairs come from an octree broad-phase. Each body's
     * neighbours are listed, and the lists are kept from step to step until a
     * step finds a body beyond a margin of where they found it. A body's search
     * reads its list while every body is within that margin, and the tree,
     * built only then, otherwise. Both give the same bodies in the same order.
     * The first iteration of a step does not check the margin again: the step
     * has just checked it, or made the lists, and nothing has moved since.
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
     *
     * The speed counts all three axes: the legacy one left out y, and froze bodies in free fall.
     *
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
