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
 * A step follows the order of Box2D and Jolt: the velocities first, gravity and damping; then the
 * contacts of every body of the world together, found once and solved by sequential impulses; then
 * the positions, by the velocities; then a correction of the overlap that remains; then sleep.
 * Constants: gravity -9.81, damping 0.995, restitution 0.5 above one unit a second, sleep threshold²
 * 0.01, sleep frames 30, velocity iterations 4, position iterations 4.
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
     * @brief Advances every body by @p dt: velocities, contacts, positions, overlap, then sleep.
     *
     * Integration and sleep walk each chunk on its own, since neither reads another body.
     * Contacts gather the bodies of every chunk first: see @ref resolveCollisionsWorld.
     *
     * @param dt The step, in seconds.
     * @return Always a success.
     */
    [[nodiscard]] core::Expected<void> step(core::f32 dt) override;

    void shutdown() override;
    [[nodiscard]] const char *name() const noexcept override;

private:
    /**
     * @brief The velocities of a step, before any contact: gravity on a body with mass, then damping.
     *
     * Blocks of four awake bodies damp through SimdFixed4, bit-identical to the scalar path
     * (test_simd_fixed_parity), so the result does not depend on the width; gravity, a branch per
     * body, stays scalar. A block that holds a sleeping body, and the bodies after the last block, go
     * one at a time.
     *
     * @param entities   The chunk's entities.
     * @param velocities Their velocities.
     * @param masses     Their masses.
     * @param count      How many.
     * @param dt         The step, in seconds.
     */
    void integrateVelocities(const ecs::EntityId *entities, math::Vec3<math::Fixed32> *velocities,
                             const math::Fixed32 *masses, core::u32 count, math::Fixed32 dt) const noexcept;

    /**
     * @brief The positions of a step, once the contacts have set the velocities: the move by velocity
     *        × dt, then, below the ground's half height, the return to it, and the bounce of a body
     *        that fell faster than one unit a second.
     *
     * Blocks of four awake bodies move through SimdFixed4, as in @ref integrateVelocities.
     *
     * @param entities   The chunk's entities.
     * @param positions  Their positions.
     * @param velocities Their velocities.
     * @param count      How many.
     * @param dt         The step, in seconds.
     */
    void integratePositions(const ecs::EntityId *entities, math::Vec3<math::Fixed32> *positions,
                            math::Vec3<math::Fixed32> *velocities, core::u32 count, math::Fixed32 dt) const noexcept;

    /**
     * @brief Finds the step's contacts over every physics entity in the world, not one chunk, and
     *        solves their velocities by sequential impulses.
     *
     * A contact is a body within 1/64 of the ground, or two boxes within 1/64 of each other across one
     * face, along the axis of least overlap: a box never turns, so a contact's normal is one axis, and
     * along the other two nothing changes. A contact across a gap is speculative: it lets the pair
     * close the gap within the step and no more. Two boxes that overlap wake each other; a pair of
     * sleepers is skipped, and across a gap a sleeper holds still.
     *
     * Each velocity iteration gives every contact, in order, the impulse that brings its normal speed
     * to its target, and keeps the sum of a contact's impulses at zero or more (Catto, GDC 2006). The
     * target is a bounce of half the approach speed above one unit a second, zero below it, or the
     * closing of a gap. An iteration that changes no impulse stops them. Each contact between two
     * bodies with a positive impulse then publishes a CollisionEvent: a gameplay signal and not
     * authoritative state, so its normal and its magnitude are floats.
     *
     * Up to 32 bodies, every pair is tried. Past 32 the pairs come from neighbour lists, kept from
     * step to step until a step finds a body beyond a margin of where they found it, and made again
     * then, through a grid or an octree, which list the same bodies.
     *
     * Chunks group entities by archetype and cap out at 256 (@ref
     * ecs::Chunk::kChunkCapacity), not by spatial region — two entities that
     * happen to sit in different chunks (different archetype, or the same
     * archetype split across chunks once it exceeds 256) still occupy the same
     * physical space and must be able to collide.
     *
     * @param inverseDt One over the step, in seconds.
     */
    void resolveCollisionsWorld(math::Fixed32 inverseDt) const noexcept;

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
