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
    #define LPL_PHYSICS_CPUPHYSICSBACKEND_HPP

#include <lpl/physics/IPhysicsBackend.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::ecs { class Registry; }

namespace lpl::physics {

/**
 * @class CpuPhysicsBackend
 * @brief Full CPU physics backend ported from legacy engine/Partition.hpp.
 *
 * Pipeline (per chunk): integrate → collide (N² or octree) → sleep.
 * Constants: gravity -9.81, damping 0.995, restitution 0.5,
 *            sleep threshold² 0.01, sleep frames 30, solver iterations 4.
 */
class CpuPhysicsBackend final : public IPhysicsBackend,
                                 public core::NonCopyable<CpuPhysicsBackend>
{
public:
    /**
     * @brief Constructs with a reference to the ECS registry.
     * @param registry Registry containing Position, Velocity, Mass, AABB, etc.
     */
    explicit CpuPhysicsBackend(ecs::Registry& registry);
    ~CpuPhysicsBackend() override;

    [[nodiscard]] core::Expected<void> init() override;
    [[nodiscard]] core::Expected<void> step(core::f32 dt) override;
    void shutdown() override;
    [[nodiscard]] const char* name() const noexcept override;

private:
    /** @brief Semi-implicit Euler integration + gravity + damping + ground. */
    void integrateChunk(math::Vec3<float>* positions,
                        math::Vec3<float>* velocities,
                        const float* masses,
                        core::u32 count,
                        core::f32 dt) const noexcept;

    /** @brief AABB collision detection + impulse resolution (4 iterations). */
    void resolveCollisionsChunk(math::Vec3<float>* positions,
                                math::Vec3<float>* velocities,
                                const float* masses,
                                const math::Vec3<float>* sizes,
                                core::u32 count) const noexcept;

    /** @brief Sleeping detection: sleep after 30 frames below threshold. */
    void updateSleepingChunk(math::Vec3<float>* velocities,
                             core::u32 count) const noexcept;

    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_CPUPHYSICSBACKEND_HPP
