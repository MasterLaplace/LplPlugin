/**
 * @file GpuPhysicsBackend.hpp
 * @brief GPU-accelerated physics backend bridging physics ↔ compute.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-29
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_GPUPHYSICSBACKEND_HPP
#    define LPL_PHYSICS_GPUPHYSICSBACKEND_HPP

#    include <lpl/gpu/IComputeBackend.hpp>
#    include <lpl/physics/IPhysicsBackend.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ecs {
class Registry;
}

namespace lpl::physics {

/**
 * @class GpuPhysicsBackend
 * @brief Drives the @c physics_tick compute kernel over the ECS registry.
 *
 * This is the host-side bridge that makes the @c gpu module honest about its
 * relationship to @c physics: it implements @c physics::IPhysicsBackend (so the
 * Engine can select it interchangeably with @c CpuPhysicsBackend) and delegates
 * the integration pass to an @c IComputeBackend via @c dispatch("physics_tick").
 *
 * Scope: integration only (gravity + semi-implicit Euler), matching the legacy
 * CUDA kernel. Collision resolution and sleeping remain CPU/future work — the
 * GPU kernel does not implement them. The kernel consumes SoA (planar) float
 * arrays while the ECS stores AoS @c Vec3<Fixed32>, so each step converts to
 * float and deinterleaves into scratch buffers, dispatches, and reinterleaves
 * the results back into Fixed32. Those float results replace the bodies'
 * state, so a step through this backend is not bit-identical across machines.
 *
 * Lives alongside @c CpuPhysicsBackend in @c physics, but is compiled ONLY when
 * the @c cuda config is enabled: @c gpu is CUDA-host and cannot be freestanding,
 * so the kernel's freestanding @c libengine (which never enables cuda) builds
 * the CPU backend only and stays kernel-clean. There is no dependency cycle —
 * @c gpu depends on neither @c physics nor @c ecs. Only meaningfully active when
 * a real backend (CUDA) is wired; otherwise the compute backend reports
 * @c NotSupported. No CI job turns @c cuda on, so no CI build compiles this
 * backend either.
 */
class GpuPhysicsBackend final : public IPhysicsBackend {
public:
    /**
     * @brief Constructs the bridge.
     * @param registry ECS registry holding Position/Velocity/Mass components.
     * @param compute  Compute backend that services @c "physics_tick".
     */
    GpuPhysicsBackend(ecs::Registry &registry, gpu::IComputeBackend &compute) noexcept;
    ~GpuPhysicsBackend() override = default;

    [[nodiscard]] core::Expected<void> init() override;
    [[nodiscard]] core::Expected<void> step(core::f32 dt) override;
    void shutdown() override;
    [[nodiscard]] const char *name() const noexcept override;

private:
    /** @brief Planar SoA scratch + device pointers reused across chunks. */
    struct DeviceScratch {
        lpl::pmr::vector<float> hostX, hostY, hostZ;    /**< Positions, staged on the host. */
        lpl::pmr::vector<float> hostVX, hostVY, hostVZ; /**< Velocities, staged on the host. */
        lpl::pmr::vector<float> hostMass;               /**< Masses, staged on the host. */
        void *dPosX = nullptr, *dPosY = nullptr, *dPosZ = nullptr;
        void *dVelX = nullptr, *dVelY = nullptr, *dVelZ = nullptr;
        void *dFrcX = nullptr, *dFrcY = nullptr, *dFrcZ = nullptr;
        void *dMass = nullptr;
        core::u32 capacity = 0;
    };

    /**
     * @brief Grows the device/host scratch to hold at least @p count entities.
     */
    [[nodiscard]] core::Expected<void> ensureCapacity(core::u32 count);

    /**
     * @brief Releases all device allocations.
     */
    void releaseScratch() noexcept;

    ecs::Registry &_registry;
    gpu::IComputeBackend &_compute;
    DeviceScratch _scratch;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_GPUPHYSICSBACKEND_HPP
