/**
 * @file IPhysicsBackend.hpp
 * @brief Abstract physics backend interface (Strategy pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_IPHYSICSBACKEND_HPP
#    define LPL_PHYSICS_IPHYSICSBACKEND_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Entity.hpp>

namespace lpl::physics {

/**
 * @class IPhysicsBackend
 * @brief Strategy interface for physics step implementations.
 *
 * Concrete backends:
 *   - @c CpuPhysicsBackend          — single-threaded CPU reference (also the
 *     only backend compiled into the freestanding kernel @c libengine).
 *   - @c GpuPhysicsBackend          — host-side bridge driving a
 *     @c gpu::IComputeBackend (CUDA) via @c dispatch("physics_tick"). Compiled
 *     only under the @c cuda config so the freestanding kernel build never pulls
 *     the CUDA-host @c gpu module. See @c GpuPhysicsBackend.hpp.
 */
class IPhysicsBackend {
public:
    virtual ~IPhysicsBackend() = default;

    /**
     * @brief Initializes the backend (allocates buffers, etc.).
     */
    [[nodiscard]] virtual core::Expected<void> init() = 0;

    /**
     * @brief Executes one fixed-tick physics step.
     * @param dt Delta-time in seconds (typically 1/144).
     */
    [[nodiscard]] virtual core::Expected<void> step(core::f32 dt) = 0;

    /**
     * @brief Tears down the backend.
     */
    virtual void shutdown() = 0;

    /**
     * @brief Returns a human-readable name for this backend.
     */
    [[nodiscard]] virtual const char *name() const noexcept = 0;

    /**
     * @brief Whether @p entity can have moved during the last step.
     *
     * A caller that files bodies by position, as the spatial index does, skips the ones that cannot
     * have moved: a body asleep through the whole step.
     *
     * @param entity The body.
     * @return False only when the backend knows the body did not move: true for a backend without
     *         sleep, which is what this default says.
     */
    [[nodiscard]] virtual bool movedLastStep([[maybe_unused]] ecs::EntityId entity) const noexcept { return true; }
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_IPHYSICSBACKEND_HPP
