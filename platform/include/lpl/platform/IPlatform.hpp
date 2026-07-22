/**
 * @file IPlatform.hpp
 * @brief Aggregate platform interface bundling the backend strategies.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_IPLATFORM_HPP
#    define LPL_PLATFORM_IPLATFORM_HPP

#    include "IClockBackend.hpp"
#    include "IDisplayBackend.hpp"
#    include "IGpuMemoryBackend.hpp"
#    include "IInputBackend.hpp"
#    include "IMemoryBackend.hpp"

namespace lpl::platform {

/**
 * @class IPlatform
 * @brief Dependency-injection seam bundling the platform backends.
 *
 * Engine::init() receives an IPlatform& and reaches the host (GLFW/Vulkan/
 * chrono) or the kernel (HAL) facilities exclusively through these backends,
 * so no engine subsystem includes a host or kernel header directly.
 */
class IPlatform {
public:
    virtual ~IPlatform() = default;

    [[nodiscard]] virtual IClockBackend &clock() noexcept = 0;
    [[nodiscard]] virtual IDisplayBackend &display() noexcept = 0;
    [[nodiscard]] virtual IInputBackend &input() noexcept = 0;
    [[nodiscard]] virtual IGpuMemoryBackend &gpuMemory() noexcept = 0;

    /**
     * @brief Source of the large blocks the engine's arenas are built on.
     *
     * Deliberately separate from gpuMemory(): that one hands out pinned,
     * GPU-attachable pages, this one hands out ordinary CPU memory the engine
     * bump-allocates from.
     */
    [[nodiscard]] virtual IMemoryBackend &memory() noexcept = 0;

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::platform

#endif // LPL_PLATFORM_IPLATFORM_HPP
