/**
 * @file KernelPlatform.hpp
 * @brief Kernel (freestanding) implementations of the platform backends.
 *
 * These backends wrap the kernel C HAL (<kernel/hal/hal.h>) and are compiled
 * ONLY into libengine.a (the freestanding kernel build, LPL_TARGET_KERNEL=1).
 * The whole translation unit collapses to nothing on the hosted build, where
 * the GLFW/Vulkan/chrono backends are used instead.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_KERNEL_KERNELPLATFORM_HPP
#    define LPL_PLATFORM_KERNEL_KERNELPLATFORM_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL

#        include <lpl/platform/IPlatform.hpp>

namespace lpl::platform::kernel {

/** @brief Tick / timestamp backend over the kernel clock + rdtsc HAL. */
class KernelClockBackend final : public IClockBackend {
public:
    [[nodiscard]] core::u32 tickCount() const noexcept override;
    [[nodiscard]] core::u32 tickHertz() const noexcept override;
    [[nodiscard]] core::u64 timestampCounter() const noexcept override;
    [[nodiscard]] const char *name() const noexcept override { return "KernelClock"; }
};

/** @brief Software-LFB display backend over the framebuffer HAL. */
class KernelDisplayBackend final : public IDisplayBackend {
public:
    [[nodiscard]] bool querySurface(SurfaceDescriptor &outDescriptor) const noexcept override;
    void clear(core::u32 colorRgb) override;
    [[nodiscard]] core::u32 readPixel(core::u32 x, core::u32 y) const noexcept override;
    void present() override;
    [[nodiscard]] const char *name() const noexcept override { return "KernelDisplay(software-LFB)"; }
};

/** @brief Input backend over the PS/2 keyboard SPSC ring HAL. */
class KernelInputBackend final : public IInputBackend {
public:
    [[nodiscard]] bool tryPopCharacter(char &outCharacter) override;
    [[nodiscard]] core::u32 pendingCount() const noexcept override;
    [[nodiscard]] const char *name() const noexcept override { return "KernelInput(PS/2)"; }
};

/** @brief Pinned graphics-memory backend over the kernel HAL. */
class KernelGpuMemoryBackend final : public IGpuMemoryBackend {
public:
    [[nodiscard]] core::Expected<GpuAllocation> allocate(core::u32 sizeBytes, GpuMemoryFlags flags) override;
    void free(const GpuAllocation &allocation) override;
    [[nodiscard]] const char *name() const noexcept override { return "KernelGpuMemory(pinned)"; }
};

/**
 * @class KernelPlatform
 * @brief IPlatform aggregate owning the four kernel backends.
 */
class KernelPlatform final : public IPlatform {
public:
    [[nodiscard]] IClockBackend &clock() noexcept override { return _clock; }
    [[nodiscard]] IDisplayBackend &display() noexcept override { return _display; }
    [[nodiscard]] IInputBackend &input() noexcept override { return _input; }
    [[nodiscard]] IGpuMemoryBackend &gpuMemory() noexcept override { return _gpuMemory; }
    [[nodiscard]] const char *name() const noexcept override { return "KernelPlatform"; }

private:
    KernelClockBackend _clock;
    KernelDisplayBackend _display;
    KernelInputBackend _input;
    KernelGpuMemoryBackend _gpuMemory;
};

} // namespace lpl::platform::kernel

#    endif // LPL_TARGET_KERNEL

#endif // LPL_PLATFORM_KERNEL_KERNELPLATFORM_HPP
