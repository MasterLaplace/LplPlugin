/**
 * @file LinuxPlatform.hpp
 * @brief Hosted (Linux) implementations of the platform backends.
 *
 * These backends are compiled ONLY into the hosted build (LPL_TARGET_KERNEL=0)
 * and provide the determinism oracle's view of the platform seam. To keep the
 * oracle dependency-free and bit-faithful to the kernel's software-LFB path,
 * the display backend is a host-memory linear framebuffer (the same model the
 * kernel exposes over the Multiboot framebuffer), not a GLFW window: the real
 * GLFW/Vulkan-swapchain surface arrives with the renderer split in a later
 * phase, behind this very same interface. The whole translation unit collapses
 * to nothing on the kernel build, where KernelPlatform is used instead.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_LINUX_LINUXPLATFORM_HPP
#    define LPL_PLATFORM_LINUX_LINUXPLATFORM_HPP

#    include <lpl/core/Platform.hpp>

#    if !LPL_TARGET_KERNEL

#        include <lpl/platform/IPlatform.hpp>

#        include <lpl/std/vector.hpp>

namespace lpl::platform::linux_host {

/** @brief Tick / timestamp backend over std::chrono::steady_clock + rdtsc. */
class LinuxClockBackend final : public IClockBackend {
public:
    LinuxClockBackend() noexcept;

    [[nodiscard]] core::u32 tickCount() const noexcept override;
    [[nodiscard]] core::u32 tickHertz() const noexcept override;
    [[nodiscard]] core::u64 timestampCounter() const noexcept override;
    [[nodiscard]] const char *name() const noexcept override { return "LinuxClock(steady_clock)"; }

private:
    core::u64 _epochNanoseconds;
};

/** @brief Software-LFB display backend over a host-memory framebuffer. */
class LinuxDisplayBackend final : public IDisplayBackend {
public:
    LinuxDisplayBackend(core::u32 width, core::u32 height);

    [[nodiscard]] bool querySurface(SurfaceDescriptor &outDescriptor) const noexcept override;
    void clear(core::u32 colorRgb) override;
    [[nodiscard]] core::u32 readPixel(core::u32 x, core::u32 y) const noexcept override;
    void present() override;
    [[nodiscard]] const char *name() const noexcept override { return "LinuxDisplay(software-LFB)"; }

private:
    core::u32 _width;
    core::u32 _height;
    lpl::pmr::vector<core::u32> _buffer;
};

/** @brief Input backend over an in-memory decoded-character ring. */
class LinuxInputBackend final : public IInputBackend {
public:
    [[nodiscard]] bool tryPopCharacter(char &outCharacter) override;
    [[nodiscard]] core::u32 pendingCount() const noexcept override;
    [[nodiscard]] const char *name() const noexcept override { return "LinuxInput(software-ring)"; }

    /** @brief Push a decoded character into the ring (GLFW callback / test feed). */
    void pushCharacter(char character);

private:
    lpl::pmr::vector<char> _ring;
    core::usize _head = 0u;
};

/** @brief Graphics-memory backend over aligned host allocation. */
class LinuxGpuMemoryBackend final : public IGpuMemoryBackend {
public:
    [[nodiscard]] core::Expected<GpuAllocation> allocate(core::u32 sizeBytes, GpuMemoryFlags flags) override;
    void free(const GpuAllocation &allocation) override;
    [[nodiscard]] const char *name() const noexcept override { return "LinuxGpuMemory(aligned-host)"; }
};

/**
 * @class LinuxPlatform
 * @brief IPlatform aggregate owning the four hosted backends.
 */
class LinuxPlatform final : public IPlatform {
public:
    explicit LinuxPlatform(core::u32 width = 1024u, core::u32 height = 768u);

    [[nodiscard]] IClockBackend &clock() noexcept override { return _clock; }
    [[nodiscard]] IDisplayBackend &display() noexcept override { return _display; }
    [[nodiscard]] IInputBackend &input() noexcept override { return _input; }
    [[nodiscard]] IGpuMemoryBackend &gpuMemory() noexcept override { return _gpuMemory; }
    [[nodiscard]] const char *name() const noexcept override { return "LinuxPlatform"; }

    /** @brief Access the concrete input backend to feed decoded characters. */
    [[nodiscard]] LinuxInputBackend &inputBackend() noexcept { return _input; }

private:
    LinuxClockBackend _clock;
    LinuxDisplayBackend _display;
    LinuxInputBackend _input;
    LinuxGpuMemoryBackend _gpuMemory;
};

} // namespace lpl::platform::linux_host

#    endif // !LPL_TARGET_KERNEL

#endif // LPL_PLATFORM_LINUX_LINUXPLATFORM_HPP
