/**
 * @file IDisplayBackend.hpp
 * @brief Abstract display surface / present interface.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_IDISPLAYBACKEND_HPP
#    define LPL_PLATFORM_IDISPLAYBACKEND_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::platform {

/**
 * @struct SurfaceDescriptor
 * @brief Description of a presentable linear-framebuffer surface (KMS-like).
 *
 * Carries the virtual buffer pointer plus the physical address so a GPU
 * uploader can attach the backing later; ownership stays with the backend.
 */
struct SurfaceDescriptor {
    core::u32 *buffer = nullptr;       ///< Virtual address of the framebuffer.
    core::u32 physicalAddress = 0u;    ///< Physical address of the framebuffer.
    core::u32 width = 0u;              ///< Visible width in pixels.
    core::u32 height = 0u;             ///< Visible height in pixels.
    core::u32 pitch = 0u;              ///< Bytes per scanline.
    core::u8 bitsPerPixel = 0u;        ///< Bits per pixel.
};

/**
 * @class IDisplayBackend
 * @brief Strategy interface for display surfaces (GLFW + DRM/KMS equivalent).
 *
 * Concrete implementations: a GLFW/Vulkan-swapchain backend on Linux, a
 * software-LFB (later VirtIO-GPU) backend in-kernel. The backend owns only the
 * surface and present path; all rendering happens engine-side.
 */
class IDisplayBackend {
public:
    virtual ~IDisplayBackend() = default;

    /** @brief Query the active surface; false if none is available. */
    [[nodiscard]] virtual bool querySurface(SurfaceDescriptor &outDescriptor) const noexcept = 0;

    /** @brief Clear the whole surface to a packed 0x00RRGGBB color. */
    virtual void clear(core::u32 colorRgb) = 0;

    /** @brief Read back one pixel as 0x00RRGGBB (0 if unavailable). */
    [[nodiscard]] virtual core::u32 readPixel(core::u32 x, core::u32 y) const noexcept = 0;

    /** @brief Present the back buffer (atomic flip / scanout). */
    virtual void present() = 0;

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::platform

#endif // LPL_PLATFORM_IDISPLAYBACKEND_HPP
