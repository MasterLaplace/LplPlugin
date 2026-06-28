/**
 * @file Surface.hpp
 * @brief Blit an Image into a raw linear framebuffer (e.g. a display scanout).
 *
 * Kept platform-agnostic: the caller supplies a raw destination pointer plus its
 * geometry (from IDisplayBackend::querySurface / hal_display), so the image
 * module never depends on the platform layer. Pixels are copied opaquely — the
 * Image's packed 0xAARRGGBB matches a 0x00RRGGBB BGRX scanout (the alpha byte
 * lands in the ignored X slot).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_IMAGE_SURFACE_HPP
#    define LPL_IMAGE_SURFACE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/image/Image.hpp>

namespace lpl::image {

/**
 * @brief Copy @p source into a 32bpp linear framebuffer at (dstX, dstY).
 *
 * Clipped to the destination bounds. @p dstPitchBytes is the scanline stride in
 * bytes (>= dstWidth*4); rows are addressed through it so non-tight surfaces
 * work.
 */
inline void blitToFramebuffer(const Image &source, core::u32 *dst, core::u32 dstWidth, core::u32 dstHeight,
                              core::u32 dstPitchBytes, core::i32 dstX, core::i32 dstY) noexcept
{
    if (dst == nullptr || dstPitchBytes == 0u)
        return;

    for (core::u32 sy = 0u; sy < source.height(); ++sy)
    {
        const core::i32 py = dstY + static_cast<core::i32>(sy);
        if (py < 0 || static_cast<core::u32>(py) >= dstHeight)
            continue;
        core::u32 *row = reinterpret_cast<core::u32 *>(reinterpret_cast<core::u8 *>(dst) +
                                                       static_cast<core::usize>(py) * dstPitchBytes);
        for (core::u32 sx = 0u; sx < source.width(); ++sx)
        {
            const core::i32 px = dstX + static_cast<core::i32>(sx);
            if (px < 0 || static_cast<core::u32>(px) >= dstWidth)
                continue;
            row[px] = source.at(static_cast<core::i32>(sx), static_cast<core::i32>(sy));
        }
    }
}

} // namespace lpl::image

#endif // LPL_IMAGE_SURFACE_HPP
