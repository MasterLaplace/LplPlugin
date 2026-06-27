/**
 * @file Painter.hpp
 * @brief Deterministic integer 2D drawing primitives over an Image.
 *
 * Lines (Bresenham), rectangles (outline + fill), circles (midpoint + fill) and
 * image blits, all in integer arithmetic so the output is bit-identical across
 * the Linux oracle and the freestanding kernel build. Source-over alpha blending
 * uses 0..255 integer weights. The Painter holds no state beyond the target
 * surface reference.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_IMAGE_PAINTER_HPP
#    define LPL_IMAGE_PAINTER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/image/Image.hpp>

namespace lpl::image {

/**
 * @class Painter
 * @brief Stateless drawing helper bound to a target Image.
 */
class Painter {
public:
    explicit Painter(Image &target) noexcept : _target(target) {}

    /** @brief Composite a single pixel with source-over alpha blending. */
    void blendPixel(core::i32 x, core::i32 y, Rgba color) noexcept;

    /** @brief Fill an axis-aligned rectangle (clipped, alpha-blended). */
    void fillRect(core::i32 x, core::i32 y, core::i32 w, core::i32 h, Rgba color) noexcept;

    /** @brief Draw a one-pixel rectangle outline (alpha-blended). */
    void drawRect(core::i32 x, core::i32 y, core::i32 w, core::i32 h, Rgba color) noexcept;

    /** @brief Draw a line from (x0,y0) to (x1,y1) (Bresenham, alpha-blended). */
    void drawLine(core::i32 x0, core::i32 y0, core::i32 x1, core::i32 y1, Rgba color) noexcept;

    /** @brief Draw a circle outline of @p radius centred at (cx,cy). */
    void drawCircle(core::i32 cx, core::i32 cy, core::i32 radius, Rgba color) noexcept;

    /** @brief Fill a disc of @p radius centred at (cx,cy). */
    void fillCircle(core::i32 cx, core::i32 cy, core::i32 radius, Rgba color) noexcept;

    /** @brief Blit @p source at (dstX,dstY) with source-over alpha blending. */
    void blit(const Image &source, core::i32 dstX, core::i32 dstY) noexcept;

private:
    Image &_target;
};

/** @brief Fold every pixel into a 32-bit FNV-1a signature (deterministic). */
[[nodiscard]] inline core::u32 foldSignature(const Image &image) noexcept
{
    core::u32 acc = 2166136261u;
    for (core::u32 y = 0u; y < image.height(); ++y)
        for (core::u32 x = 0u; x < image.width(); ++x)
            acc = (acc ^ image.at(static_cast<core::i32>(x), static_cast<core::i32>(y))) * 16777619u;
    return acc;
}

/**
 * @brief Paint a fixed reference scene into @p image.
 *
 * Shared by the host parity test and the kernel smoke so both fold the exact
 * same drawn pixels — any cross-target divergence in the integer rasterisers
 * shows up as a different foldSignature().
 */
inline void paintParityScene(Image &image) noexcept
{
    Painter painter(image);
    image.fill(packRgba(10, 20, 30));
    painter.fillRect(2, 2, 10, 6, packRgba(200, 40, 40));
    painter.drawLine(0, 0, static_cast<core::i32>(image.width()) - 1, static_cast<core::i32>(image.height()) - 1,
                     packRgba(0, 200, 0));
    painter.drawCircle(8, 8, 5, packRgba(0, 0, 220));
    painter.fillCircle(20, 20, 4, packRgba(255, 255, 0, 128));
}

} // namespace lpl::image

#endif // LPL_IMAGE_PAINTER_HPP
