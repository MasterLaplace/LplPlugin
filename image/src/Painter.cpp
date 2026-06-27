/**
 * @file Painter.cpp
 * @brief Integer 2D drawing primitives (Bresenham line, midpoint circle, blit).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */
#include <lpl/image/Painter.hpp>

namespace lpl::image {

namespace {

/// Source-over composite of @p src over @p dst using 0..255 integer alpha.
[[nodiscard]] Rgba composite(Rgba dst, Rgba src) noexcept
{
    const core::u32 sa = alphaOf(src);
    if (sa == 0u)
        return dst;
    if (sa == 255u)
        return src;

    const core::u32 ia = 255u - sa;
    const auto chan = [sa, ia](core::u32 s, core::u32 d) -> core::u8 {
        return static_cast<core::u8>((s * sa + d * ia + 127u) / 255u);
    };
    const core::u8 r = chan(redOf(src), redOf(dst));
    const core::u8 g = chan(greenOf(src), greenOf(dst));
    const core::u8 b = chan(blueOf(src), blueOf(dst));
    // Resulting coverage: sa + da*(1-sa).
    const core::u32 da = alphaOf(dst);
    const core::u8 a = static_cast<core::u8>(sa + (da * ia + 127u) / 255u);
    return packRgba(r, g, b, a);
}

} // namespace

void Painter::blendPixel(core::i32 x, core::i32 y, Rgba color) noexcept
{
    if (x < 0 || y < 0 || static_cast<core::u32>(x) >= _target.width() ||
        static_cast<core::u32>(y) >= _target.height())
        return;
    _target.set(x, y, composite(_target.at(x, y), color));
}

void Painter::fillRect(core::i32 x, core::i32 y, core::i32 w, core::i32 h, Rgba color) noexcept
{
    if (w <= 0 || h <= 0)
        return;
    // Clip to the surface up front so the inner loop is branch-light.
    const core::i32 x0 = (x < 0) ? 0 : x;
    const core::i32 y0 = (y < 0) ? 0 : y;
    const core::i32 x1 = (x + w > static_cast<core::i32>(_target.width())) ? static_cast<core::i32>(_target.width())
                                                                           : x + w;
    const core::i32 y1 = (y + h > static_cast<core::i32>(_target.height())) ? static_cast<core::i32>(_target.height())
                                                                            : y + h;
    for (core::i32 py = y0; py < y1; ++py)
        for (core::i32 px = x0; px < x1; ++px)
            blendPixel(px, py, color);
}

void Painter::drawRect(core::i32 x, core::i32 y, core::i32 w, core::i32 h, Rgba color) noexcept
{
    if (w <= 0 || h <= 0)
        return;
    drawLine(x, y, x + w - 1, y, color);                     // top
    drawLine(x, y + h - 1, x + w - 1, y + h - 1, color);     // bottom
    drawLine(x, y, x, y + h - 1, color);                     // left
    drawLine(x + w - 1, y, x + w - 1, y + h - 1, color);     // right
}

void Painter::drawLine(core::i32 x0, core::i32 y0, core::i32 x1, core::i32 y1, Rgba color) noexcept
{
    core::i32 dx = (x1 > x0) ? (x1 - x0) : (x0 - x1);
    core::i32 dy = (y1 > y0) ? (y1 - y0) : (y0 - y1);
    const core::i32 sx = (x0 < x1) ? 1 : -1;
    const core::i32 sy = (y0 < y1) ? 1 : -1;
    dy = -dy;
    core::i32 err = dx + dy;

    for (;;)
    {
        blendPixel(x0, y0, color);
        if (x0 == x1 && y0 == y1)
            break;
        const core::i32 e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void Painter::drawCircle(core::i32 cx, core::i32 cy, core::i32 radius, Rgba color) noexcept
{
    if (radius < 0)
        return;
    core::i32 x = radius;
    core::i32 y = 0;
    core::i32 err = 1 - radius; // midpoint decision variable

    while (x >= y)
    {
        blendPixel(cx + x, cy + y, color);
        blendPixel(cx + y, cy + x, color);
        blendPixel(cx - y, cy + x, color);
        blendPixel(cx - x, cy + y, color);
        blendPixel(cx - x, cy - y, color);
        blendPixel(cx - y, cy - x, color);
        blendPixel(cx + y, cy - x, color);
        blendPixel(cx + x, cy - y, color);
        ++y;
        if (err < 0)
            err += 2 * y + 1;
        else
        {
            --x;
            err += 2 * (y - x) + 1;
        }
    }
}

void Painter::fillCircle(core::i32 cx, core::i32 cy, core::i32 radius, Rgba color) noexcept
{
    if (radius < 0)
        return;
    const core::i32 r2 = radius * radius;
    for (core::i32 dy = -radius; dy <= radius; ++dy)
    {
        // Half-width of the disc at this row: floor(sqrt(r^2 - dy^2)) via a
        // bounded integer scan (no libm), kept deterministic.
        const core::i32 rem = r2 - dy * dy;
        core::i32 dx = 0;
        while ((dx + 1) * (dx + 1) <= rem)
            ++dx;
        drawLine(cx - dx, cy + dy, cx + dx, cy + dy, color);
    }
}

void Painter::blit(const Image &source, core::i32 dstX, core::i32 dstY) noexcept
{
    for (core::u32 sy = 0u; sy < source.height(); ++sy)
        for (core::u32 sx = 0u; sx < source.width(); ++sx)
            blendPixel(dstX + static_cast<core::i32>(sx), dstY + static_cast<core::i32>(sy),
                       source.at(static_cast<core::i32>(sx), static_cast<core::i32>(sy)));
}

} // namespace lpl::image
