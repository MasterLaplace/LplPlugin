/**
 * @file Texture.hpp
 * @brief Integer-deterministic 2D texture with nearest + bilinear sampling.
 *
 * Texels are packed 0x00RRGGBB. UV coordinates are Q16.16 fixed point and
 * sampling is pure integer arithmetic (Q16-weighted bilinear), so results are
 * bit-identical across the Linux oracle and the i686 kernel with no float and
 * no libm. Wrap addressing uses modulo (any dimensions).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_TEXTURE_HPP
#    define LPL_RENDER_TEXTURE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

class Texture {
public:
    Texture() = default;
    Texture(core::u32 width, core::u32 height) : _width(width), _height(height)
    {
        _texels.resize(static_cast<core::usize>(width) * height, 0u);
    }

    [[nodiscard]] core::u32 width() const noexcept { return _width; }
    [[nodiscard]] core::u32 height() const noexcept { return _height; }

    void setTexel(core::u32 x, core::u32 y, core::u32 rgb)
    {
        if (x < _width && y < _height)
            _texels[static_cast<core::usize>(y) * _width + x] = rgb;
    }

    [[nodiscard]] core::u32 texel(core::u32 x, core::u32 y) const
    {
        return _texels[static_cast<core::usize>(y % _height) * _width + (x % _width)];
    }

    /** @brief Nearest sample at Q16.16 (u, v); wraps. */
    [[nodiscard]] core::u32 sampleNearest(core::u32 uQ16, core::u32 vQ16) const noexcept
    {
        const core::u32 x = (static_cast<core::u64>(uQ16) * _width >> 16) % _width;
        const core::u32 y = (static_cast<core::u64>(vQ16) * _height >> 16) % _height;
        return _texels[static_cast<core::usize>(y) * _width + x];
    }

    /** @brief Bilinear sample at Q16.16 (u, v); Q16-weighted, wraps. */
    [[nodiscard]] core::u32 sampleBilinear(core::u32 uQ16, core::u32 vQ16) const noexcept
    {
        const core::u64 fu = static_cast<core::u64>(uQ16) * _width; // Q16 in texel space
        const core::u64 fv = static_cast<core::u64>(vQ16) * _height;
        const core::u32 x0 = static_cast<core::u32>(fu >> 16) % _width;
        const core::u32 y0 = static_cast<core::u32>(fv >> 16) % _height;
        const core::u32 x1 = (x0 + 1u) % _width;
        const core::u32 y1 = (y0 + 1u) % _height;
        const core::u32 fx = static_cast<core::u32>(fu & 0xFFFFu);
        const core::u32 fy = static_cast<core::u32>(fv & 0xFFFFu);

        const core::u32 c00 = _texels[static_cast<core::usize>(y0) * _width + x0];
        const core::u32 c10 = _texels[static_cast<core::usize>(y0) * _width + x1];
        const core::u32 c01 = _texels[static_cast<core::usize>(y1) * _width + x0];
        const core::u32 c11 = _texels[static_cast<core::usize>(y1) * _width + x1];

        return blend4(c00, c10, c01, c11, fx, fy);
    }

    /** @brief Procedural checkerboard of two colors, `cells` tiles per axis. */
    [[nodiscard]] static Texture makeChecker(core::u32 width, core::u32 height, core::u32 colorA, core::u32 colorB,
                                             core::u32 cells)
    {
        Texture t(width, height);
        const core::u32 cw = (width / cells) > 0u ? (width / cells) : 1u;
        const core::u32 ch = (height / cells) > 0u ? (height / cells) : 1u;
        for (core::u32 y = 0; y < height; ++y)
            for (core::u32 x = 0; x < width; ++x)
                t._texels[static_cast<core::usize>(y) * width + x] = (((x / cw) + (y / ch)) & 1u) ? colorB : colorA;
        return t;
    }

private:
    [[nodiscard]] static core::u32 channel(core::u32 c, core::u32 shift) noexcept { return (c >> shift) & 0xFFu; }

    [[nodiscard]] static core::u32 blend4(core::u32 c00, core::u32 c10, core::u32 c01, core::u32 c11, core::u32 fx,
                                          core::u32 fy) noexcept
    {
        core::u32 out = 0u;
        for (core::u32 shift = 0u; shift <= 16u; shift += 8u)
        {
            const core::u32 top = channel(c00, shift) * (65536u - fx) + channel(c10, shift) * fx;
            const core::u32 bot = channel(c01, shift) * (65536u - fx) + channel(c11, shift) * fx;
            // top/bot are Q16; combine vertically and shift back to 8-bit.
            const core::u64 v = (static_cast<core::u64>(top) * (65536u - fy) + static_cast<core::u64>(bot) * fy) >> 32;
            out |= (static_cast<core::u32>(v) & 0xFFu) << shift;
        }
        return out;
    }

    pmr::vector<core::u32> _texels;
    core::u32 _width{0u};
    core::u32 _height{0u};
};

} // namespace lpl::render

#endif // LPL_RENDER_TEXTURE_HPP
