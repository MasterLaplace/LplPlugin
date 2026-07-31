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

    /**
     * @brief Procedural grain: value noise between two colours, tiling seamlessly.
     *
     * No asset, which in ring 0 is not a stylistic choice — there is no filesystem
     * to load one from. Wrapping the lattice at the texture's edge is what makes it
     * tile: sampling neighbours modulo the size means the last column interpolates
     * back towards the first, so a surface covered in copies of it has no seam.
     */
    [[nodiscard]] static Texture makeNoise(core::u32 width, core::u32 height, core::u32 seed, core::u32 colorA,
                                           core::u32 colorB, core::u32 lattice = 8u)
    {
        Texture t(width, height);
        const core::u32 cells = lattice > 0u ? lattice : 1u;

        const auto latticeValue = [seed, cells](core::u32 x, core::u32 z) -> core::u32 {
            core::u32 hash = 0x811C9DC5u ^ seed;
            hash = (hash ^ (x % cells)) * 0x01000193u;
            hash = (hash ^ 0x9E3779B9u) * 0x01000193u;
            hash = (hash ^ (z % cells)) * 0x01000193u;
            return (hash >> 8) & 0xFFu;
        };

        for (core::u32 y = 0u; y < height; ++y)
        {
            for (core::u32 x = 0u; x < width; ++x)
            {
                const core::u32 fx = (x * cells) / width;
                const core::u32 fz = (y * cells) / height;
                const core::u32 stepX = (width / cells) > 0u ? width / cells : 1u;
                const core::u32 stepZ = (height / cells) > 0u ? height / cells : 1u;
                const core::u32 tx = ((x % stepX) * 256u) / stepX;
                const core::u32 tz = ((y % stepZ) * 256u) / stepZ;

                // Smoothstep on the lattice weights, so the grain has no visible
                // diamond grid — linear weights make the lattice itself readable.
                const core::u32 sx = (tx * tx * (768u - 2u * tx)) >> 16;
                const core::u32 sz = (tz * tz * (768u - 2u * tz)) >> 16;

                const core::u32 v00 = latticeValue(fx, fz);
                const core::u32 v10 = latticeValue(fx + 1u, fz);
                const core::u32 v01 = latticeValue(fx, fz + 1u);
                const core::u32 v11 = latticeValue(fx + 1u, fz + 1u);
                const core::u32 top = v00 * (256u - sx) + v10 * sx;
                const core::u32 bottom = v01 * (256u - sx) + v11 * sx;
                const core::u32 value = (top * (256u - sz) + bottom * sz) >> 16;

                core::u32 texel = 0u;
                for (core::u32 shift = 0u; shift <= 16u; shift += 8u)
                {
                    const core::u32 a = channel(colorA, shift);
                    const core::u32 b = channel(colorB, shift);
                    texel |= (((a * (255u - value) + b * value) / 255u) & 0xFFu) << shift;
                }
                t._texels[static_cast<core::usize>(y) * width + x] = texel;
            }
        }
        return t;
    }

    /** @brief Box-filtered half-size copy: one level of a mip chain. */
    [[nodiscard]] Texture halved() const
    {
        const core::u32 w = _width > 1u ? _width / 2u : 1u;
        const core::u32 h = _height > 1u ? _height / 2u : 1u;
        Texture out(w, h);
        for (core::u32 y = 0u; y < h; ++y)
            for (core::u32 x = 0u; x < w; ++x)
            {
                const core::u32 c00 = texel(x * 2u, y * 2u);
                const core::u32 c10 = texel(x * 2u + 1u, y * 2u);
                const core::u32 c01 = texel(x * 2u, y * 2u + 1u);
                const core::u32 c11 = texel(x * 2u + 1u, y * 2u + 1u);
                core::u32 average = 0u;
                for (core::u32 shift = 0u; shift <= 16u; shift += 8u)
                {
                    const core::u32 sum =
                        channel(c00, shift) + channel(c10, shift) + channel(c01, shift) + channel(c11, shift);
                    average |= ((sum / 4u) & 0xFFu) << shift;
                }
                out._texels[static_cast<core::usize>(y) * w + x] = average;
            }
        return out;
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
