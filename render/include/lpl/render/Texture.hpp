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

    void setTexel(core::u32 x, core::u32 y, core::u32 rgb);

    [[nodiscard]] core::u32 texel(core::u32 x, core::u32 y) const
    {
        return _texels[static_cast<core::usize>(y % _height) * _width + (x % _width)];
    }

    /**
     * @brief Nearest sample at Q16.16 (u, v); wraps.
     */
    [[nodiscard]] core::u32 sampleNearest(core::u32 uQ16, core::u32 vQ16) const noexcept;

    /**
     * @brief Bilinear sample at Q16.16 (u, v); Q16-weighted, wraps.
     *
     * u times the width, and v times the height, are Q16 positions in texel space.
     */
    [[nodiscard]] core::u32 sampleBilinear(core::u32 uQ16, core::u32 vQ16) const noexcept;

    /**
     * @brief Procedural checkerboard of two colors, `cells` tiles per axis.
     */
    [[nodiscard]] static Texture makeChecker(core::u32 width, core::u32 height, core::u32 colorA, core::u32 colorB,
                                             core::u32 cells);

    /**
     * @brief Procedural grain: value noise between two colours, tiling seamlessly.
     *
     * No asset, which in ring 0 is not a stylistic choice — there is no filesystem
     * to load one from. Wrapping the lattice at the texture's edge is what makes it
     * tile: sampling neighbours modulo the size means the last column interpolates
     * back towards the first, so a surface covered in copies of it has no seam.
     *
     * The lattice weights go through a smoothstep, so the grain has no visible
     * diamond grid — linear weights make the lattice itself readable.
     */
    [[nodiscard]] static Texture makeNoise(core::u32 width, core::u32 height, core::u32 seed, core::u32 colorA,
                                           core::u32 colorB, core::u32 lattice = 8u);

    /**
     * @brief Box-filtered half-size copy: one level of a mip chain.
     */
    [[nodiscard]] Texture halved() const;

private:
    [[nodiscard]] static core::u32 channel(core::u32 c, core::u32 shift) noexcept { return (c >> shift) & 0xFFu; }

    /**
     * @brief Blends four texels, weighted by the Q16 fractions fx and fy.
     *
     * Each channel is blended along x into a Q16 value per row; the two rows
     * are then combined vertically and shifted back to 8 bits.
     */
    [[nodiscard]] static core::u32 blend4(core::u32 c00, core::u32 c10, core::u32 c01, core::u32 c11, core::u32 fx,
                                          core::u32 fy) noexcept;

    pmr::vector<core::u32> _texels;
    core::u32 _width{0u};
    core::u32 _height{0u};
};

} // namespace lpl::render

#    include "Texture.inl"

#endif // LPL_RENDER_TEXTURE_HPP
