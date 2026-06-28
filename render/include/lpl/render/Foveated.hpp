/**
 * @file Foveated.hpp
 * @brief Software foveated (variable-rate) rasterization helper.
 *
 * There is no GPU variable-rate-shading on a software LFB, so foveation is a CPU
 * optimization: the screen is divided into shading tiles whose coarseness grows
 * with distance from a gaze point. Each tile shades ONE representative fragment
 * and replicates it across the tile's pixels — fewer shader evaluations toward
 * the periphery. The shade rate map and the resulting image both fold
 * bit-identically across the Linux oracle and the i686 kernel (integer tile
 * math + the deterministic float shade). The folded image + the shaded-fragment
 * count are the cross-target signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_FOVEATED_HPP
#    define LPL_RENDER_FOVEATED_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/RenderParity.hpp>

namespace lpl::render {

/** @brief Result of a foveated shading pass. */
struct FoveatedResult {
    core::u32 width{0u};
    core::u32 height{0u};
    core::u32 shaded_fragments{0u}; ///< Representative fragments actually shaded.
    core::u32 full_fragments{0u};   ///< width*height (cost of full-rate shading).
    core::u32 image_signature{0u};  ///< FNV-1a fold of the replicated image.
};

namespace detail {

/** @brief Shade rate (tile edge in pixels) for a tile at Chebyshev ring r. */
[[nodiscard]] inline core::u32 shadeRateForRing(core::u32 ring) noexcept
{
    // 1x1 in the fovea, doubling each ring out to a 4x4 coarse periphery.
    if (ring == 0u)
        return 1u;
    if (ring == 1u)
        return 2u;
    return 4u;
}

} // namespace detail

/**
 * @brief Shades a procedural gradient with foveation centered at (gazeX,gazeY).
 *
 * Tiles are 8x8 screen blocks; their Chebyshev ring distance from the gaze tile
 * selects a 1/2/4-pixel shade rate. Within a tile, only the top-left fragment of
 * each rate-sized cell is shaded (a deterministic float gradient) and copied
 * across the cell.
 *
 * @param color  Destination RGB buffer (width*height).
 * @param width  Image width.
 * @param height Image height.
 * @param gazeX  Gaze point X in pixels.
 * @param gazeY  Gaze point Y in pixels.
 */
[[nodiscard]] inline FoveatedResult foveatedShade(core::u32 *color, core::u32 width, core::u32 height, core::u32 gazeX,
                                                  core::u32 gazeY)
{
    FoveatedResult out{};
    out.width = width;
    out.height = height;
    out.full_fragments = width * height;
    if (color == nullptr || width == 0u || height == 0u)
        return out;

    constexpr core::u32 kTile = 8u;
    const core::u32 gazeTileX = gazeX / kTile;
    const core::u32 gazeTileY = gazeY / kTile;

    const auto shadePixel = [&](core::u32 x, core::u32 y) -> core::u32 {
        // Deterministic float gradient (stands in for a real fragment shader).
        const core::f32 fx = static_cast<core::f32>(x) / static_cast<core::f32>(width);
        const core::f32 fy = static_cast<core::f32>(y) / static_cast<core::f32>(height);
        const core::u32 r = static_cast<core::u32>(detail::saturate(fx) * 255.0f + 0.5f);
        const core::u32 g = static_cast<core::u32>(detail::saturate(fy) * 255.0f + 0.5f);
        const core::u32 b = static_cast<core::u32>(detail::saturate(1.0f - fx * fy) * 255.0f + 0.5f);
        return (r << 16) | (g << 8) | b;
    };

    for (core::u32 ty = 0; ty * kTile < height; ++ty)
        for (core::u32 tx = 0; tx * kTile < width; ++tx)
        {
            const core::u32 dx = tx > gazeTileX ? tx - gazeTileX : gazeTileX - tx;
            const core::u32 dy = ty > gazeTileY ? ty - gazeTileY : gazeTileY - ty;
            const core::u32 ring = dx > dy ? dx : dy;
            const core::u32 rate = detail::shadeRateForRing(ring);

            const core::u32 x0 = tx * kTile;
            const core::u32 y0 = ty * kTile;
            for (core::u32 cy = 0; cy < kTile && (y0 + cy) < height; cy += rate)
                for (core::u32 cx = 0; cx < kTile && (x0 + cx) < width; cx += rate)
                {
                    const core::u32 rgb = shadePixel(x0 + cx, y0 + cy);
                    ++out.shaded_fragments;
                    for (core::u32 fy = 0; fy < rate && (y0 + cy + fy) < height; ++fy)
                        for (core::u32 fx = 0; fx < rate && (x0 + cx + fx) < width; ++fx)
                            color[static_cast<core::usize>(y0 + cy + fy) * width + (x0 + cx + fx)] = rgb;
                }
        }

    core::u32 hash = 0x811C9DC5u;
    for (core::u32 i = 0; i < width * height; ++i)
        hash = detail::fnv1aStep(hash, color[i]);
    out.image_signature = hash;
    return out;
}

} // namespace lpl::render

#endif // LPL_RENDER_FOVEATED_HPP
