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
    core::u32 shaded_fragments{0u}; /**< Representative fragments actually shaded. */
    core::u32 full_fragments{0u};   /**< width*height (cost of full-rate shading). */
    core::u32 image_signature{0u};  /**< FNV-1a fold of the replicated image. */
};

namespace detail {

/**
 * @brief Shade rate (cell edge in pixels) for a tile at Chebyshev ring r.
 *
 * 1x1 in the fovea, doubling each ring out to a 4x4 coarse periphery.
 */
[[nodiscard]] inline core::u32 shadeRateForRing(core::u32 ring) noexcept;

} // namespace detail

/**
 * @brief Shades a procedural gradient with foveation centered at (gazeX,gazeY).
 *
 * Tiles are 8x8 screen blocks; their Chebyshev ring distance from the gaze tile
 * selects a 1/2/4-pixel shade rate. Within a tile, only the top-left fragment of
 * each rate-sized cell is shaded (a deterministic float gradient, standing in
 * for a real fragment shader) and copied across the cell.
 *
 * @param color  Destination RGB buffer (width*height).
 * @param width  Image width.
 * @param height Image height.
 * @param gazeX  Gaze point X in pixels.
 * @param gazeY  Gaze point Y in pixels.
 */
[[nodiscard]] inline FoveatedResult foveatedShade(core::u32 *color, core::u32 width, core::u32 height, core::u32 gazeX,
                                                  core::u32 gazeY);

} // namespace lpl::render

#    include "Foveated.inl"

#endif // LPL_RENDER_FOVEATED_HPP
