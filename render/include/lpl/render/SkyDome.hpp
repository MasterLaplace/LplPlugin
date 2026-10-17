/**
 * @file SkyDome.hpp
 * @brief Paints a whole render target by looking down every pixel's own ray.
 *
 * Sits above @c Sky.hpp (which answers "what colour is this direction?") and
 * @c SoftwareRasterizer.hpp (which owns the target), so neither has to know
 * about the other.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_SKY_DOME_HPP
#    define LPL_RENDER_SKY_DOME_HPP

#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/Sky.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::render {

/**
 * @brief Fills the target with the sky, one evaluation of direction per pixel.
 *
 * Replaces a flat clear at the same cost class: no texture, no cube map, nothing
 * to load — which on the kernel side is not an optimisation but the only option
 * available, since there is no filesystem to load a sky from.
 *
 * Rays are built from the camera basis directly rather than by inverting the
 * projection: the basis already exists for the view matrix, and the inverse of a
 * matrix nobody needs is work nobody should do. The depth buffer is set to far,
 * so everything drawn afterwards wins against the sky by construction.
 *
 * One evaluation per block, written to every pixel of it. The sky is the
 * smoothest thing on screen — over two pixels it changes by less than one step
 * of an 8-bit channel — so a block of 2 or 3 is free in appearance and divides
 * the cost by its area.
 *
 * @param tanHalfFov Tangent of half the vertical field of view.
 */
inline void drawSky(const RenderTarget &rt, const CameraBasis &basis, const SunState &sun, const SkyParams &params,
                    core::f32 tanHalfFov = 0.5773502692f, core::u32 block = 1u) noexcept;

/**
 * @brief The haze distant geometry fades into: the sky along the horizon.
 *
 * Taken in the view direction, so a far ridge dissolves into the colour the sky
 * already has behind it rather than into a fog colour chosen by hand.
 */
[[nodiscard]] inline core::u32 hazeTint(const CameraBasis &basis, const SunState &sun, const SkyParams &params) noexcept
{
    return skyColour(basis.forward.x, 0.02f, basis.forward.z, sun, params);
}

} // namespace lpl::render

#    include "SkyDome.inl"

#endif // LPL_RENDER_SKY_DOME_HPP
