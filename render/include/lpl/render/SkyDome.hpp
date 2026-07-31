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
 * @param tanHalfFov Tangent of half the vertical field of view.
 */
inline void drawSky(const RenderTarget &rt, const CameraBasis &basis, const SunState &sun, const SkyParams &params,
                    core::f32 tanHalfFov = 0.5773502692f, core::u32 block = 1u) noexcept
{
    const core::f32 aspect = static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height);
    const core::u32 step = block == 0u ? 1u : block;

    for (core::u32 y = 0u; y < rt.height; y += step)
    {
        const core::f32 ndcY = 1.0f - 2.0f * (static_cast<core::f32>(y) + 0.5f) / static_cast<core::f32>(rt.height);
        for (core::u32 x = 0u; x < rt.width; x += step)
        {
            const core::f32 ndcX = 2.0f * (static_cast<core::f32>(x) + 0.5f) / static_cast<core::f32>(rt.width) - 1.0f;
            const core::f32 h = ndcX * tanHalfFov * aspect;
            const core::f32 v = ndcY * tanHalfFov;

            core::f32 dx = basis.forward.x + basis.right.x * h + basis.up.x * v;
            core::f32 dy = basis.forward.y + basis.right.y * h + basis.up.y * v;
            core::f32 dz = basis.forward.z + basis.right.z * h + basis.up.z * v;
            const core::f32 inverse = inverseSqrtNewton(dx * dx + dy * dy + dz * dz);
            dx *= inverse;
            dy *= inverse;
            dz *= inverse;

            const core::u32 colour = skyColour(dx, dy, dz, sun, params);

            // One evaluation per block, written to every pixel of it. The sky is
            // the smoothest thing on screen — over two pixels it changes by less
            // than one step of an 8-bit channel — so a block of 2 or 3 is free in
            // appearance and divides the cost by its area.
            const core::u32 maxY = (y + step) < rt.height ? (y + step) : rt.height;
            const core::u32 maxX = (x + step) < rt.width ? (x + step) : rt.width;
            for (core::u32 by = y; by < maxY; ++by)
                for (core::u32 bx = x; bx < maxX; ++bx)
                {
                    rt.color[by * rt.width + bx] = colour;
                    rt.depth[by * rt.width + bx] = 1.0e30f;
                }
        }
    }
}

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

#endif // LPL_RENDER_SKY_DOME_HPP
