/**
 * @file Box.hpp
 * @brief The six faces of an axis-aligned box, once.
 *
 * A box is the least interesting shape there is, which is exactly why its six faces get
 * written out by hand wherever one is needed — and why two copies then disagree about
 * winding, so one of them culls its own front faces and nobody can see why.
 *
 * Only the GEOMETRY is shared here. How a face is LIT is not the same question in two
 * places and must not be forced into one: a diagnostic screenshot wants a fixed key light
 * so a box is legible at any orientation, and a building in a world wants the sun that
 * lights everything else. So @ref forEachBoxFace hands out faces and their normals and
 * says nothing about colour, and @ref drawBox is the sun-lit consumer.
 *
 * The face ORDER is part of the contract. A caller keyed to it — one carrying a table of
 * per-face brightnesses, say — depends on top, bottom, +Z, -Z, +X, -X, in that order.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_BOX_HPP
#    define LPL_RENDER_BOX_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::render {

/**
 * @brief Hands each of a box's six faces to @p emit, with its outward normal.
 *
 * @param x0 Low corner.
 * @param y0 Low corner.
 * @param z0 Low corner.
 * @param x1 High corner.
 * @param y1 High corner.
 * @param z1 High corner.
 * @param emit `emit(const core::f32 quad[12], core::f32 nx, core::f32 ny, core::f32 nz)`.
 */
template <typename Emit>
void forEachBoxFace(core::f32 x0, core::f32 y0, core::f32 z0, core::f32 x1, core::f32 y1, core::f32 z1, Emit &&emit)
{
    const core::f32 faces[6][12] = {
        {x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1}, // top
        {x0, y0, z1, x1, y0, z1, x1, y0, z0, x0, y0, z0}, // bottom
        {x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1}, // +Z
        {x1, y0, z0, x0, y0, z0, x0, y1, z0, x1, y1, z0}, // -Z
        {x1, y0, z1, x1, y0, z0, x1, y1, z0, x1, y1, z1}, // +X
        {x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0}, // -X
    };
    const core::f32 normals[6][3] = {
        {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f},
    };
    for (core::u32 face = 0u; face < 6u; ++face)
        emit(faces[face], normals[face][0], normals[face][1], normals[face][2]);
}

/**
 * @brief Draws an axis-aligned box lit by a directional light.
 *
 * Flat per face, which is what a box wants: an axis-aligned face has one normal, so
 * interpolating across it would compute the same number six hundred times.
 *
 * @param rt        Target.
 * @param mvp       View-projection.
 * @param x0        Low corner.
 * @param y0        Low corner.
 * @param z0        Low corner.
 * @param x1        High corner.
 * @param y1        High corner.
 * @param z1        High corner.
 * @param colour    Albedo.
 * @param sun       Direction TOWARDS the light.
 * @param ambient   What a face turned away from it still receives.
 * @return Triangles submitted.
 */
inline core::u32 drawBox(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, core::f32 x0, core::f32 y0,
                         core::f32 z0, core::f32 x1, core::f32 y1, core::f32 z1, core::u32 colour,
                         const math::Vec3<core::f32> &sun, core::f32 ambient)
{
    core::u32 triangles = 0u;
    forEachBoxFace(x0, y0, z0, x1, y1, z1,
                   [&](const core::f32 *quad, core::f32 nx, core::f32 ny, core::f32 nz) {
                       const core::f32 lambert = nx * sun.x + ny * sun.y + nz * sun.z;
                       const core::f32 lit = ambient + (1.0f - ambient) * (lambert > 0.0f ? lambert : 0.0f);
                       triangles += fillPolygonClipped(rt, mvp, quad, 4u, modulate(colour, lit));
                   });
    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_BOX_HPP
