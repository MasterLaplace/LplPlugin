/**
 * @file SoftwareRasterizer.hpp
 * @brief Portable, depth-buffered software 3D triangle rasterizer.
 *
 * Renders an indexed triangle mesh into a caller-provided linear color buffer
 * with a parallel float depth buffer (z-test). Geometry/model rotation is
 * authored in Fixed32 (CORDIC) = authoritative; projection, the perspective
 * divide and the per-pixel barycentric fill run in float (SSE,
 * -ffp-contract=off) which is bit-identical host vs kernel. The same code path
 * drives the Linux oracle parity test and the in-kernel present smoke.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_SOFTWARERASTERIZER_HPP
#    define LPL_RENDER_SOFTWARERASTERIZER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/render/Texture.hpp>

namespace lpl::render {

/** @brief Caller-owned render target: packed 0x00RRGGBB color + float depth. */
struct RenderTarget {
    core::u32 *color{nullptr};   ///< width*height packed pixels (row-major, no padding).
    core::f32 *depth{nullptr};   ///< width*height depth, smaller = nearer.
    core::u32 width{0u};
    core::u32 height{0u};
};

namespace detail {

/** @brief One projected vertex: screen x/y (float), NDC depth, in-front flag. */
struct ScreenVertex {
    core::f32 x{0.0f};
    core::f32 y{0.0f};
    core::f32 z{0.0f};
    bool valid{false};
};

[[nodiscard]] inline ScreenVertex projectVertex(const math::Mat4<core::f32> &mvp, core::f32 wx, core::f32 wy,
                                                core::f32 wz, core::u32 width, core::u32 height) noexcept
{
    const core::f32 cx = mvp(0, 0) * wx + mvp(0, 1) * wy + mvp(0, 2) * wz + mvp(0, 3);
    const core::f32 cy = mvp(1, 0) * wx + mvp(1, 1) * wy + mvp(1, 2) * wz + mvp(1, 3);
    const core::f32 cz = mvp(2, 0) * wx + mvp(2, 1) * wy + mvp(2, 2) * wz + mvp(2, 3);
    const core::f32 cw = mvp(3, 0) * wx + mvp(3, 1) * wy + mvp(3, 2) * wz + mvp(3, 3);

    ScreenVertex v{};
    if (cw <= 0.0f)
        return v;
    const core::f32 invW = 1.0f / cw;
    v.x = (cx * invW * 0.5f + 0.5f) * static_cast<core::f32>(width);
    v.y = (0.5f - cy * invW * 0.5f) * static_cast<core::f32>(height);
    v.z = cz * invW;
    v.valid = true;
    return v;
}

[[nodiscard]] inline core::f32 edge(const ScreenVertex &a, const ScreenVertex &b, core::f32 px, core::f32 py) noexcept
{
    return (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x);
}

[[nodiscard]] inline core::i32 clampInt(core::i32 v, core::i32 lo, core::i32 hi) noexcept
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/** @brief Depth-tested barycentric fill of one front-facing triangle. */
inline void fillTriangle(const RenderTarget &rt, const ScreenVertex &v0, const ScreenVertex &v1,
                         const ScreenVertex &v2, core::u32 color) noexcept
{
    if (!v0.valid || !v1.valid || !v2.valid)
        return;

    const core::f32 area = edge(v0, v1, v2.x, v2.y);
    if (area <= 0.0f) // back-facing or degenerate (cull).
        return;
    const core::f32 invArea = 1.0f / area;

    core::f32 minXf = v0.x, maxXf = v0.x, minYf = v0.y, maxYf = v0.y;
    minXf = v1.x < minXf ? v1.x : minXf;
    minXf = v2.x < minXf ? v2.x : minXf;
    maxXf = v1.x > maxXf ? v1.x : maxXf;
    maxXf = v2.x > maxXf ? v2.x : maxXf;
    minYf = v1.y < minYf ? v1.y : minYf;
    minYf = v2.y < minYf ? v2.y : minYf;
    maxYf = v1.y > maxYf ? v1.y : maxYf;
    maxYf = v2.y > maxYf ? v2.y : maxYf;

    const core::i32 minX = clampInt(static_cast<core::i32>(minXf), 0, static_cast<core::i32>(rt.width) - 1);
    const core::i32 maxX = clampInt(static_cast<core::i32>(maxXf), 0, static_cast<core::i32>(rt.width) - 1);
    const core::i32 minY = clampInt(static_cast<core::i32>(minYf), 0, static_cast<core::i32>(rt.height) - 1);
    const core::i32 maxY = clampInt(static_cast<core::i32>(maxYf), 0, static_cast<core::i32>(rt.height) - 1);

    for (core::i32 y = minY; y <= maxY; ++y)
    {
        const core::f32 py = static_cast<core::f32>(y) + 0.5f;
        for (core::i32 x = minX; x <= maxX; ++x)
        {
            const core::f32 px = static_cast<core::f32>(x) + 0.5f;
            const core::f32 w0 = edge(v1, v2, px, py) * invArea;
            const core::f32 w1 = edge(v2, v0, px, py) * invArea;
            const core::f32 w2 = edge(v0, v1, px, py) * invArea;
            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
                continue;

            const core::f32 depth = w0 * v0.z + w1 * v1.z + w2 * v2.z;
            const core::u32 idx = static_cast<core::u32>(y) * rt.width + static_cast<core::u32>(x);
            if (depth < rt.depth[idx])
            {
                rt.depth[idx] = depth;
                rt.color[idx] = color;
            }
        }
    }
}

/** @brief Depth-tested, affine-UV textured fill of one front-facing triangle. */
inline void fillTriangleTextured(const RenderTarget &rt, const ScreenVertex &v0, const ScreenVertex &v1,
                                 const ScreenVertex &v2, const core::f32 *uv0, const core::f32 *uv1,
                                 const core::f32 *uv2, const Texture &tex) noexcept
{
    if (!v0.valid || !v1.valid || !v2.valid)
        return;
    const core::f32 area = edge(v0, v1, v2.x, v2.y);
    if (area <= 0.0f)
        return;
    const core::f32 invArea = 1.0f / area;

    core::f32 minXf = v0.x, maxXf = v0.x, minYf = v0.y, maxYf = v0.y;
    minXf = v1.x < minXf ? v1.x : minXf;
    minXf = v2.x < minXf ? v2.x : minXf;
    maxXf = v1.x > maxXf ? v1.x : maxXf;
    maxXf = v2.x > maxXf ? v2.x : maxXf;
    minYf = v1.y < minYf ? v1.y : minYf;
    minYf = v2.y < minYf ? v2.y : minYf;
    maxYf = v1.y > maxYf ? v1.y : maxYf;
    maxYf = v2.y > maxYf ? v2.y : maxYf;

    const core::i32 minX = clampInt(static_cast<core::i32>(minXf), 0, static_cast<core::i32>(rt.width) - 1);
    const core::i32 maxX = clampInt(static_cast<core::i32>(maxXf), 0, static_cast<core::i32>(rt.width) - 1);
    const core::i32 minY = clampInt(static_cast<core::i32>(minYf), 0, static_cast<core::i32>(rt.height) - 1);
    const core::i32 maxY = clampInt(static_cast<core::i32>(maxYf), 0, static_cast<core::i32>(rt.height) - 1);

    for (core::i32 y = minY; y <= maxY; ++y)
    {
        const core::f32 py = static_cast<core::f32>(y) + 0.5f;
        for (core::i32 x = minX; x <= maxX; ++x)
        {
            const core::f32 px = static_cast<core::f32>(x) + 0.5f;
            const core::f32 w0 = edge(v1, v2, px, py) * invArea;
            const core::f32 w1 = edge(v2, v0, px, py) * invArea;
            const core::f32 w2 = edge(v0, v1, px, py) * invArea;
            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
                continue;

            const core::f32 depth = w0 * v0.z + w1 * v1.z + w2 * v2.z;
            const core::u32 idx = static_cast<core::u32>(y) * rt.width + static_cast<core::u32>(x);
            if (depth >= rt.depth[idx])
                continue;

            // Affine UV interpolation, clamped to [0,1), to Q16 for the sampler.
            core::f32 u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0];
            core::f32 v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1];
            u = u < 0.0f ? 0.0f : (u > 0.999985f ? 0.999985f : u);
            v = v < 0.0f ? 0.0f : (v > 0.999985f ? 0.999985f : v);
            const core::u32 uQ16 = static_cast<core::u32>(u * 65536.0f);
            const core::u32 vQ16 = static_cast<core::u32>(v * 65536.0f);

            rt.depth[idx] = depth;
            rt.color[idx] = tex.sampleBilinear(uQ16, vQ16);
        }
    }
}

} // namespace detail

/** @brief Clears the render target to a background color and far depth. */
inline void clearTarget(const RenderTarget &rt, core::u32 background) noexcept
{
    const core::u32 count = rt.width * rt.height;
    for (core::u32 i = 0; i < count; ++i)
    {
        rt.color[i] = background;
        rt.depth[i] = 1.0e30f;
    }
}

/**
 * @brief Renders the canonical parity cube (12 depth-tested triangles, per-face
 *        flat colors) through a rotate+perspective camera into the target.
 *
 * @param rt            Render target (color + depth, width*height each).
 * @param rotationAngle Model Y-axis rotation, Fixed32 radians (authoritative).
 */
inline void renderCube(const RenderTarget &rt, math::Fixed32 rotationAngle) noexcept
{
    using F = math::Fixed32;
    using Vec3f = math::Vec3<core::f32>;

    clearTarget(rt, 0x00102030u);

    static const core::f32 corners[8][3] = {
        {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, 1.0f},  {1.0f, -1.0f, 1.0f},  {1.0f, 1.0f, 1.0f},  {-1.0f, 1.0f, 1.0f},
    };
    // 12 triangles (CCW front faces), 6 faces x 2.
    static const core::u32 indices[36] = {
        0, 1, 2, 0, 2, 3, // -Z
        5, 4, 7, 5, 7, 6, // +Z
        4, 0, 3, 4, 3, 7, // -X
        1, 5, 6, 1, 6, 2, // +X
        4, 5, 1, 4, 1, 0, // -Y
        3, 2, 6, 3, 6, 7, // +Y
    };
    static const core::u32 faceColors[6] = {
        0x00C04040u, 0x0040C040u, 0x004040C0u, 0x00C0C040u, 0x00C040C0u, 0x0040C0C0u,
    };

    F s{F::fromInt(0)};
    F c{F::fromInt(0)};
    math::Cordic::sincos(rotationAngle, s, c);
    const core::f32 cf = c.toFloat();
    const core::f32 sf = s.toFloat();

    const auto view = math::Mat4<core::f32>::lookAt(Vec3f(0.0f, 0.0f, 5.0f), Vec3f(0.0f, 0.0f, 0.0f),
                                                    Vec3f(0.0f, 1.0f, 0.0f));
    const core::f32 aspect = static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height);
    const auto proj = perspectiveFov(F::fromFloat(1.04719755f), aspect, 0.1f, 100.0f);
    const auto mvp = proj * view;

    detail::ScreenVertex sv[8];
    for (core::u32 i = 0; i < 8u; ++i)
    {
        const core::f32 rx = cf * corners[i][0] + sf * corners[i][2];
        const core::f32 rz = -sf * corners[i][0] + cf * corners[i][2];
        sv[i] = detail::projectVertex(mvp, rx, corners[i][1], rz, rt.width, rt.height);
    }

    for (core::u32 t = 0; t < 12u; ++t)
        detail::fillTriangle(rt, sv[indices[t * 3 + 0]], sv[indices[t * 3 + 1]], sv[indices[t * 3 + 2]],
                             faceColors[t / 2u]);
}

/**
 * @brief Renders the canonical cube with a texture sampled per face (each face
 *        mapped to the full [0,1] UV square, affine-interpolated, bilinear).
 */
inline void renderTexturedCube(const RenderTarget &rt, math::Fixed32 rotationAngle, const Texture &tex) noexcept
{
    using F = math::Fixed32;
    using Vec3f = math::Vec3<core::f32>;

    clearTarget(rt, 0x00102030u);

    static const core::f32 corners[8][3] = {
        {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, 1.0f},  {1.0f, -1.0f, 1.0f},  {1.0f, 1.0f, 1.0f},  {-1.0f, 1.0f, 1.0f},
    };
    static const core::u32 indices[36] = {
        0, 1, 2, 0, 2, 3, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7,
        1, 5, 6, 1, 6, 2, 4, 5, 1, 4, 1, 0, 3, 2, 6, 3, 6, 7,
    };
    // Per-triangle UVs: each face's two triangles tile the unit square.
    static const core::f32 faceUV[6][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f},
                                           {0.0f, 1.0f}};

    F s{F::fromInt(0)};
    F c{F::fromInt(0)};
    math::Cordic::sincos(rotationAngle, s, c);
    const core::f32 cf = c.toFloat();
    const core::f32 sf = s.toFloat();

    const auto view = math::Mat4<core::f32>::lookAt(Vec3f(0.0f, 0.0f, 5.0f), Vec3f(0.0f, 0.0f, 0.0f),
                                                    Vec3f(0.0f, 1.0f, 0.0f));
    const core::f32 aspect = static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height);
    const auto proj = perspectiveFov(F::fromFloat(1.04719755f), aspect, 0.1f, 100.0f);
    const auto mvp = proj * view;

    detail::ScreenVertex sv[8];
    for (core::u32 i = 0; i < 8u; ++i)
    {
        const core::f32 rx = cf * corners[i][0] + sf * corners[i][2];
        const core::f32 rz = -sf * corners[i][0] + cf * corners[i][2];
        sv[i] = detail::projectVertex(mvp, rx, corners[i][1], rz, rt.width, rt.height);
    }

    for (core::u32 t = 0; t < 12u; ++t)
        detail::fillTriangleTextured(rt, sv[indices[t * 3 + 0]], sv[indices[t * 3 + 1]], sv[indices[t * 3 + 2]],
                                     faceUV[(t & 1u) ? 3 : 0], faceUV[(t & 1u) ? 4 : 1], faceUV[(t & 1u) ? 5 : 2],
                                     tex);
}

/** @brief FNV-1a fold of the whole color buffer (cross-target signature). */
[[nodiscard]] inline core::u32 foldTarget(const RenderTarget &rt) noexcept
{
    core::u32 hash = 0x811C9DC5u;
    const core::u32 count = rt.width * rt.height;
    for (core::u32 i = 0; i < count; ++i)
        hash = detail::fnv1aStep(hash, rt.color[i]);
    return hash;
}

} // namespace lpl::render

#endif // LPL_RENDER_SOFTWARERASTERIZER_HPP
