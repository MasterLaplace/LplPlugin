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
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/render/Texture.hpp>

namespace lpl::render {

/** @brief Caller-owned render target: packed 0x00RRGGBB color + float depth. */
struct RenderTarget {
    core::u32 *color{nullptr}; ///< width*height packed pixels (row-major, no padding).
    core::f32 *depth{nullptr}; ///< width*height depth, smaller = nearer.
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
inline void fillTriangle(const RenderTarget &rt, const ScreenVertex &v0, const ScreenVertex &v1, const ScreenVertex &v2,
                         core::u32 color) noexcept
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
            const core::u32 uQ16 = static_cast<core::u32>(u * detail::kQ16FoldScale);
            const core::u32 vQ16 = static_cast<core::u32>(v * detail::kQ16FoldScale);

            rt.depth[idx] = depth;
            rt.color[idx] = tex.sampleBilinear(uQ16, vQ16);
        }
    }
}

/**
 * @brief Fills a triangle whichever way it faces.
 *
 * @c fillTriangle culls anything whose screen-space area is negative, which is
 * exactly right for a closed solid: a cube's far faces are hidden by its near
 * ones, so drawing them is wasted work. A heightfield is not a solid. It is a
 * single sheet with no inside, and half of it faces away from any given camera.
 * Culled, the ground vanishes entirely and only what stands on it survives,
 * which looks like a broken projection and is in fact a correct rasterizer
 * being asked the wrong question.
 *
 * Swapping two vertices when the area comes out negative costs one edge
 * function and draws the sheet from both sides.
 */
inline void fillTriangleDoubleSided(const RenderTarget &rt, const ScreenVertex &a, const ScreenVertex &b,
                                    const ScreenVertex &c, core::u32 colour) noexcept
{
    if (edge(a, b, c.x, c.y) > 0.0f)
        fillTriangle(rt, a, b, c, colour);
    else
        fillTriangle(rt, a, c, b, colour);
}

/** @brief Two double-sided triangles over four already-projected corners. */
inline void fillQuadDoubleSided(const RenderTarget &rt, const ScreenVertex &a, const ScreenVertex &b,
                                const ScreenVertex &c, const ScreenVertex &d, core::u32 colour) noexcept
{
    fillTriangleDoubleSided(rt, a, b, c, colour);
    fillTriangleDoubleSided(rt, a, c, d, colour);
}

/** @brief One vertex in clip space, before the divide by w. */
struct ClipVertex {
    core::f32 x{0.0f};
    core::f32 y{0.0f};
    core::f32 z{0.0f};
    core::f32 w{0.0f};
};

/** @brief Transforms a world position into clip space. */
[[nodiscard]] inline ClipVertex toClip(const math::Mat4<core::f32> &mvp, core::f32 wx, core::f32 wy,
                                       core::f32 wz) noexcept
{
    ClipVertex v{};
    v.x = mvp(0, 0) * wx + mvp(0, 1) * wy + mvp(0, 2) * wz + mvp(0, 3);
    v.y = mvp(1, 0) * wx + mvp(1, 1) * wy + mvp(1, 2) * wz + mvp(1, 3);
    v.z = mvp(2, 0) * wx + mvp(2, 1) * wy + mvp(2, 2) * wz + mvp(2, 3);
    v.w = mvp(3, 0) * wx + mvp(3, 1) * wy + mvp(3, 2) * wz + mvp(3, 3);
    return v;
}

/** @brief Divides by w and maps to pixels. */
[[nodiscard]] inline ScreenVertex toScreen(const ClipVertex &v, core::u32 width, core::u32 height) noexcept
{
    ScreenVertex out{};
    if (v.w <= 0.0f)
        return out;
    const core::f32 invW = 1.0f / v.w;
    out.x = (v.x * invW * 0.5f + 0.5f) * static_cast<core::f32>(width);
    out.y = (0.5f - v.y * invW * 0.5f) * static_cast<core::f32>(height);
    out.z = v.z * invW;
    out.valid = true;
    return out;
}

} // namespace detail

/// Smallest @c w a vertex may have and still be projected.
inline constexpr core::f32 kNearPlaneEpsilon = 0.05f;

/**
 * @struct ShadedVertex
 * @brief A clip-space vertex that also carries the world point it came from.
 *
 * The world position is what a per-pixel shader wants: sample a procedural
 * texture at it, reflect a view ray about it, decide a material from its
 * altitude. Carrying it alongside the clip position means the clipper can
 * interpolate both with one parameter, so a clipped edge gets the right world
 * point and not one guessed from the screen.
 */
struct ShadedVertex {
    detail::ClipVertex clip{};
    core::f32 wx{0.0f};
    core::f32 wy{0.0f};
    core::f32 wz{0.0f};
};

/**
 * @brief Fills a triangle by CALLING a shader for every pixel it covers.
 *
 * The rasterizer had one fill: a flat colour decided per primitive. Everything
 * that varies within a surface — a texture, a reflection, a gradient along a
 * slope — had to be approximated by cutting the surface into more primitives, and
 * a lake ends up as a thousand flat quads that still cannot reflect anything,
 * because a reflection depends on the direction to the pixel, not to the quad.
 *
 * @c shader receives the interpolated world position and returns a packed colour.
 * Interpolation is PERSPECTIVE-CORRECT: the barycentric weights are divided
 * through by w, without which a texture on a surface seen at a grazing angle
 * swims as the camera turns — the classic artefact of affine interpolation, and
 * the reason the existing textured fill is only used on cube faces seen head-on.
 */
template <typename Shader>
inline void fillTriangleShaded(const RenderTarget &rt, const ShadedVertex &a, const ShadedVertex &b,
                               const ShadedVertex &c, Shader &&shader) noexcept
{
    const detail::ScreenVertex v0 = detail::toScreen(a.clip, rt.width, rt.height);
    const detail::ScreenVertex v1 = detail::toScreen(b.clip, rt.width, rt.height);
    const detail::ScreenVertex v2 = detail::toScreen(c.clip, rt.width, rt.height);
    if (!v0.valid || !v1.valid || !v2.valid)
        return;

    core::f32 area = detail::edge(v0, v1, v2.x, v2.y);
    const detail::ScreenVertex *s0 = &v0;
    const detail::ScreenVertex *s1 = &v1;
    const detail::ScreenVertex *s2 = &v2;
    const ShadedVertex *w0v = &a;
    const ShadedVertex *w1v = &b;
    const ShadedVertex *w2v = &c;
    if (area < 0.0f)
    {
        // Double-sided, like the rest of the terrain path: a heightfield is a
        // sheet, and half of it faces away from any camera.
        s1 = &v2;
        s2 = &v1;
        w1v = &c;
        w2v = &b;
        area = -area;
    }
    if (area <= 0.0f)
        return;
    const core::f32 invArea = 1.0f / area;

    const core::f32 invW0 = 1.0f / w0v->clip.w;
    const core::f32 invW1 = 1.0f / w1v->clip.w;
    const core::f32 invW2 = 1.0f / w2v->clip.w;

    core::f32 minXf = s0->x, maxXf = s0->x, minYf = s0->y, maxYf = s0->y;
    minXf = s1->x < minXf ? s1->x : minXf;
    minXf = s2->x < minXf ? s2->x : minXf;
    maxXf = s1->x > maxXf ? s1->x : maxXf;
    maxXf = s2->x > maxXf ? s2->x : maxXf;
    minYf = s1->y < minYf ? s1->y : minYf;
    minYf = s2->y < minYf ? s2->y : minYf;
    maxYf = s1->y > maxYf ? s1->y : maxYf;
    maxYf = s2->y > maxYf ? s2->y : maxYf;

    const core::i32 minX = detail::clampInt(static_cast<core::i32>(minXf), 0, static_cast<core::i32>(rt.width) - 1);
    const core::i32 maxX = detail::clampInt(static_cast<core::i32>(maxXf), 0, static_cast<core::i32>(rt.width) - 1);
    const core::i32 minY = detail::clampInt(static_cast<core::i32>(minYf), 0, static_cast<core::i32>(rt.height) - 1);
    const core::i32 maxY = detail::clampInt(static_cast<core::i32>(maxYf), 0, static_cast<core::i32>(rt.height) - 1);

    for (core::i32 y = minY; y <= maxY; ++y)
    {
        const core::f32 py = static_cast<core::f32>(y) + 0.5f;
        for (core::i32 x = minX; x <= maxX; ++x)
        {
            const core::f32 px = static_cast<core::f32>(x) + 0.5f;
            const core::f32 b0 = detail::edge(*s1, *s2, px, py) * invArea;
            const core::f32 b1 = detail::edge(*s2, *s0, px, py) * invArea;
            const core::f32 b2 = detail::edge(*s0, *s1, px, py) * invArea;
            if (b0 < 0.0f || b1 < 0.0f || b2 < 0.0f)
                continue;

            const core::f32 depth = b0 * s0->z + b1 * s1->z + b2 * s2->z;
            const core::u32 index = static_cast<core::u32>(y) * rt.width + static_cast<core::u32>(x);
            if (depth >= rt.depth[index])
                continue;

            const core::f32 wSum = b0 * invW0 + b1 * invW1 + b2 * invW2;
            if (wSum <= 0.0f)
                continue;
            const core::f32 invSum = 1.0f / wSum;
            const core::f32 worldX = (b0 * w0v->wx * invW0 + b1 * w1v->wx * invW1 + b2 * w2v->wx * invW2) * invSum;
            const core::f32 worldY = (b0 * w0v->wy * invW0 + b1 * w1v->wy * invW1 + b2 * w2v->wy * invW2) * invSum;
            const core::f32 worldZ = (b0 * w0v->wz * invW0 + b1 * w1v->wz * invW1 + b2 * w2v->wz * invW2) * invSum;

            rt.depth[index] = depth;
            rt.color[index] = shader(worldX, worldY, worldZ);
        }
    }
}

/**
 * @brief Near-plane-clipped, per-pixel-shaded convex polygon.
 *
 * The shaded twin of @ref fillPolygonClipped, and it clips for the same reason:
 * the polygon under the walker's feet crosses the near plane every frame.
 */
template <typename Shader>
inline core::u32 fillPolygonShadedClipped(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                          const core::f32 *worldPoints, core::u32 count, Shader &&shader) noexcept
{
    if (worldPoints == nullptr || count < 3u || count > 8u)
        return 0u;

    ShadedVertex input[16];
    bool anyBehind = false;
    for (core::u32 i = 0u; i < count; ++i)
    {
        input[i].wx = worldPoints[i * 3u];
        input[i].wy = worldPoints[i * 3u + 1u];
        input[i].wz = worldPoints[i * 3u + 2u];
        input[i].clip = detail::toClip(mvp, input[i].wx, input[i].wy, input[i].wz);
        anyBehind = anyBehind || input[i].clip.w < kNearPlaneEpsilon;
    }

    ShadedVertex polygon[16];
    core::u32 polygonCount = 0u;
    if (!anyBehind)
    {
        for (core::u32 i = 0u; i < count; ++i)
            polygon[polygonCount++] = input[i];
    }
    else
    {
        for (core::u32 i = 0u; i < count; ++i)
        {
            const ShadedVertex &current = input[i];
            const ShadedVertex &next = input[(i + 1u) % count];
            const bool currentIn = current.clip.w >= kNearPlaneEpsilon;
            const bool nextIn = next.clip.w >= kNearPlaneEpsilon;
            if (currentIn)
                polygon[polygonCount++] = current;
            if (currentIn != nextIn)
            {
                const core::f32 t = (kNearPlaneEpsilon - current.clip.w) / (next.clip.w - current.clip.w);
                ShadedVertex cut{};
                cut.clip.x = current.clip.x + (next.clip.x - current.clip.x) * t;
                cut.clip.y = current.clip.y + (next.clip.y - current.clip.y) * t;
                cut.clip.z = current.clip.z + (next.clip.z - current.clip.z) * t;
                cut.clip.w = kNearPlaneEpsilon;
                cut.wx = current.wx + (next.wx - current.wx) * t;
                cut.wy = current.wy + (next.wy - current.wy) * t;
                cut.wz = current.wz + (next.wz - current.wz) * t;
                polygon[polygonCount++] = cut;
            }
            if (polygonCount + 2u >= 16u)
                break;
        }
        if (polygonCount < 3u)
            return 0u;
    }

    core::u32 triangles = 0u;
    for (core::u32 i = 1u; i + 1u < polygonCount; ++i)
    {
        fillTriangleShaded(rt, polygon[0], polygon[i], polygon[i + 1u], shader);
        ++triangles;
    }
    return triangles;
}

/**
 * @brief Fills a convex polygon, clipping it against the near plane first.
 *
 * The reason this exists, and why every large primitive should go through it:
 * @c projectVertex only rejects vertices with @c w <= 0. A vertex a centimetre in
 * FRONT of the eye has a tiny positive w, survives that test, and comes out at
 * screen coordinates in the hundreds of thousands. The triangle it belongs to is
 * then stretched across the entire frame — on screen, a beam radiating from a
 * point, or a black wedge where ground should be. Both were blamed on the
 * geometry that produced them; the geometry was fine, and the projection was
 * being asked to divide by nearly zero.
 *
 * Dropping such a primitive instead is not a fix either: a terrain quad under the
 * walker's feet always straddles the near plane, so dropping it punches a hole in
 * the world exactly where the walker is looking. The only correct answer is to cut
 * the polygon along the plane and draw the part that is in front — Sutherland and
 * Hodgman in clip space, which is where the plane is a straight comparison rather
 * than a projection.
 *
 * @param worldPoints Vertices, x/y/z triples, convex and in order.
 * @param count       Number of vertices (3 or 4 in practice).
 * @return Triangles filled.
 */
inline core::u32 fillPolygonClipped(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                    const core::f32 *worldPoints, core::u32 count, core::u32 colour) noexcept
{
    if (worldPoints == nullptr || count < 3u || count > 8u)
        return 0u;

    detail::ClipVertex input[8];
    bool anyBehind = false;
    for (core::u32 i = 0u; i < count; ++i)
    {
        input[i] = detail::toClip(mvp, worldPoints[i * 3u], worldPoints[i * 3u + 1u], worldPoints[i * 3u + 2u]);
        anyBehind = anyBehind || input[i].w < kNearPlaneEpsilon;
    }

    detail::ScreenVertex screen[16];
    core::u32 screenCount = 0u;

    if (!anyBehind)
    {
        // The common case, and it must stay as cheap as it was: nothing crosses
        // the plane, so there is nothing to clip.
        for (core::u32 i = 0u; i < count; ++i)
            screen[screenCount++] = detail::toScreen(input[i], rt.width, rt.height);
    }
    else
    {
        detail::ClipVertex clipped[16];
        core::u32 clippedCount = 0u;
        for (core::u32 i = 0u; i < count; ++i)
        {
            const detail::ClipVertex &current = input[i];
            const detail::ClipVertex &next = input[(i + 1u) % count];
            const bool currentIn = current.w >= kNearPlaneEpsilon;
            const bool nextIn = next.w >= kNearPlaneEpsilon;

            if (currentIn)
                clipped[clippedCount++] = current;
            if (currentIn != nextIn)
            {
                // Where the edge crosses w = epsilon, linearly in clip space.
                const core::f32 t = (kNearPlaneEpsilon - current.w) / (next.w - current.w);
                detail::ClipVertex cut{};
                cut.x = current.x + (next.x - current.x) * t;
                cut.y = current.y + (next.y - current.y) * t;
                cut.z = current.z + (next.z - current.z) * t;
                cut.w = kNearPlaneEpsilon;
                clipped[clippedCount++] = cut;
            }
            if (clippedCount + 2u >= 16u)
                break;
        }
        if (clippedCount < 3u)
            return 0u; // entirely behind the eye
        for (core::u32 i = 0u; i < clippedCount; ++i)
            screen[screenCount++] = detail::toScreen(clipped[i], rt.width, rt.height);
    }

    core::u32 triangles = 0u;
    for (core::u32 i = 1u; i + 1u < screenCount; ++i)
    {
        detail::fillTriangleDoubleSided(rt, screen[0], screen[i], screen[i + 1u], colour);
        ++triangles;
    }
    return triangles;
}

/**
 * @brief A vertical quad hanging from an edge segment: a LOD skirt.
 *
 * Adjacent chunks sampled at different strides do not agree along their shared
 * edge, and the disagreement shows as a crack straight through to the sky.
 * A curtain dropped from the edge does not make the two agree; it puts opaque
 * geometry behind the gap, which is all the eye needs.
 */
inline core::u32 drawSkirtQuad(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, core::f32 ax, core::f32 ay,
                               core::f32 az, core::f32 bx, core::f32 by, core::f32 bz, core::f32 drop,
                               core::u32 colour) noexcept
{
    const core::f32 quad[12] = {ax, ay, az, bx, by, bz, bx, by - drop, bz, ax, ay - drop, az};
    return fillPolygonClipped(rt, mvp, quad, 4u, colour);
}

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
        {-1.0f, -1.0f, -1.0f},
        {1.0f,  -1.0f, -1.0f},
        {1.0f,  1.0f,  -1.0f},
        {-1.0f, 1.0f,  -1.0f},
        {-1.0f, -1.0f, 1.0f },
        {1.0f,  -1.0f, 1.0f },
        {1.0f,  1.0f,  1.0f },
        {-1.0f, 1.0f,  1.0f },
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

    const auto view =
        math::Mat4<core::f32>::lookAt(Vec3f(0.0f, 0.0f, 5.0f), Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f));
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
        {-1.0f, -1.0f, -1.0f},
        {1.0f,  -1.0f, -1.0f},
        {1.0f,  1.0f,  -1.0f},
        {-1.0f, 1.0f,  -1.0f},
        {-1.0f, -1.0f, 1.0f },
        {1.0f,  -1.0f, 1.0f },
        {1.0f,  1.0f,  1.0f },
        {-1.0f, 1.0f,  1.0f },
    };
    static const core::u32 indices[36] = {
        0, 1, 2, 0, 2, 3, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 1, 5, 6, 1, 6, 2, 4, 5, 1, 4, 1, 0, 3, 2, 6, 3, 6, 7,
    };
    // Per-triangle UVs: each face's two triangles tile the unit square.
    static const core::f32 faceUV[6][2] = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f}
    };

    F s{F::fromInt(0)};
    F c{F::fromInt(0)};
    math::Cordic::sincos(rotationAngle, s, c);
    const core::f32 cf = c.toFloat();
    const core::f32 sf = s.toFloat();

    const auto view =
        math::Mat4<core::f32>::lookAt(Vec3f(0.0f, 0.0f, 5.0f), Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f));
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
                                     faceUV[(t & 1u) ? 3 : 0], faceUV[(t & 1u) ? 4 : 1], faceUV[(t & 1u) ? 5 : 2], tex);
}

/**
 * @brief Renders the canonical cube with classical per-face flat lighting
 *        (Lambert/Phong/Blinn-Phong) from one directional + one point light.
 */
inline void renderLitCube(const RenderTarget &rt, math::Fixed32 rotationAngle, ShadingModel model) noexcept
{
    using F = math::Fixed32;

    clearTarget(rt, 0x00102030u);

    static const core::f32 corners[8][3] = {
        {-1.0f, -1.0f, -1.0f},
        {1.0f,  -1.0f, -1.0f},
        {1.0f,  1.0f,  -1.0f},
        {-1.0f, 1.0f,  -1.0f},
        {-1.0f, -1.0f, 1.0f },
        {1.0f,  -1.0f, 1.0f },
        {1.0f,  1.0f,  1.0f },
        {-1.0f, 1.0f,  1.0f },
    };
    static const core::u32 indices[36] = {
        0, 1, 2, 0, 2, 3, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 1, 5, 6, 1, 6, 2, 4, 5, 1, 4, 1, 0, 3, 2, 6, 3, 6, 7,
    };
    static const core::f32 faceNormals[6][3] = {
        {0.0f,  0.0f,  -1.0f},
        {0.0f,  0.0f,  1.0f },
        {-1.0f, 0.0f,  0.0f },
        {1.0f,  0.0f,  0.0f },
        {0.0f,  -1.0f, 0.0f },
        {0.0f,  1.0f,  0.0f },
    };

    F sn{F::fromInt(0)};
    F cs{F::fromInt(0)};
    math::Cordic::sincos(rotationAngle, sn, cs);
    const core::f32 cf = cs.toFloat();
    const core::f32 sf = sn.toFloat();

    const Vec3f eye(0.0f, 0.0f, 5.0f);
    const auto view = math::Mat4<core::f32>::lookAt(eye, Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f));
    const core::f32 aspect = static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height);
    const auto proj = perspectiveFov(F::fromFloat(1.04719755f), aspect, 0.1f, 100.0f);
    const auto mvp = proj * view;

    Light lights[2];
    lights[0].type = LightType::Directional;
    lights[0].direction = Vec3f(-0.4f, -0.7f, -0.6f);
    lights[0].color = Vec3f(1.0f, 0.95f, 0.85f);
    lights[0].intensity = 1.0f;
    lights[1].type = LightType::Point;
    lights[1].position = Vec3f(2.5f, 2.0f, 3.0f);
    lights[1].color = Vec3f(0.3f, 0.5f, 1.0f);
    lights[1].intensity = 1.0f;
    lights[1].range = 12.0f;

    Material mat;
    mat.albedo = Vec3f(0.8f, 0.7f, 0.6f);
    mat.shininess = 32u;

    detail::ScreenVertex sv[8];
    core::f32 wpos[8][3];
    for (core::u32 i = 0; i < 8u; ++i)
    {
        const core::f32 rx = cf * corners[i][0] + sf * corners[i][2];
        const core::f32 rz = -sf * corners[i][0] + cf * corners[i][2];
        wpos[i][0] = rx;
        wpos[i][1] = corners[i][1];
        wpos[i][2] = rz;
        sv[i] = detail::projectVertex(mvp, rx, corners[i][1], rz, rt.width, rt.height);
    }

    for (core::u32 t = 0; t < 12u; ++t)
    {
        const core::u32 face = t / 2u;
        const core::f32 nx = cf * faceNormals[face][0] + sf * faceNormals[face][2];
        const core::f32 nz = -sf * faceNormals[face][0] + cf * faceNormals[face][2];
        const Vec3f normal(nx, faceNormals[face][1], nz);

        const core::u32 i0 = indices[t * 3 + 0];
        const core::u32 i1 = indices[t * 3 + 1];
        const core::u32 i2 = indices[t * 3 + 2];
        const Vec3f center((wpos[i0][0] + wpos[i1][0] + wpos[i2][0]) / 3.0f,
                           (wpos[i0][1] + wpos[i1][1] + wpos[i2][1]) / 3.0f,
                           (wpos[i0][2] + wpos[i1][2] + wpos[i2][2]) / 3.0f);
        const core::u32 color = shadeToRgb(model, mat, lights, 2u, normal, center, eye);
        detail::fillTriangle(rt, sv[i0], sv[i1], sv[i2], color);
    }
}

/** @brief Copies a render target's color buffer into a same-size Texture. */
/**
 * @brief Copies a render target into an EXISTING texture, allocating nothing.
 *
 * @ref targetToTexture returns a fresh texture, which is right for a one-shot fold
 * and wrong once per frame: at 240x150 that is 144 KiB of allocation and a copy
 * every time, inside the render path. Measured, it was enough to stall the world's
 * streaming — the reflection looked correct and the world stopped arriving.
 *
 * @return false if the texture's size does not match, so a caller cannot silently
 *         sample a stale image of the wrong shape.
 */
inline bool copyTargetToTexture(const RenderTarget &rt, Texture &out) noexcept
{
    if (out.width() != rt.width || out.height() != rt.height || rt.color == nullptr)
        return false;
    for (core::u32 y = 0u; y < rt.height; ++y)
        for (core::u32 x = 0u; x < rt.width; ++x)
            out.setTexel(x, y, rt.color[y * rt.width + x]);
    return true;
}

[[nodiscard]] inline Texture targetToTexture(const RenderTarget &rt)
{
    Texture tex(rt.width, rt.height);
    for (core::u32 y = 0; y < rt.height; ++y)
        for (core::u32 x = 0; x < rt.width; ++x)
            tex.setTexel(x, y, rt.color[y * rt.width + x]);
    return tex;
}

/**
 * @brief Renders a 2x2 multi-viewport composite: four lit cubes at distinct
 *        rotation angles, each into its own quadrant of the target.
 */
inline void renderMultiViewport(const RenderTarget &composite) noexcept
{
    clearTarget(composite, 0x00000000u);

    const core::u32 halfW = composite.width / 2u;
    const core::u32 halfH = composite.height / 2u;
    if (halfW == 0u || halfH == 0u)
        return;

    const math::Fixed32 angles[4] = {
        math::Fixed32::fromInt(0),
        math::Fixed32::fromFloat(0.78539816f),
        math::Fixed32::fromFloat(1.57079633f),
        math::Fixed32::fromFloat(2.35619449f),
    };
    const ShadingModel models[4] = {ShadingModel::Lambert, ShadingModel::Phong, ShadingModel::BlinnPhong,
                                    ShadingModel::BlinnPhong};

    pmr::vector<core::u32> quadColor;
    pmr::vector<core::f32> quadDepth;
    quadColor.resize(static_cast<core::usize>(halfW) * halfH, 0u);
    quadDepth.resize(static_cast<core::usize>(halfW) * halfH, 0.0f);
    RenderTarget quad{quadColor.data(), quadDepth.data(), halfW, halfH};

    for (core::u32 q = 0; q < 4u; ++q)
    {
        renderLitCube(quad, angles[q], models[q]);
        const core::u32 ox = (q & 1u) ? halfW : 0u;
        const core::u32 oy = (q & 2u) ? halfH : 0u;
        for (core::u32 y = 0; y < halfH; ++y)
            for (core::u32 x = 0; x < halfW; ++x)
                composite.color[(oy + y) * composite.width + (ox + x)] = quadColor[y * halfW + x];
    }
}

/**
 * @brief Render-to-texture: renders a lit cube into an offscreen texture, then
 *        maps that texture onto a second cube drawn into the target.
 */
inline void renderToTextureCube(const RenderTarget &rt, math::Fixed32 angle) noexcept
{
    constexpr core::u32 kTexDim = 64u;
    pmr::vector<core::u32> texColor;
    pmr::vector<core::f32> texDepth;
    texColor.resize(static_cast<core::usize>(kTexDim) * kTexDim, 0u);
    texDepth.resize(static_cast<core::usize>(kTexDim) * kTexDim, 0.0f);
    RenderTarget offscreen{texColor.data(), texDepth.data(), kTexDim, kTexDim};

    renderLitCube(offscreen, angle, ShadingModel::BlinnPhong);
    const Texture rtt = targetToTexture(offscreen);
    renderTexturedCube(rt, angle, rtt);
}

/** @brief FNV-1a fold of the whole color buffer (cross-target signature). */
[[nodiscard]] inline core::u32 foldTarget(const RenderTarget &rt) noexcept
{
    core::u32 hash = detail::kFnv1aOffsetBasis;
    const core::u32 count = rt.width * rt.height;
    for (core::u32 i = 0; i < count; ++i)
        hash = detail::fnv1aStep(hash, rt.color[i]);
    return hash;
}

} // namespace lpl::render

#endif // LPL_RENDER_SOFTWARERASTERIZER_HPP
