/**
 * @file Topology.hpp
 * @brief Deterministic curve/surface topology: Bezier, Catmull-Rom, parametric
 *        surface tessellation and a Delaunay (Bowyer-Watson) triangulation.
 *
 * Control points are authored in Fixed32 (authoritative geometry). Curve and
 * surface evaluation runs in float (SSE, -ffp-contract=off) which is
 * bit-identical across the Linux oracle and the i686 kernel; no transcendentals
 * are used (only +,-,*,/ and the hardware sqrt for the circumcircle test). The
 * Delaunay test compares squared distances in float against the triangle
 * circumcircle, so the resulting triangle index list is the cross-target
 * signature. Evaluated samples are quantized to Q16.16 before folding.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_TOPOLOGY_HPP
#    define LPL_RENDER_TOPOLOGY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

using Vec3fTopo = math::Vec3<core::f32>;

namespace detail {

/** @brief Quantize a float position to Q16.16 and fold its three components. */
[[nodiscard]] inline core::u32 foldVec3(core::u32 hash, const Vec3fTopo &p) noexcept
{
    hash = fnv1aStep(hash, static_cast<core::u32>(static_cast<core::i32>(p.x * detail::kQ16FoldScale)));
    hash = fnv1aStep(hash, static_cast<core::u32>(static_cast<core::i32>(p.y * detail::kQ16FoldScale)));
    hash = fnv1aStep(hash, static_cast<core::u32>(static_cast<core::i32>(p.z * detail::kQ16FoldScale)));
    return hash;
}

} // namespace detail

/**
 * @brief Evaluates a cubic Bezier at parameter t in [0,1] (de Casteljau form).
 */
[[nodiscard]] inline Vec3fTopo bezierCubic(const Vec3fTopo &p0, const Vec3fTopo &p1, const Vec3fTopo &p2,
                                           const Vec3fTopo &p3, core::f32 t) noexcept
{
    const core::f32 u = 1.0f - t;
    const core::f32 w0 = u * u * u;
    const core::f32 w1 = 3.0f * u * u * t;
    const core::f32 w2 = 3.0f * u * t * t;
    const core::f32 w3 = t * t * t;
    return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

/**
 * @brief Evaluates a centripetal-uniform Catmull-Rom spline segment p1..p2.
 *
 * Uniform parameterization (tangents = (p2-p0)/2 and (p3-p1)/2), Hermite basis.
 */
[[nodiscard]] inline Vec3fTopo catmullRom(const Vec3fTopo &p0, const Vec3fTopo &p1, const Vec3fTopo &p2,
                                          const Vec3fTopo &p3, core::f32 t) noexcept
{
    const core::f32 t2 = t * t;
    const core::f32 t3 = t2 * t;
    const core::f32 h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    const core::f32 h10 = t3 - 2.0f * t2 + t;
    const core::f32 h01 = -2.0f * t3 + 3.0f * t2;
    const core::f32 h11 = t3 - t2;
    const Vec3fTopo m1 = (p2 - p0) * 0.5f;
    const Vec3fTopo m2 = (p3 - p1) * 0.5f;
    return p1 * h00 + m1 * h10 + p2 * h01 + m2 * h11;
}

/** @brief Result of tessellating a curve/surface: folded sample stream. */
struct TessellationResult {
    core::u32 sample_count{0u};     ///< Number of evaluated samples.
    core::u32 sample_signature{0u}; ///< FNV-1a fold of Q16.16-quantized samples.
};

/**
 * @brief Tessellates a closed Catmull-Rom loop through Fixed32 control points
 *        at `segments` samples per span and folds the quantized positions.
 *
 * @param ctrl     Control points (authoritative Fixed32, XYZ triples).
 * @param count    Number of control points (>= 4).
 * @param segments Samples emitted per control-point span.
 */
[[nodiscard]] inline TessellationResult tessellateCatmullLoop(const math::Fixed32 (*ctrl)[3], core::u32 count,
                                                              core::u32 segments) noexcept
{
    TessellationResult out{};
    if (count < 4u || segments == 0u)
        return out;

    core::u32 hash = detail::kFnv1aOffsetBasis;
    const core::f32 inv = 1.0f / static_cast<core::f32>(segments);
    for (core::u32 i = 0; i < count; ++i)
    {
        const auto at = [&](core::u32 k) {
            const core::u32 j = (i + k) % count;
            return Vec3fTopo(ctrl[j][0].toFloat(), ctrl[j][1].toFloat(), ctrl[j][2].toFloat());
        };
        const Vec3fTopo p0 = at(count - 1u + 0u); // i-1
        const Vec3fTopo p1 = at(0u);              // i
        const Vec3fTopo p2 = at(1u);              // i+1
        const Vec3fTopo p3 = at(2u);              // i+2
        for (core::u32 s = 0; s < segments; ++s)
        {
            const core::f32 t = static_cast<core::f32>(s) * inv;
            hash = detail::foldVec3(hash, catmullRom(p0, p1, p2, p3, t));
            ++out.sample_count;
        }
    }
    out.sample_signature = hash;
    return out;
}

/**
 * @brief Tessellates a height-field parametric surface z = f(x,y) sampled on a
 *        Fixed32 grid, producing (res+1)^2 vertices folded in scan order.
 *
 * The surface is the analytic saddle z = (x*x - y*y) over [-1,1]^2 — purely
 * polynomial, so no transcendentals. `res` is the subdivision count per axis.
 */
[[nodiscard]] inline TessellationResult tessellateSaddle(core::u32 res) noexcept
{
    TessellationResult out{};
    if (res == 0u)
        return out;

    core::u32 hash = detail::kFnv1aOffsetBasis;
    const core::f32 step = 2.0f / static_cast<core::f32>(res);
    for (core::u32 j = 0; j <= res; ++j)
        for (core::u32 i = 0; i <= res; ++i)
        {
            const core::f32 x = -1.0f + step * static_cast<core::f32>(i);
            const core::f32 y = -1.0f + step * static_cast<core::f32>(j);
            const core::f32 z = (x * x - y * y) * 0.5f; // displacement
            hash = detail::foldVec3(hash, Vec3fTopo(x, y, z));
            ++out.sample_count;
        }
    out.sample_signature = hash;
    return out;
}

/** @brief Result of a Delaunay triangulation: triangle count + index fold. */
struct DelaunayResult {
    core::u32 point_count{0u};        ///< Number of input points.
    core::u32 triangle_count{0u};     ///< Triangles in the final triangulation.
    core::u32 triangle_signature{0u}; ///< FNV-1a fold of all (i,j,k) indices.
};

namespace detail {

struct Tri2 {
    core::u32 a, b, c;
};

/** @brief True if point p lies inside triangle (A,B,C)'s circumcircle. */
[[nodiscard]] inline bool inCircumcircle(core::f32 ax, core::f32 ay, core::f32 bx, core::f32 by, core::f32 cx,
                                         core::f32 cy, core::f32 px, core::f32 py) noexcept
{
    // Standard incircle determinant; assumes (A,B,C) CCW. Sign normalized below.
    const core::f32 adx = ax - px, ady = ay - py;
    const core::f32 bdx = bx - px, bdy = by - py;
    const core::f32 cdx = cx - px, cdy = cy - py;
    const core::f32 ad = adx * adx + ady * ady;
    const core::f32 bd = bdx * bdx + bdy * bdy;
    const core::f32 cd = cdx * cdx + cdy * cdy;
    const core::f32 det = adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx);
    // Orientation of (A,B,C).
    const core::f32 orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    return (orient > 0.0f) ? (det > 0.0f) : (det < 0.0f);
}

} // namespace detail

/**
 * @brief Bowyer-Watson Delaunay triangulation of 2D points (Fixed32 authority).
 *
 * Points are authored in Fixed32 (only X,Y used). The incircle predicate runs
 * in float (deterministic SSE), so the emitted triangle index list is
 * bit-identical across targets. A super-triangle bootstraps the insertion and
 * its vertices are discarded from the result.
 *
 * @param pts   Fixed32 point triples (Z ignored).
 * @param count Number of input points (>= 3).
 */
[[nodiscard]] inline DelaunayResult delaunay2D(const math::Fixed32 (*pts)[3], core::u32 count)
{
    DelaunayResult out{};
    out.point_count = count;
    if (count < 3u)
        return out;

    // Working point list: inputs followed by 3 super-triangle vertices.
    pmr::vector<core::f32> px;
    pmr::vector<core::f32> py;
    px.resize(count + 3u, 0.0f);
    py.resize(count + 3u, 0.0f);
    core::f32 minX = pts[0][0].toFloat(), maxX = minX, minY = pts[0][1].toFloat(), maxY = minY;
    for (core::u32 i = 0; i < count; ++i)
    {
        px[i] = pts[i][0].toFloat();
        py[i] = pts[i][1].toFloat();
        minX = px[i] < minX ? px[i] : minX;
        maxX = px[i] > maxX ? px[i] : maxX;
        minY = py[i] < minY ? py[i] : minY;
        maxY = py[i] > maxY ? py[i] : maxY;
    }
    const core::f32 dx = maxX - minX, dy = maxY - minY;
    const core::f32 dmax = (dx > dy ? dx : dy) + 1.0f;
    const core::f32 midX = (minX + maxX) * 0.5f, midY = (minY + maxY) * 0.5f;
    const core::u32 s0 = count, s1 = count + 1u, s2 = count + 2u;
    px[s0] = midX - 20.0f * dmax;
    py[s0] = midY - dmax;
    px[s1] = midX;
    py[s1] = midY + 20.0f * dmax;
    px[s2] = midX + 20.0f * dmax;
    py[s2] = midY - dmax;

    pmr::vector<detail::Tri2> tris;
    tris.push_back(detail::Tri2{s0, s1, s2});

    pmr::vector<detail::Tri2> bad;
    pmr::vector<core::u32> polyA;
    pmr::vector<core::u32> polyB;
    for (core::u32 ip = 0; ip < count; ++ip)
    {
        bad.clear();
        for (core::u32 ti = 0; ti < static_cast<core::u32>(tris.size()); ++ti)
        {
            const detail::Tri2 &t = tris[ti];
            if (detail::inCircumcircle(px[t.a], py[t.a], px[t.b], py[t.b], px[t.c], py[t.c], px[ip], py[ip]))
                bad.push_back(t);
        }
        // Boundary edges of the bad-triangle cavity (edges not shared by two).
        polyA.clear();
        polyB.clear();
        for (core::u32 bi = 0; bi < static_cast<core::u32>(bad.size()); ++bi)
        {
            const core::u32 e[3][2] = {
                {bad[bi].a, bad[bi].b},
                {bad[bi].b, bad[bi].c},
                {bad[bi].c, bad[bi].a}
            };
            for (core::u32 k = 0; k < 3u; ++k)
            {
                bool shared = false;
                for (core::u32 bj = 0; bj < static_cast<core::u32>(bad.size()); ++bj)
                {
                    if (bj == bi)
                        continue;
                    const core::u32 f[3][2] = {
                        {bad[bj].a, bad[bj].b},
                        {bad[bj].b, bad[bj].c},
                        {bad[bj].c, bad[bj].a}
                    };
                    for (core::u32 m = 0; m < 3u; ++m)
                        if ((e[k][0] == f[m][0] && e[k][1] == f[m][1]) || (e[k][0] == f[m][1] && e[k][1] == f[m][0]))
                            shared = true;
                }
                if (!shared)
                {
                    polyA.push_back(e[k][0]);
                    polyB.push_back(e[k][1]);
                }
            }
        }
        // Remove bad triangles.
        pmr::vector<detail::Tri2> keep;
        for (core::u32 ti = 0; ti < static_cast<core::u32>(tris.size()); ++ti)
        {
            bool isBad = false;
            for (core::u32 bi = 0; bi < static_cast<core::u32>(bad.size()); ++bi)
                if (tris[ti].a == bad[bi].a && tris[ti].b == bad[bi].b && tris[ti].c == bad[bi].c)
                    isBad = true;
            if (!isBad)
                keep.push_back(tris[ti]);
        }
        tris = keep;
        // Re-triangulate the cavity against the new point.
        for (core::u32 k = 0; k < static_cast<core::u32>(polyA.size()); ++k)
            tris.push_back(detail::Tri2{polyA[k], polyB[k], ip});
    }

    // Emit triangles that don't touch the super-triangle, folding sorted indices.
    core::u32 hash = detail::kFnv1aOffsetBasis;
    for (core::u32 ti = 0; ti < static_cast<core::u32>(tris.size()); ++ti)
    {
        const detail::Tri2 &t = tris[ti];
        if (t.a >= count || t.b >= count || t.c >= count)
            continue;
        hash = detail::fnv1aStep(hash, t.a);
        hash = detail::fnv1aStep(hash, t.b);
        hash = detail::fnv1aStep(hash, t.c);
        ++out.triangle_count;
    }
    out.triangle_signature = hash;
    return out;
}

} // namespace lpl::render

#endif // LPL_RENDER_TOPOLOGY_HPP
