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

/**
 * @brief Quantize a float position to Q16.16 and fold its three components.
 */
[[nodiscard]] inline core::u32 foldVec3(core::u32 hash, const Vec3fTopo &p) noexcept;

} // namespace detail

/**
 * @brief Evaluates a cubic Bezier at parameter t in [0,1] (de Casteljau form).
 */
[[nodiscard]] inline Vec3fTopo bezierCubic(const Vec3fTopo &p0, const Vec3fTopo &p1, const Vec3fTopo &p2,
                                           const Vec3fTopo &p3, core::f32 t) noexcept;

/**
 * @brief Evaluates a centripetal-uniform Catmull-Rom spline segment p1..p2.
 *
 * Uniform parameterization (tangents = (p2-p0)/2 and (p3-p1)/2), Hermite basis.
 */
[[nodiscard]] inline Vec3fTopo catmullRom(const Vec3fTopo &p0, const Vec3fTopo &p1, const Vec3fTopo &p2,
                                          const Vec3fTopo &p3, core::f32 t) noexcept;

/** @brief Result of tessellating a curve/surface: folded sample stream. */
struct TessellationResult {
    core::u32 sample_count{0u};     /**< Number of evaluated samples. */
    core::u32 sample_signature{0u}; /**< FNV-1a fold of Q16.16-quantized samples. */
};

/**
 * @brief Tessellates a closed Catmull-Rom loop and EMITS the points.
 *
 * The fold-only twin below returns a signature and a count, which is everything a
 * parity gate needs and nothing a mesh needs. That is why this module could pass
 * its gate for months without a single triangle in the world coming out of it: the
 * geometry was computed and thrown away. This variant writes the samples out, so a
 * caller can revolve or extrude them into something you can look at.
 *
 * @param outPoints Destination, at least @c count * @c segments entries.
 * @param capacity  Entries available; sampling stops when it is full.
 * @return Points written.
 */
inline core::u32 catmullLoopPoints(const math::Fixed32 (*ctrl)[3], core::u32 count, core::u32 segments,
                                   Vec3fTopo *outPoints, core::u32 capacity) noexcept;

/**
 * @brief Tessellates a closed Catmull-Rom loop through Fixed32 control points
 *        at `segments` samples per span and folds the quantized positions.
 *
 * @param ctrl     Control points (authoritative Fixed32, XYZ triples).
 * @param count    Number of control points (>= 4).
 * @param segments Samples emitted per control-point span.
 */
[[nodiscard]] inline TessellationResult tessellateCatmullLoop(const math::Fixed32 (*ctrl)[3], core::u32 count,
                                                              core::u32 segments) noexcept;

/**
 * @brief Tessellates a height-field parametric surface z = f(x,y) sampled on a
 *        Fixed32 grid, producing (res+1)^2 vertices folded in scan order.
 *
 * The surface is the analytic saddle z = (x*x - y*y) / 2 over [-1,1]^2 — purely
 * polynomial, so no transcendentals. `res` is the subdivision count per axis.
 */
[[nodiscard]] inline TessellationResult tessellateSaddle(core::u32 res) noexcept;

/** @brief Result of a Delaunay triangulation: triangle count + index fold. */
struct DelaunayResult {
    core::u32 point_count{0u};        /**< Number of input points. */
    core::u32 triangle_count{0u};     /**< Triangles in the final triangulation. */
    core::u32 triangle_signature{0u}; /**< FNV-1a fold of all (i,j,k) indices. */
};

namespace detail {

struct Tri2 {
    core::u32 a, b, c;
};

/**
 * @brief True if point p lies inside triangle (A,B,C)'s circumcircle.
 *
 * The standard incircle determinant assumes (A,B,C) counter-clockwise; its sign
 * is normalized by the orientation of (A,B,C), so either winding answers right.
 */
[[nodiscard]] inline bool inCircumcircle(core::f32 ax, core::f32 ay, core::f32 bx, core::f32 by, core::f32 cx,
                                         core::f32 cy, core::f32 px, core::f32 py) noexcept;

} // namespace detail

/**
 * @brief Bowyer-Watson Delaunay triangulation of 2D points (Fixed32 authority).
 *
 * Points are authored in Fixed32 (only X,Y used). The incircle predicate runs
 * in float (deterministic SSE), so the emitted triangle index list is
 * bit-identical across targets. A super-triangle bootstraps the insertion and
 * its vertices are discarded from the result.
 *
 * Each point removes the triangles whose circumcircle holds it, finds the
 * boundary of the cavity they leave (the edges no two of them share) and joins
 * every boundary edge to the point. The signature folds the (a, b, c) indices of
 * every triangle that does not touch the super-triangle, in list order.
 *
 * @param pts   Fixed32 point triples (Z ignored).
 * @param count Number of input points (>= 3).
 */
[[nodiscard]] inline DelaunayResult delaunay2D(const math::Fixed32 (*pts)[3], core::u32 count);

} // namespace lpl::render

#    include "Topology.inl"

#endif // LPL_RENDER_TOPOLOGY_HPP
