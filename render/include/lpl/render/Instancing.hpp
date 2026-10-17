/**
 * @file Instancing.hpp
 * @brief SoA instance storage + frustum culling for batched rendering.
 *
 * Per-instance transforms (position + uniform scale) are authored in Fixed32
 * (authoritative). Frustum culling extracts the six clip-space planes from the
 * float view-projection (Gribb-Hartmann) and rejects instance bounding spheres
 * — a render-time decision in float (SSE, -ffp-contract=off), bit-identical
 * across the Linux oracle and the i686 kernel. The visible-index list and count
 * are the cross-target signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_INSTANCING_HPP
#    define LPL_RENDER_INSTANCING_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Mat4.hpp>
#    include <lpl/std/cmath.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

/** @brief Structure-of-arrays per-instance transforms (Fixed32 authority). */
class InstanceSet {
public:
    void add(math::Fixed32 x, math::Fixed32 y, math::Fixed32 z, math::Fixed32 scale);

    void clear();

    [[nodiscard]] core::u32 count() const noexcept { return static_cast<core::u32>(_px.size()); }
    [[nodiscard]] math::Fixed32 x(core::u32 i) const { return _px[i]; }
    [[nodiscard]] math::Fixed32 y(core::u32 i) const { return _py[i]; }
    [[nodiscard]] math::Fixed32 z(core::u32 i) const { return _pz[i]; }
    [[nodiscard]] math::Fixed32 scale(core::u32 i) const { return _scale[i]; }

private:
    pmr::vector<math::Fixed32> _px;
    pmr::vector<math::Fixed32> _py;
    pmr::vector<math::Fixed32> _pz;
    pmr::vector<math::Fixed32> _scale;
};

/** @brief One normalized clip-space plane: a*x + b*y + c*z + d >= 0 inside. */
struct Plane {
    core::f32 a{0.0f};
    core::f32 b{0.0f};
    core::f32 c{0.0f};
    core::f32 d{0.0f};

    [[nodiscard]] core::f32 distance(core::f32 x, core::f32 y, core::f32 z) const noexcept
    {
        return a * x + b * y + c * z + d;
    }
};

/** @brief The six frustum planes extracted from a view-projection matrix. */
struct Frustum {
    Plane planes[6];

    /**
     * @brief Extracts the planes of a view-projection matrix.
     *
     * Gribb-Hartmann: row3 +/- row{0,1,2}, read through operator()(row,col). The
     * planes come out in the order left, right, bottom, top, near, far.
     */
    [[nodiscard]] static Frustum fromViewProjection(const math::Mat4<core::f32> &vp) noexcept;

    /**
     * @brief True if a world-space bounding sphere intersects the frustum.
     */
    [[nodiscard]] bool containsSphere(core::f32 x, core::f32 y, core::f32 z, core::f32 radius) const noexcept;
};

/**
 * @brief Culls an instance set against a frustum, appending visible indices.
 *
 * @param set         SoA instance transforms.
 * @param frustum     Pre-extracted frustum.
 * @param localRadius Bounding-sphere radius of one unit instance (pre-scale).
 * @param outVisible  Receives the indices of visible instances (cleared first).
 */
inline void frustumCull(const InstanceSet &set, const Frustum &frustum, core::f32 localRadius,
                        pmr::vector<core::u32> &outVisible);

} // namespace lpl::render

#    include "Instancing.inl"

#endif // LPL_RENDER_INSTANCING_HPP
