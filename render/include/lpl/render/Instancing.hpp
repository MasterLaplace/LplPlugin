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
    void add(math::Fixed32 x, math::Fixed32 y, math::Fixed32 z, math::Fixed32 scale)
    {
        _px.push_back(x);
        _py.push_back(y);
        _pz.push_back(z);
        _scale.push_back(scale);
    }

    void clear()
    {
        _px.clear();
        _py.clear();
        _pz.clear();
        _scale.clear();
    }

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

    [[nodiscard]] static Frustum fromViewProjection(const math::Mat4<core::f32> &vp) noexcept
    {
        Frustum f{};
        // Gribb-Hartmann: row3 +/- row{0,1,2}. operator()(row,col).
        const core::f32 r0[4] = {vp(0, 0), vp(0, 1), vp(0, 2), vp(0, 3)};
        const core::f32 r1[4] = {vp(1, 0), vp(1, 1), vp(1, 2), vp(1, 3)};
        const core::f32 r2[4] = {vp(2, 0), vp(2, 1), vp(2, 2), vp(2, 3)};
        const core::f32 r3[4] = {vp(3, 0), vp(3, 1), vp(3, 2), vp(3, 3)};

        const core::f32 src[6][4] = {
            {r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]}, // left
            {r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]}, // right
            {r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]}, // bottom
            {r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]}, // top
            {r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]}, // near
            {r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]}, // far
        };
        for (core::u32 i = 0; i < 6u; ++i)
        {
            const core::f32 len = lpl::pmr::sqrt(src[i][0] * src[i][0] + src[i][1] * src[i][1] + src[i][2] * src[i][2]);
            const core::f32 inv = (len > 0.0f) ? (1.0f / len) : 1.0f;
            f.planes[i] = Plane{src[i][0] * inv, src[i][1] * inv, src[i][2] * inv, src[i][3] * inv};
        }
        return f;
    }

    /** @brief True if a world-space bounding sphere intersects the frustum. */
    [[nodiscard]] bool containsSphere(core::f32 x, core::f32 y, core::f32 z, core::f32 radius) const noexcept
    {
        for (core::u32 i = 0; i < 6u; ++i)
            if (planes[i].distance(x, y, z) < -radius)
                return false;
        return true;
    }
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
                        pmr::vector<core::u32> &outVisible)
{
    outVisible.clear();
    const core::u32 n = set.count();
    for (core::u32 i = 0; i < n; ++i)
    {
        const core::f32 wx = set.x(i).toFloat();
        const core::f32 wy = set.y(i).toFloat();
        const core::f32 wz = set.z(i).toFloat();
        const core::f32 radius = set.scale(i).toFloat() * localRadius;
        if (frustum.containsSphere(wx, wy, wz, radius))
            outVisible.push_back(i);
    }
}

} // namespace lpl::render

#endif // LPL_RENDER_INSTANCING_HPP
