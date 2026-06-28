/**
 * @file RenderParity.hpp
 * @brief Shared deterministic 3D projection scene used by both the Linux oracle
 *        parity test and the in-kernel smoke, so they fold the SAME geometry.
 *
 * Geometry and the model transform are authored in Fixed32 (CORDIC rotation) =
 * authoritative state. The view/projection and the perspective divide run in
 * float (SSE, -ffp-contract=off) which P1 proved bit-identical across the host
 * and the i686 kernel. The folded signature MUST match on both targets.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_RENDERPARITY_HPP
#    define LPL_RENDER_RENDERPARITY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/Mat4.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/render/Instancing.hpp>
#    include <lpl/render/Projection.hpp>

namespace lpl::render {

/** @brief Result of projecting the canonical parity cube through a camera. */
struct ProjectedSceneResult {
    core::u32 screen_signature{0u}; ///< FNV-1a fold of all 8 floored screen (x,y).
    core::u32 depth_signature{0u};  ///< FNV-1a fold of all 8 quantized NDC depths.
    core::i32 vertex0_x{0};         ///< Floored screen X of vertex 0 (witness).
    core::i32 vertex0_y{0};         ///< Floored screen Y of vertex 0 (witness).
    core::u32 in_front_count{0u};   ///< Vertices with w > 0 (in front of camera).
};

namespace detail {

[[nodiscard]] inline core::u32 fnv1aStep(core::u32 hash, core::u32 value) noexcept
{
    for (core::u32 i = 0; i < 4u; ++i)
    {
        hash ^= (value >> (i * 8u)) & 0xFFu;
        hash *= 0x01000193u;
    }
    return hash;
}

} // namespace detail

/**
 * @brief Projects an 8-vertex unit cube through a rotate+translate model matrix
 *        and a perspective camera, folding the screen coordinates and depths.
 *
 * @param rotationAngle Model Y-axis rotation, Fixed32 radians (authoritative).
 * @param screenWidth   Viewport width in pixels.
 * @param screenHeight  Viewport height in pixels.
 */
[[nodiscard]] inline ProjectedSceneResult projectParityCube(math::Fixed32 rotationAngle, core::u32 screenWidth,
                                                            core::u32 screenHeight) noexcept
{
    using F = math::Fixed32;
    using Vec3f = math::Vec3<core::f32>;

    // Unit cube authored in Fixed32 (authoritative geometry), corners at +/-1.
    const F one = F::fromInt(1);
    const F neg = F::fromInt(-1);
    const F cubeFx[8][3] = {
        {neg, neg, neg}, {one, neg, neg}, {one, one, neg}, {neg, one, neg},
        {neg, neg, one}, {one, neg, one}, {one, one, one}, {neg, one, one},
    };

    // Model matrix: Fixed32 CORDIC rotation about Y, then translate -Z slightly.
    F s{F::fromInt(0)};
    F c{F::fromInt(0)};
    math::Cordic::sincos(rotationAngle, s, c);

    // Camera: eye at (0,0,5) looking at origin, 60-degree vertical FOV.
    const auto view = math::Mat4<core::f32>::lookAt(Vec3f(0.0f, 0.0f, 5.0f), Vec3f(0.0f, 0.0f, 0.0f),
                                                    Vec3f(0.0f, 1.0f, 0.0f));
    const core::f32 aspect = static_cast<core::f32>(screenWidth) / static_cast<core::f32>(screenHeight);
    const auto proj = perspectiveFov(F::fromFloat(1.04719755f), aspect, 0.1f, 100.0f);
    const auto viewProj = proj * view;

    const core::f32 cf = c.toFloat();
    const core::f32 sf = s.toFloat();

    ProjectedSceneResult out{};
    core::u32 screenHash = 0x811C9DC5u;
    core::u32 depthHash = 0x811C9DC5u;

    for (core::u32 i = 0; i < 8u; ++i)
    {
        // Rotate about Y in float (derived from CORDIC-deterministic sin/cos).
        const core::f32 x0 = cubeFx[i][0].toFloat();
        const core::f32 y0 = cubeFx[i][1].toFloat();
        const core::f32 z0 = cubeFx[i][2].toFloat();
        const core::f32 rx = cf * x0 + sf * z0;
        const core::f32 rz = -sf * x0 + cf * z0;
        const Vec3f world(rx, y0, rz);

        // Manual clip-space transform to recover w for the perspective divide.
        const core::f32 cx = viewProj(0, 0) * world.x + viewProj(0, 1) * world.y + viewProj(0, 2) * world.z +
                             viewProj(0, 3);
        const core::f32 cy = viewProj(1, 0) * world.x + viewProj(1, 1) * world.y + viewProj(1, 2) * world.z +
                             viewProj(1, 3);
        const core::f32 cz = viewProj(2, 0) * world.x + viewProj(2, 1) * world.y + viewProj(2, 2) * world.z +
                             viewProj(2, 3);
        const core::f32 cw = viewProj(3, 0) * world.x + viewProj(3, 1) * world.y + viewProj(3, 2) * world.z +
                             viewProj(3, 3);

        if (cw > 0.0f)
            ++out.in_front_count;

        const core::f32 invW = (cw != 0.0f) ? (1.0f / cw) : 1.0f;
        const core::f32 ndcX = cx * invW;
        const core::f32 ndcY = cy * invW;
        const core::f32 ndcZ = cz * invW;

        const core::i32 sx = static_cast<core::i32>((ndcX * 0.5f + 0.5f) * static_cast<core::f32>(screenWidth));
        const core::i32 sy = static_cast<core::i32>((0.5f - ndcY * 0.5f) * static_cast<core::f32>(screenHeight));
        const core::i32 dq = static_cast<core::i32>(ndcZ * 65536.0f);

        screenHash = detail::fnv1aStep(screenHash, static_cast<core::u32>(sx));
        screenHash = detail::fnv1aStep(screenHash, static_cast<core::u32>(sy));
        depthHash = detail::fnv1aStep(depthHash, static_cast<core::u32>(dq));

        if (i == 0)
        {
            out.vertex0_x = sx;
            out.vertex0_y = sy;
        }
    }

    out.screen_signature = screenHash;
    out.depth_signature = depthHash;
    return out;
}

/** @brief Result of frustum-culling the canonical parity instance grid. */
struct CullResult {
    core::u32 total{0u};             ///< Total instances in the grid.
    core::u32 visible{0u};           ///< Instances surviving the frustum cull.
    core::u32 visible_signature{0u}; ///< FNV-1a fold of the visible-index list.
};

/**
 * @brief Builds a deterministic 7x7 instance grid (SoA) on the XZ plane and
 *        culls it against a perspective camera looking at the origin.
 */
[[nodiscard]] inline CullResult cullParityInstanceGrid(core::u32 screenWidth, core::u32 screenHeight)
{
    using F = math::Fixed32;
    using Vec3f = math::Vec3<core::f32>;

    InstanceSet set;
    for (core::i32 gz = -3; gz <= 3; ++gz)
        for (core::i32 gx = -3; gx <= 3; ++gx)
            set.add(F::fromInt(gx * 5), F::fromInt(0), F::fromInt(gz * 5), F::fromInt(1));

    const auto view = math::Mat4<core::f32>::lookAt(Vec3f(0.0f, 8.0f, 18.0f), Vec3f(0.0f, 0.0f, 0.0f),
                                                    Vec3f(0.0f, 1.0f, 0.0f));
    const core::f32 aspect = static_cast<core::f32>(screenWidth) / static_cast<core::f32>(screenHeight);
    const auto proj = perspectiveFov(F::fromFloat(1.04719755f), aspect, 0.1f, 100.0f);
    const auto frustum = Frustum::fromViewProjection(proj * view);

    pmr::vector<core::u32> visible;
    frustumCull(set, frustum, 1.73205081f /* unit-cube circumradius */, visible);

    CullResult out{};
    out.total = set.count();
    out.visible = static_cast<core::u32>(visible.size());
    core::u32 hash = 0x811C9DC5u;
    for (core::u32 i = 0; i < out.visible; ++i)
        hash = detail::fnv1aStep(hash, visible[i]);
    out.visible_signature = hash;
    return out;
}

} // namespace lpl::render

#endif // LPL_RENDER_RENDERPARITY_HPP
