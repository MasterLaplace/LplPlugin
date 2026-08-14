/**
 * @file Projection.hpp
 * @brief Deterministic, link-safe camera projection builders.
 *
 * Mat4<float>::perspective() calls lpl::pmr::tan(), which on the kernel target
 * lowers to a runtime tanf() libm call — undefined in a freestanding link and
 * non-deterministic across targets. These helpers instead derive tan(fov/2)
 * from the CORDIC fixed-point engine (identical lookup tables on host and
 * kernel), keeping the projection both link-safe AND bit-identical between the
 * Linux oracle and the i686 kernel. The resulting matrix is float — projection
 * is a non-authoritative render path — but produced from deterministic inputs.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_PROJECTION_HPP
#    define LPL_RENDER_PROJECTION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/Mat4.hpp>

namespace lpl::render {

/**
 * @brief Derives the vertical field of view that reproduces a physical viewing
 * geometry, so the screen subtends the same angle as the real display would
 * at the given eye distance.
 *
 * FOV = 2 * atan((screenHeight / 2) / eyeDistance), computed through CORDIC
 * atan2 so the result is the same deterministic Fixed32 radians on the Linux
 * oracle and the i686 kernel — this is meant to feed perspectiveFov() below,
 * and matters most for VR, where the per-eye FOV must match the headset's
 * physical lens geometry rather than an arbitrary aesthetic angle.
 *
 * @param screenHeight  Physical screen height (e.g. monitor height in cm, or
 *                       a per-eye lens frustum height for VR), in any unit.
 * @param eyeDistance    Distance from the eye to the screen, same unit as
 *                       screenHeight, must be > 0.
 */
[[nodiscard]] inline math::Fixed32 fovFromScreen(math::Fixed32 screenHeight, math::Fixed32 eyeDistance) noexcept
{
    const math::Fixed32 halfHeight = screenHeight / math::Fixed32::fromInt(2);
    return math::Fixed32::fromInt(2) * math::Cordic::atan2(halfHeight, eyeDistance);
}

/**
 * @brief Builds a perspective projection matrix from a Fixed32 field-of-view.
 *
 * @param fovRad      Vertical field of view, in Fixed32 radians.
 * @param aspect      Aspect ratio (width / height).
 * @param nearPlane   Near clip distance (> 0).
 * @param farPlane    Far clip distance (> near).
 */
[[nodiscard]] inline math::Mat4<core::f32> perspectiveFov(math::Fixed32 fovRad, core::f32 aspect, core::f32 nearPlane,
                                                          core::f32 farPlane) noexcept
{
    math::Fixed32 halfSin{math::Fixed32::fromInt(0)};
    math::Fixed32 halfCos{math::Fixed32::fromInt(0)};
    math::Cordic::sincos(fovRad / math::Fixed32::fromInt(2), halfSin, halfCos);
    const core::f32 tanHalf = (halfSin / halfCos).toFloat();

    math::Mat4<core::f32> r;
    r.m.fill(0.0f);
    r(0, 0) = 1.0f / (aspect * tanHalf);
    r(1, 1) = 1.0f / tanHalf;
    r(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
    r(2, 3) = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
    r(3, 2) = -1.0f;
    return r;
}

/**
 * @brief Builds an orthographic projection matrix (no transcendentals).
 */
[[nodiscard]] inline math::Mat4<core::f32> orthographic(core::f32 left, core::f32 right, core::f32 bottom,
                                                        core::f32 top, core::f32 nearPlane, core::f32 farPlane) noexcept
{
    math::Mat4<core::f32> r;
    r.m.fill(0.0f);
    r(0, 0) = 2.0f / (right - left);
    r(1, 1) = 2.0f / (top - bottom);
    r(2, 2) = -2.0f / (farPlane - nearPlane);
    r(0, 3) = -(right + left) / (right - left);
    r(1, 3) = -(top + bottom) / (top - bottom);
    r(2, 3) = -(farPlane + nearPlane) / (farPlane - nearPlane);
    r(3, 3) = 1.0f;
    return r;
}

} // namespace lpl::render

#endif // LPL_RENDER_PROJECTION_HPP
