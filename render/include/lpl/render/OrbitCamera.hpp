/**
 * @file OrbitCamera.hpp
 * @brief Orbit / walk camera: state, basis, matrices and a screen-space box cull.
 *
 * Extracted from the world viewer sample, where it had accumulated as private
 * helpers. Nothing in it was viewer-specific: an orbit around a focus point, a
 * heading that walking and strafing are expressed in, and a cull that asks
 * whether a box can possibly cover a pixel are what every application built on
 * this engine needs on its first day.
 *
 * Freestanding by construction: angles go through CORDIC, because there is no
 * libm on the kernel side and the authoritative half of the engine may not link
 * one anyway.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_ORBIT_CAMERA_HPP
#    define LPL_RENDER_ORBIT_CAMERA_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/Mat4.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::render {

/**
 * @struct CameraBasis
 * @brief An orthonormal view frame, plus the eye it is anchored at.
 *
 * The rows of a look-at matrix ARE this frame, so it is read back from the
 * matrix rather than derived a second time. Two derivations of the same frame
 * are two things that can disagree, and the disagreement shows up as a sky that
 * does not line up with the geometry under it.
 */
struct CameraBasis {
    math::Vec3<core::f32> eye{};
    math::Vec3<core::f32> forward{};
    math::Vec3<core::f32> right{};
    math::Vec3<core::f32> up{};
};

/**
 * @struct CameraLens
 * @brief What the projection needs, kept apart from where the camera is.
 */
struct CameraLens {
    core::f32 fovRadians{1.04719755f}; /**< 60 degrees. */
    core::f32 nearPlane{0.4f};
    core::f32 farPlane{600.0f};
};

/**
 * @class OrbitCamera
 * @brief Yaw/pitch/distance around a focus that can be walked around the world.
 *
 * Two modes out of one state, which is why they share a class: at a large
 * distance it orbits a point (a map view), and as the distance goes to the eye
 * height it becomes a first-person camera standing at the focus. Nothing
 * switches; the distance is the mode.
 */
class OrbitCamera {
public:
    /**
     * @brief Turns left/right.
     */
    void turn(core::f32 radians) noexcept { _yaw += radians; }

    /**
     * @brief Tilts up/down, clamped short of straight up and straight down.
     */
    void tilt(core::f32 radians) noexcept { _pitch = clamp(_pitch + radians, -1.35f, 1.45f); }

    /**
     * @brief Pulls out / pushes in, clamped to a usable range.
     */
    void zoom(core::f32 factor, core::f32 minimum = 1.6f, core::f32 maximum = 240.0f) noexcept
    {
        _distance = clamp(_distance * factor, minimum, maximum);
    }

    /**
     * @brief Moves the focus along the heading. Positive walks forward.
     */
    void walk(core::f32 amount) noexcept;

    /**
     * @brief Moves the focus along the heading rotated a quarter turn.
     */
    void strafe(core::f32 amount) noexcept;

    void setFocus(core::f32 x, core::f32 z) noexcept;

    void setYaw(core::f32 radians) noexcept { _yaw = radians; }
    void setPitch(core::f32 radians) noexcept { _pitch = radians; }
    void setDistance(core::f32 distance) noexcept { _distance = distance; }

    /**
     * @brief Height of the look-at target above the sampled ground.
     */
    void setEyeHeight(core::f32 height) noexcept { _eyeHeight = height; }

    [[nodiscard]] core::f32 yaw() const noexcept { return _yaw; }
    [[nodiscard]] core::f32 pitch() const noexcept { return _pitch; }
    [[nodiscard]] core::f32 distance() const noexcept { return _distance; }
    [[nodiscard]] core::f32 focusX() const noexcept { return _focusX; }
    [[nodiscard]] core::f32 focusZ() const noexcept { return _focusZ; }
    [[nodiscard]] core::f32 eyeHeight() const noexcept { return _eyeHeight; }

    /**
     * @brief Switches between standing in the world and orbiting a point.
     *
     * An explicit flag, not a threshold on the distance. Inferring the mode from
     * how close the orbit is puts the eye a couple of units BEHIND where the
     * walker stands, which is inside the hill they are standing on as often as
     * not — the screen fills with the back face of the ground sheet. The two
     * modes differ in what the eye is, so that is what the flag records.
     */
    void setFirstPerson(bool firstPerson) noexcept { _firstPerson = firstPerson; }

    [[nodiscard]] bool isFirstPerson() const noexcept { return _firstPerson; }

    /**
     * @brief Builds the view matrix and the frame that goes with it.
     *
     * The direction built from yaw and pitch points from the look target BACK
     * towards the eye, which is what makes the two modes one expression:
     * orbiting pushes the eye along it, standing puts the eye at the focus and
     * the target one step against it.
     *
     * @param groundHeight Terrain height under the focus, so the target sits on
     *                     the ground instead of at an arbitrary altitude.
     */
    [[nodiscard]] CameraBasis frame(core::f32 groundHeight, math::Mat4<core::f32> &outView) const noexcept;

    /**
     * @brief View-projection for a target of the given aspect ratio.
     */
    [[nodiscard]] math::Mat4<core::f32> viewProjection(core::f32 groundHeight, core::f32 aspect, const CameraLens &lens,
                                                       CameraBasis &outBasis) const noexcept;

    /**
     * @name Sine and cosine
     *
     * sin/cos through CORDIC: no libm here, and the camera basis and any
     * heading derived from it must come from the same place.
     * @{
     */
    [[nodiscard]] static core::f32 sinOf(core::f32 angle) noexcept;

    [[nodiscard]] static core::f32 cosOf(core::f32 angle) noexcept;
    /** @} */

    [[nodiscard]] static core::f32 clamp(core::f32 value, core::f32 low, core::f32 high) noexcept
    {
        return value < low ? low : (value > high ? high : value);
    }

private:
    core::f32 _yaw{0.7f};
    core::f32 _pitch{0.65f};
    core::f32 _distance{62.0f};
    core::f32 _focusX{0.0f};
    core::f32 _focusZ{0.0f};
    core::f32 _eyeHeight{2.0f};
    bool _firstPerson{false};
};

/**
 * @brief Reciprocal square root without libm, for any positive input.
 *
 * Newton's iteration converges to @c 1/sqrt(v) only from a starting point near
 * the answer, and this routine used to start from 1.0 — which is fine for a view
 * RAY, whose length is always close to one, and silently catastrophic for
 * anything else. Fed a vector 30 units long it does not converge slowly, it
 * diverges: the first pass lands at -448 and the rest run away.
 *
 * That is exactly how it failed, and the failure had a picture: a water surface
 * reflecting a garbage direction came out black while the sky above it was blue.
 * The precondition was even written down one file over. A routine that only works
 * on a range nobody checks is a trap, so the seed now comes from halving the
 * exponent in the bit pattern — the classic trick, correct for every positive
 * float, after which two passes are plenty.
 */
[[nodiscard]] inline core::f32 inverseSqrtNewton(core::f32 value, core::u32 passes = 2u) noexcept;

/**
 * @brief Cheap 2D length: max plus a fraction of min, within about 4% of exact.
 *
 * Exact enough to sort by distance or to blend fog, and it costs no square root.
 */
[[nodiscard]] inline core::f32 approximateLength(core::f32 dx, core::f32 dz) noexcept;

/**
 * @brief True when an axis-aligned box cannot cover a single pixel.
 *
 * Culls by PROJECTING the box's eight corners through the same matrix the
 * triangles use, not by an angle in the horizontal plane.
 *
 * The angle test is the one that looks obviously right and is wrong: it assumes
 * the camera looks along the ground. Pitch it down and the forward vector's
 * horizontal part shrinks until the angle it measures means nothing, and a wedge
 * of the world goes missing at the edges of the view. Corners through the same
 * matrix as the geometry cannot disagree with what is actually drawn, which is
 * the entire property a cull has to have.
 *
 * A corner behind the near plane proves nothing — the box may still cover the
 * screen — so it is used as evidence for nothing.
 */
[[nodiscard]] inline bool boxOutsideFrustum(const math::Mat4<core::f32> &mvp, core::f32 centreX, core::f32 centreY,
                                            core::f32 centreZ, core::f32 halfX, core::f32 halfY, core::f32 halfZ,
                                            core::u32 targetWidth, core::u32 targetHeight) noexcept;

} // namespace lpl::render

#    include "OrbitCamera.inl"

#endif // LPL_RENDER_ORBIT_CAMERA_HPP
