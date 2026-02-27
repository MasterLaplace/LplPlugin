/**
 * @file Camera.hpp
 * @brief Camera with view/projection matrix generation (float space).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_CAMERA_HPP
    #define LPL_RENDER_CAMERA_HPP

#include <lpl/math/Vec3.hpp>
#include <lpl/math/Quat.hpp>
#include <lpl/math/Mat4.hpp>
#include <lpl/core/Types.hpp>

namespace lpl::render {

/**
 * @class Camera
 * @brief Perspective camera used for rendering interpolation.
 *
 * All values are float (rendering is not required to be deterministic).
 */
class Camera
{
public:
    Camera() noexcept = default;

    /** @brief Sets the field-of-view in radians. */
    void setFov(core::f32 fovRadians) noexcept;

    /** @brief Sets the aspect ratio (width / height). */
    void setAspect(core::f32 aspect) noexcept;

    /** @brief Sets near/far clipping planes. */
    void setClipPlanes(core::f32 nearPlane, core::f32 farPlane) noexcept;

    /** @brief Sets position and orientation. */
    void setTransform(const math::Vec3<core::f32>& position,
                      const math::Quat<core::f32>& orientation) noexcept;

    /** @brief Returns the view matrix. */
    [[nodiscard]] math::Mat4<core::f32> viewMatrix() const noexcept;

    /** @brief Returns the projection matrix. */
    [[nodiscard]] math::Mat4<core::f32> projectionMatrix() const noexcept;

    /** @brief Returns the combined view-projection matrix. */
    [[nodiscard]] math::Mat4<core::f32> viewProjectionMatrix() const noexcept;

    [[nodiscard]] const math::Vec3<core::f32>& position() const noexcept { return _position; }
    [[nodiscard]] const math::Quat<core::f32>& orientation() const noexcept { return _orientation; }

private:
    math::Vec3<core::f32>  _position{};
    math::Quat<core::f32>  _orientation{};
    core::f32              _fov{1.0472f};
    core::f32              _aspect{16.0f / 9.0f};
    core::f32              _nearPlane{0.1f};
    core::f32              _farPlane{1000.0f};
};

} // namespace lpl::render

#endif // LPL_RENDER_CAMERA_HPP
