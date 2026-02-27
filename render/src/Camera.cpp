/**
 * @file Camera.cpp
 * @brief Camera implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/render/Camera.hpp>

namespace lpl::render {

void Camera::setFov(core::f32 fovRadians) noexcept           { _fov = fovRadians; }
void Camera::setAspect(core::f32 aspect) noexcept             { _aspect = aspect; }

void Camera::setClipPlanes(core::f32 nearPlane, core::f32 farPlane) noexcept
{
    _nearPlane = nearPlane;
    _farPlane = farPlane;
}

void Camera::setTransform(const math::Vec3<core::f32>& position,
                           const math::Quat<core::f32>& orientation) noexcept
{
    _position = position;
    _orientation = orientation;
}

math::Mat4<core::f32> Camera::viewMatrix() const noexcept
{
    const auto forward = math::Vec3<core::f32>(0.0f, 0.0f, -1.0f);
    const auto up      = math::Vec3<core::f32>(0.0f, 1.0f, 0.0f);
    const auto target  = _position + _orientation.rotate(forward);
    const auto upDir   = _orientation.rotate(up);
    return math::Mat4<core::f32>::lookAt(_position, target, upDir);
}

math::Mat4<core::f32> Camera::projectionMatrix() const noexcept
{
    return math::Mat4<core::f32>::perspective(_fov, _aspect, _nearPlane, _farPlane);
}

math::Mat4<core::f32> Camera::viewProjectionMatrix() const noexcept
{
    return projectionMatrix() * viewMatrix();
}

} // namespace lpl::render
