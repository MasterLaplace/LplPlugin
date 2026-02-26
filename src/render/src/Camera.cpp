// /////////////////////////////////////////////////////////////////////////////
/// @file Camera.cpp
/// @brief Camera implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/render/Camera.hpp>

namespace lpl::render {

void Camera::setFov(core::f32 fovRadians) noexcept           { fov_ = fovRadians; }
void Camera::setAspect(core::f32 aspect) noexcept             { aspect_ = aspect; }

void Camera::setClipPlanes(core::f32 nearPlane, core::f32 farPlane) noexcept
{
    nearPlane_ = nearPlane;
    farPlane_ = farPlane;
}

void Camera::setTransform(const math::Vec3<core::f32>& position,
                           const math::Quat<core::f32>& orientation) noexcept
{
    position_ = position;
    orientation_ = orientation;
}

math::Mat4<core::f32> Camera::viewMatrix() const noexcept
{
    const auto forward = math::Vec3<core::f32>(0.0f, 0.0f, -1.0f);
    const auto up      = math::Vec3<core::f32>(0.0f, 1.0f, 0.0f);
    const auto target  = position_ + orientation_.rotate(forward);
    const auto upDir   = orientation_.rotate(up);
    return math::Mat4<core::f32>::lookAt(position_, target, upDir);
}

math::Mat4<core::f32> Camera::projectionMatrix() const noexcept
{
    return math::Mat4<core::f32>::perspective(fov_, aspect_, nearPlane_, farPlane_);
}

math::Mat4<core::f32> Camera::viewProjectionMatrix() const noexcept
{
    return projectionMatrix() * viewMatrix();
}

} // namespace lpl::render
