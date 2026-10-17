#ifndef LPL_RENDER_PROJECTION_INL
#define LPL_RENDER_PROJECTION_INL

namespace lpl::render {

inline math::Fixed32 fovFromScreen(math::Fixed32 screenHeight, math::Fixed32 eyeDistance) noexcept
{
    const math::Fixed32 halfHeight = screenHeight / math::Fixed32::fromInt(2);
    return math::Fixed32::fromInt(2) * math::Cordic::atan2(halfHeight, eyeDistance);
}

inline math::Mat4<core::f32> perspectiveFov(math::Fixed32 fovRad, core::f32 aspect, core::f32 nearPlane,
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

inline math::Mat4<core::f32> orthographic(core::f32 left, core::f32 right, core::f32 bottom, core::f32 top,
                                          core::f32 nearPlane, core::f32 farPlane) noexcept
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

#endif // LPL_RENDER_PROJECTION_INL
