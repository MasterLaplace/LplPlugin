#ifndef LPL_RENDER_ORBIT_CAMERA_INL
#define LPL_RENDER_ORBIT_CAMERA_INL

namespace lpl::render {

inline void OrbitCamera::walk(core::f32 amount) noexcept
{
    _focusX -= amount * sinOf(_yaw);
    _focusZ -= amount * cosOf(_yaw);
}

inline void OrbitCamera::strafe(core::f32 amount) noexcept
{
    _focusX += amount * cosOf(_yaw);
    _focusZ -= amount * sinOf(_yaw);
}

inline void OrbitCamera::setFocus(core::f32 x, core::f32 z) noexcept
{
    _focusX = x;
    _focusZ = z;
}

inline CameraBasis OrbitCamera::frame(core::f32 groundHeight, math::Mat4<core::f32> &outView) const noexcept
{
    const core::f32 cp = cosOf(_pitch);
    const core::f32 dirX = cp * sinOf(_yaw);
    const core::f32 dirY = sinOf(_pitch);
    const core::f32 dirZ = cp * cosOf(_yaw);
    const math::Vec3<core::f32> stand(_focusX, groundHeight + _eyeHeight, _focusZ);

    math::Vec3<core::f32> eye = stand;
    math::Vec3<core::f32> target(stand.x - dirX, stand.y - dirY, stand.z - dirZ);
    if (!_firstPerson)
    {
        target = stand;
        eye = math::Vec3<core::f32>(stand.x + _distance * dirX, stand.y + _distance * dirY, stand.z + _distance * dirZ);
    }

    outView = math::Mat4<core::f32>::lookAt(eye, target, math::Vec3<core::f32>(0.0f, 1.0f, 0.0f));

    CameraBasis basis{};
    basis.eye = eye;
    basis.forward = math::Vec3<core::f32>(-dirX, -dirY, -dirZ);
    basis.right = math::Vec3<core::f32>(outView(0, 0), outView(0, 1), outView(0, 2));
    basis.up = math::Vec3<core::f32>(outView(1, 0), outView(1, 1), outView(1, 2));
    return basis;
}

inline math::Mat4<core::f32> OrbitCamera::viewProjection(core::f32 groundHeight, core::f32 aspect,
                                                         const CameraLens &lens, CameraBasis &outBasis) const noexcept
{
    math::Mat4<core::f32> view{};
    outBasis = frame(groundHeight, view);
    return perspectiveFov(math::Fixed32::fromFloat(lens.fovRadians), aspect, lens.nearPlane, lens.farPlane) * view;
}

inline core::f32 OrbitCamera::sinOf(core::f32 angle) noexcept
{
    math::Fixed32 s{}, c{};
    math::Cordic::sincos(math::Fixed32::fromFloat(angle), s, c);
    return s.toFloat();
}

inline core::f32 OrbitCamera::cosOf(core::f32 angle) noexcept
{
    math::Fixed32 s{}, c{};
    math::Cordic::sincos(math::Fixed32::fromFloat(angle), s, c);
    return c.toFloat();
}

inline core::f32 inverseSqrtNewton(core::f32 value, core::u32 passes) noexcept
{
    if (value <= 0.0f)
        return 0.0f;

    core::u32 bits = 0u;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    bits = 0x5F3759DFu - (bits >> 1u);
    core::f32 inverse = 0.0f;
    __builtin_memcpy(&inverse, &bits, sizeof(inverse));

    for (core::u32 i = 0u; i < passes; ++i)
        inverse = inverse * (1.5f - 0.5f * value * inverse * inverse);
    return inverse;
}

inline core::f32 approximateLength(core::f32 dx, core::f32 dz) noexcept
{
    const core::f32 ax = dx < 0.0f ? -dx : dx;
    const core::f32 az = dz < 0.0f ? -dz : dz;
    return ax > az ? ax + 0.42f * az : az + 0.42f * ax;
}

inline bool boxOutsideFrustum(const math::Mat4<core::f32> &mvp, core::f32 centreX, core::f32 centreY, core::f32 centreZ,
                              core::f32 halfX, core::f32 halfY, core::f32 halfZ, core::u32 targetWidth,
                              core::u32 targetHeight) noexcept
{
    bool anyInFront = false;
    bool allLeft = true;
    bool allRight = true;
    bool allAbove = true;
    bool allBelow = true;

    for (core::u32 corner = 0u; corner < 8u; ++corner)
    {
        const core::f32 x = centreX + ((corner & 1u) != 0u ? halfX : -halfX);
        const core::f32 z = centreZ + ((corner & 2u) != 0u ? halfZ : -halfZ);
        const core::f32 y = centreY + ((corner & 4u) != 0u ? halfY : -halfY);
        const auto projected = detail::projectVertex(mvp, x, y, z, targetWidth, targetHeight);
        if (!projected.valid)
        {
            allLeft = allRight = allAbove = allBelow = false;
            continue;
        }
        anyInFront = true;
        allLeft = allLeft && projected.x < 0.0f;
        allRight = allRight && projected.x > static_cast<core::f32>(targetWidth);
        allAbove = allAbove && projected.y < 0.0f;
        allBelow = allBelow && projected.y > static_cast<core::f32>(targetHeight);
    }

    return anyInFront && (allLeft || allRight || allAbove || allBelow);
}

} // namespace lpl::render

#endif // LPL_RENDER_ORBIT_CAMERA_INL
