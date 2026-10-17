#ifndef LPL_RENDER_REFLECTION_INL
#define LPL_RENDER_REFLECTION_INL

namespace lpl::render {

inline CameraBasis mirrorBasisAboutPlane(const CameraBasis &basis, core::f32 planeY) noexcept
{
    CameraBasis mirrored = basis;
    mirrored.eye.y = 2.0f * planeY - basis.eye.y;
    mirrored.forward.y = -basis.forward.y;
    mirrored.up.y = -basis.up.y;
    return mirrored;
}

inline math::Mat4<core::f32> mirrorViewProjection(const CameraBasis &mirrored, core::f32 aspect,
                                                  const CameraLens &lens) noexcept
{
    const math::Vec3<core::f32> target(mirrored.eye.x + mirrored.forward.x, mirrored.eye.y + mirrored.forward.y,
                                       mirrored.eye.z + mirrored.forward.z);
    const auto view = math::Mat4<core::f32>::lookAt(
        mirrored.eye, target, math::Vec3<core::f32>(0.0f, mirrored.up.y < 0.0f ? -1.0f : 1.0f, 0.0f));
    return perspectiveFov(math::Fixed32::fromFloat(lens.fovRadians), aspect, lens.nearPlane, lens.farPlane) * view;
}

inline bool sampleProbe(const ReflectionProbe &probe, const Texture &pixels, core::f32 worldX, core::f32 worldY,
                        core::f32 worldZ, core::u32 &outColour) noexcept
{
    if (!probe.valid || probe.width == 0u || probe.height == 0u)
        return false;

    const detail::ClipVertex clip = detail::toClip(probe.mirrorMvp, worldX, worldY, worldZ);
    if (clip.w < kNearPlaneEpsilon)
        return false;
    const core::f32 invW = 1.0f / clip.w;
    const core::f32 u = clip.x * invW * 0.5f + 0.5f;
    const core::f32 v = 0.5f - clip.y * invW * 0.5f;
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)
        return false;

    outColour = pixels.sampleBilinear(static_cast<core::u32>(u * 65535.0f), static_cast<core::u32>(v * 65535.0f));
    return true;
}

} // namespace lpl::render

#endif // LPL_RENDER_REFLECTION_INL
