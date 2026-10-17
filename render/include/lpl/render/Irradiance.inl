#ifndef LPL_RENDER_IRRADIANCE_INL
#define LPL_RENDER_IRRADIANCE_INL

namespace lpl::render {

inline IrradianceProbe IrradianceProbe::uniform(const Vec3f &radiance) noexcept
{
    IrradianceProbe probe{};
    probe.coefficient[0] = Vec3f(radiance.x / 0.282095f, radiance.y / 0.282095f, radiance.z / 0.282095f);
    return probe;
}

namespace detail {

inline void shBasis(core::f32 x, core::f32 y, core::f32 z, core::f32 *out) noexcept
{
    out[0] = 0.282095f;
    out[1] = 0.488603f * y;
    out[2] = 0.488603f * z;
    out[3] = 0.488603f * x;
    out[4] = 1.092548f * x * y;
    out[5] = 1.092548f * y * z;
    out[6] = 0.315392f * (3.0f * z * z - 1.0f);
    out[7] = 1.092548f * x * z;
    out[8] = 0.546274f * (x * x - y * y);
}

} // namespace detail

inline IrradianceProbe projectSky(const SunState &sun, const SkyParams &params, core::u32 resolution) noexcept
{
    IrradianceProbe probe{};
    if (resolution == 0u)
        resolution = 1u;

    const core::f32 step = 2.0f / static_cast<core::f32>(resolution);
    core::f32 totalWeight = 0.0f;

    for (core::u32 face = 0u; face < 6u; ++face)
        for (core::u32 v = 0u; v < resolution; ++v)
            for (core::u32 u = 0u; u < resolution; ++u)
            {
                const core::f32 a = -1.0f + (static_cast<core::f32>(u) + 0.5f) * step;
                const core::f32 b = -1.0f + (static_cast<core::f32>(v) + 0.5f) * step;

                core::f32 dx = 0.0f;
                core::f32 dy = 0.0f;
                core::f32 dz = 0.0f;
                switch (face)
                {
                case 0u:
                    dx = 1.0f;
                    dy = -b;
                    dz = -a;
                    break;
                case 1u:
                    dx = -1.0f;
                    dy = -b;
                    dz = a;
                    break;
                case 2u:
                    dx = a;
                    dy = 1.0f;
                    dz = b;
                    break;
                case 3u:
                    dx = a;
                    dy = -1.0f;
                    dz = -b;
                    break;
                case 4u:
                    dx = a;
                    dy = -b;
                    dz = 1.0f;
                    break;
                default:
                    dx = -a;
                    dy = -b;
                    dz = -1.0f;
                    break;
                }

                const core::f32 lengthSquared = dx * dx + dy * dy + dz * dz;
                const core::f32 inverse = inverseSqrtNewton(lengthSquared);
                const core::f32 weight = step * step * inverse * inverse * inverse;

                const core::f32 nx = dx * inverse;
                const core::f32 ny = dy * inverse;
                const core::f32 nz = dz * inverse;

                const core::u32 packed = skyColour(nx, ny, nz, sun, params);
                const Vec3f radiance(static_cast<core::f32>((packed >> 16) & 0xFFu) / 255.0f,
                                     static_cast<core::f32>((packed >> 8) & 0xFFu) / 255.0f,
                                     static_cast<core::f32>(packed & 0xFFu) / 255.0f);

                core::f32 basis[9];
                detail::shBasis(nx, ny, nz, basis);
                for (core::u32 i = 0u; i < 9u; ++i)
                {
                    probe.coefficient[i].x += radiance.x * basis[i] * weight;
                    probe.coefficient[i].y += radiance.y * basis[i] * weight;
                    probe.coefficient[i].z += radiance.z * basis[i] * weight;
                }
                totalWeight += weight;
            }

    if (totalWeight > 0.0f)
    {
        const core::f32 correction = 12.566370614f / totalWeight;
        for (core::u32 i = 0u; i < 9u; ++i)
        {
            probe.coefficient[i].x *= correction;
            probe.coefficient[i].y *= correction;
            probe.coefficient[i].z *= correction;
        }
    }
    return probe;
}

inline Vec3f evaluateIrradiance(const IrradianceProbe &probe, core::f32 nx, core::f32 ny, core::f32 nz) noexcept
{
    constexpr core::f32 kBand0 = 3.141592654f;
    constexpr core::f32 kBand1 = 2.094395102f;
    constexpr core::f32 kBand2 = 0.785398163f;
    constexpr core::f32 kWeight[9] = {kBand0, kBand1, kBand1, kBand1, kBand2, kBand2, kBand2, kBand2, kBand2};

    core::f32 basis[9];
    detail::shBasis(nx, ny, nz, basis);

    Vec3f irradiance(0.0f, 0.0f, 0.0f);
    for (core::u32 i = 0u; i < 9u; ++i)
    {
        const core::f32 term = kWeight[i] * basis[i];
        irradiance.x += probe.coefficient[i].x * term;
        irradiance.y += probe.coefficient[i].y * term;
        irradiance.z += probe.coefficient[i].z * term;
    }

    constexpr core::f32 kInversePi = 0.318309886f;
    irradiance.x = irradiance.x > 0.0f ? irradiance.x * kInversePi : 0.0f;
    irradiance.y = irradiance.y > 0.0f ? irradiance.y * kInversePi : 0.0f;
    irradiance.z = irradiance.z > 0.0f ? irradiance.z * kInversePi : 0.0f;
    return irradiance;
}

} // namespace lpl::render

#endif // LPL_RENDER_IRRADIANCE_INL
