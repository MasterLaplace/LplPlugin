#ifndef LPL_RENDER_LIGHTING_INL
#define LPL_RENDER_LIGHTING_INL

namespace lpl::render {

namespace detail {

inline core::f32 intPow(core::f32 base, core::u32 exp) noexcept
{
    core::f32 result = 1.0f;
    core::f32 b = base;
    while (exp != 0u)
    {
        if (exp & 1u)
            result *= b;
        b *= b;
        exp >>= 1u;
    }
    return result;
}

} // namespace detail

inline Vec3f shadeFragment(ShadingModel model, const Material &mat, const Light &light, Vec3f normal, Vec3f fragPos,
                           Vec3f viewPos) noexcept
{
    const Vec3f N = normal.normalize();

    Vec3f L{};
    core::f32 attenuation = 1.0f;
    if (light.type == LightType::Directional)
    {
        L = (-light.direction).normalize();
    }
    else
    {
        const Vec3f toLight = light.position - fragPos;
        const core::f32 dist = lpl::pmr::sqrt(toLight.lengthSquared());
        L = (dist > 0.0f) ? (toLight / dist) : Vec3f(0.0f, 1.0f, 0.0f);
        const core::f32 d = dist / light.range;
        attenuation = detail::saturate(1.0f - d * d);
        if (light.type == LightType::Spot)
        {
            const core::f32 spotCos = (-L).dot(light.direction.normalize());
            if (spotCos < light.spotCosCutoff)
                attenuation = 0.0f;
        }
    }

    const Vec3f radiance = light.color * (light.intensity * attenuation);

    const core::f32 ndotl = detail::saturate(N.dot(L));
    Vec3f color = mat.albedo * ndotl;

    if (model != ShadingModel::Lambert && ndotl > 0.0f)
    {
        const Vec3f V = (viewPos - fragPos).normalize();
        core::f32 specBase = 0.0f;
        if (model == ShadingModel::Phong)
        {
            const Vec3f R = N * (2.0f * N.dot(L)) - L;
            specBase = detail::saturate(R.dot(V));
        }
        else
        {
            const Vec3f H = (L + V).normalize();
            specBase = detail::saturate(N.dot(H));
        }
        const core::f32 spec = detail::intPow(specBase, mat.shininess) * mat.specularStrength;
        color = color + Vec3f(spec, spec, spec);
    }

    return Vec3f(color.x * radiance.x, color.y * radiance.y, color.z * radiance.z);
}

inline core::u32 shadeToRgb(ShadingModel model, const Material &mat, const Light *lights, core::u32 lightCount,
                            Vec3f normal, Vec3f fragPos, Vec3f viewPos) noexcept
{
    Vec3f acc = mat.albedo * mat.ambient;
    for (core::u32 i = 0; i < lightCount; ++i)
        acc = acc + shadeFragment(model, mat, lights[i], normal, fragPos, viewPos);

    const core::u32 r = static_cast<core::u32>(detail::saturate(acc.x) * 255.0f + 0.5f);
    const core::u32 g = static_cast<core::u32>(detail::saturate(acc.y) * 255.0f + 0.5f);
    const core::u32 b = static_cast<core::u32>(detail::saturate(acc.z) * 255.0f + 0.5f);
    return (r << 16) | (g << 8) | b;
}

inline core::u32 modulate(core::u32 colour, core::f32 lit) noexcept
{
    const core::u32 scale = static_cast<core::u32>((lit < 0.0f ? 0.0f : lit) * 256.0f);
    const auto channel = [scale](core::u32 value) -> core::u32 {
        const core::u32 lit8 = (value * scale) >> 8;
        return lit8 > 255u ? 255u : lit8;
    };
    return (channel((colour >> 16) & 0xFFu) << 16) | (channel((colour >> 8) & 0xFFu) << 8) | channel(colour & 0xFFu);
}

inline core::u32 mixColours(core::u32 from, core::u32 to, core::f32 t) noexcept
{
    const core::f32 clamped = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const core::u32 weight = static_cast<core::u32>(clamped * 256.0f);
    core::u32 out = 0u;
    for (core::u32 shift = 0u; shift <= 16u; shift += 8u)
    {
        const core::u32 a = (from >> shift) & 0xFFu;
        const core::u32 b = (to >> shift) & 0xFFu;
        out |= (((a * (256u - weight) + b * weight) >> 8) & 0xFFu) << shift;
    }
    return out;
}

inline core::u32 heatRamp(core::f32 t) noexcept
{
    t = detail::saturate(t);
    const core::u32 r = static_cast<core::u32>((0.05f + 0.95f * t) * 255.0f);
    const core::u32 g = static_cast<core::u32>((0.05f + 0.62f * t * t) * 255.0f);
    const core::u32 b = static_cast<core::u32>((0.10f + 0.15f * t * t * t) * 255.0f);
    return (r << 16) | (g << 8) | b;
}

} // namespace lpl::render

#endif // LPL_RENDER_LIGHTING_INL
