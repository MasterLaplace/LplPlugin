#ifndef LPL_RENDER_SKY_INL
#define LPL_RENDER_SKY_INL

namespace lpl::render {

inline SunState sunAt(core::f32 dayFraction, core::f32 tilt) noexcept
{
    const core::f32 angle = (dayFraction - 0.25f) * 6.28318531f;
    math::Fixed32 s{};
    math::Fixed32 c{};
    math::Cordic::sincos(math::Fixed32::fromFloat(angle), s, c);
    math::Fixed32 ts{};
    math::Fixed32 tc{};
    math::Cordic::sincos(math::Fixed32::fromFloat(tilt), ts, tc);

    SunState sun;
    sun.elevation = s.toFloat();
    sun.y = sun.elevation;
    sun.x = c.toFloat() * tc.toFloat();
    sun.z = c.toFloat() * ts.toFloat();

    core::f32 strength = sun.elevation * 4.0f;
    if (strength > 1.0f)
        strength = 1.0f;
    if (strength < 0.0f)
        strength = 0.0f;
    sun.intensity = strength;
    return sun;
}

inline core::u32 skyColour(core::f32 dirX, core::f32 dirY, core::f32 dirZ, const SunState &sun,
                           const SkyParams &params) noexcept
{
    const auto clamp01 = [](core::f32 v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };

    const core::f32 up = clamp01(dirY);
    const core::f32 path = 1.0f - up;
    const core::f32 pathSquared = path * path;

    const core::f32 lowSun = clamp01(1.0f - sun.elevation * 3.0f);

    core::f32 horizonR = params.horizonR + (params.duskR - params.horizonR) * lowSun;
    core::f32 horizonG = params.horizonG + (params.duskG - params.horizonG) * lowSun;
    core::f32 horizonB = params.horizonB + (params.duskB - params.horizonB) * lowSun;

    core::f32 r = params.zenithR + (horizonR - params.zenithR) * pathSquared;
    core::f32 g = params.zenithG + (horizonG - params.zenithG) * pathSquared;
    core::f32 b = params.zenithB + (horizonB - params.zenithB) * pathSquared;

    const core::f32 alignment = clamp01(dirX * sun.x + dirY * sun.y + dirZ * sun.z);
    core::f32 lobe = alignment;
    for (core::u32 i = 1u; i < static_cast<core::u32>(params.mieSharpness); ++i)
        lobe *= alignment;
    const core::f32 halo = lobe * params.mieStrength;
    r += halo * (0.9f + 0.1f * lowSun);
    g += halo * (0.7f - 0.2f * lowSun);
    b += halo * (0.5f - 0.3f * lowSun);

    if (alignment > params.sunSize)
    {
        r += 1.4f;
        g += 1.2f;
        b += 0.8f;
    }

    const core::f32 lit = params.nightFloor + (1.0f - params.nightFloor) * sun.intensity;
    r *= lit;
    g *= lit;
    b *= lit;

    if (dirY < 0.0f)
    {
        const core::f32 down = clamp01(-dirY * 4.0f);
        r = r + (params.groundR * lit - r) * down;
        g = g + (params.groundG * lit - g) * down;
        b = b + (params.groundB * lit - b) * down;
    }

    const auto channel = [&clamp01](core::f32 v) { return static_cast<core::u32>(clamp01(v) * 255.0f); };
    return (channel(r) << 16) | (channel(g) << 8) | channel(b);
}

inline core::u32 applyAerialPerspective(core::u32 colour, core::u32 skyTint, core::f32 distance,
                                        core::f32 density) noexcept
{
    const core::f32 t = distance * density;
    core::f32 blend = (t * t) / (1.0f + t * t);
    if (blend > 1.0f)
        blend = 1.0f;

    const core::u32 scale = static_cast<core::u32>(blend * 256.0f);
    const auto mix = [scale](core::u32 from, core::u32 to) { return ((from * (256u - scale)) + (to * scale)) >> 8; };
    const core::u32 r = mix((colour >> 16) & 0xFFu, (skyTint >> 16) & 0xFFu);
    const core::u32 g = mix((colour >> 8) & 0xFFu, (skyTint >> 8) & 0xFFu);
    const core::u32 b = mix(colour & 0xFFu, skyTint & 0xFFu);
    return (r << 16) | (g << 8) | b;
}

template <typename Sampler>
inline core::f32 terrainShadow(Sampler &&sampleHeight, core::i32 worldX, core::i32 worldZ, core::f32 height,
                               const SunState &sun, core::u32 steps) noexcept
{
    if (sun.elevation < 0.05f)
        return 0.0f;

    const core::f32 horizontal = sun.x * sun.x + sun.z * sun.z;
    if (horizontal < 0.0001f)
        return 0.0f;
    core::f32 inverse = 1.0f;
    for (core::u32 i = 0u; i < 3u; ++i)
        inverse = inverse * (1.5f - 0.5f * horizontal * inverse * inverse);
    const core::f32 stepX = sun.x * inverse;
    const core::f32 stepZ = sun.z * inverse;
    const core::f32 climb = sun.elevation * inverse;

    core::f32 occlusion = 0.0f;
    for (core::u32 i = 1u; i <= steps; ++i)
    {
        const core::f32 distance = static_cast<core::f32>(i);
        const core::i32 sampleX = worldX + static_cast<core::i32>(stepX * distance);
        const core::i32 sampleZ = worldZ + static_cast<core::i32>(stepZ * distance);
        const core::f32 rayHeight = height + climb * distance;
        const core::f32 terrain = sampleHeight(sampleX, sampleZ);
        if (terrain > rayHeight)
        {
            const core::f32 excess = (terrain - rayHeight) * 0.5f;
            occlusion = excess > 1.0f ? 1.0f : excess;
            break;
        }
    }
    return occlusion;
}

} // namespace lpl::render

#endif // LPL_RENDER_SKY_INL
