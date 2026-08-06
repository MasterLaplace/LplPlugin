/**
 * @file TerrainSurface.inl
 * @brief Out-of-line definitions for engine::TerrainSurface.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_TERRAIN_SURFACE_INL
#    define LPL_ENGINE_TERRAIN_SURFACE_INL

namespace lpl::engine {

inline void TerrainSurface::configure(const Config &config, const TerrainSurfaceParams &params, core::u32 seed)
{
    _params = params;
    _perPixel = config.enablePerPixelSurface() || config.enablePbrSurface();
    _physicallyBased = config.enablePbrSurface();
    _shadows = config.enableTerrainShadows();
    _shadowChunksPerTick = config.shadowChunksPerTick();
    _reflection = config.enableWaterReflection();
    _waterTessellation = config.waterTessellation();
    _skyBlock = config.skyBlockSize() == 0u ? 1u : config.skyBlockSize();

    // Two textures, not one: rock and vegetation want different grain, and one
    // stretched over both is what makes a procedural world look like a single
    // material painted different colours.
    _grassGrain =
        render::MipTexture{render::Texture::makeNoise(64u, 64u, seed ^ 0x6A55u, 0x00FFFFFFu, 0x00B4C8A0u, 10u)};
    _rockGrain = render::MipTexture{render::Texture::makeNoise(64u, 64u, seed ^ 0x0C1Fu, 0x00FFFFFFu, 0x00A0A0AAu, 6u)};
}

inline void TerrainSurface::attachProbe(core::u32 *colour, core::f32 *depth, core::u32 width, core::u32 height)
{
    _probeColour = colour;
    _probeDepth = depth;
    _probe.width = width;
    _probe.height = height;
    _probePixels = render::Texture{width, height};
}

inline void TerrainSurface::advance(core::f32 dayFraction, core::f32 ripplePhase) noexcept
{
    _dayFraction += dayFraction;
    if (_dayFraction >= 1.0f)
        _dayFraction -= 1.0f;
    _sun = render::sunAt(_dayFraction);
    _water.phase += ripplePhase;
    if (_water.phase > 512.0f)
        _water.phase -= 512.0f;
}

inline void TerrainSurface::beginFrame(const render::RenderTarget &rt, const render::CameraBasis &basis) noexcept
{
    render::drawSky(rt, basis, _sun, _skyParams, 0.5773502692f, _skyBlock);
    _haze = render::hazeTint(basis, _sun, _skyParams);
    _underground = false;

    // Re-integrate the sky only when the sun has moved enough to change it. A day
    // that advances by a thousandth per frame would otherwise pay for a projection
    // to get back a probe indistinguishable from the last one.
    if (_physicallyBased)
    {
        const core::f32 drift = _sun.elevation - _irradianceElevation;
        if (drift > 0.01f || drift < -0.01f)
        {
            _irradiance = render::projectSky(_sun, _skyParams);
            _irradianceElevation = _sun.elevation;
        }
    }
}

inline void TerrainSurface::beginCaveFrame(const render::RenderTarget &rt, core::u32 tint, core::f32 density) noexcept
{
    // Cleared rather than skipped. Leaving the last frame's sky in the buffer would
    // show it wherever nothing is drawn, and underground that is most of the screen.
    render::clearTarget(rt, tint);
    _haze = tint;
    _caveFog = density;
    _underground = true;
}

inline core::u32 TerrainSurface::shade(core::f32 worldX, core::f32 worldZ, core::u32 base, core::f32 lit,
                                       core::f32 distance, bool rocky) const noexcept
{
    const render::MipTexture &grain = rocky ? _rockGrain : _grassGrain;
    core::u32 tinted = base;
    if (!grain.empty())
    {
        const auto wrapQ16 = [](core::f32 v) {
            const core::f32 wrapped = v - static_cast<core::f32>(static_cast<core::i32>(v)) + 1.0f;
            const core::f32 unit = wrapped - static_cast<core::f32>(static_cast<core::i32>(wrapped));
            return static_cast<core::u32>(unit * 65535.0f);
        };
        // The level follows the distance: without the level moving, the grain
        // crawls whenever the camera does — which is what mips are FOR.
        const core::u32 sampled =
            grain.sample(wrapQ16(worldX * _params.grainTiles), wrapQ16(worldZ * _params.grainTiles),
                         render::MipTexture::levelForFootprint(distance * 0.06f));
        // The grain modulates, it does not replace: a texture that overrode the
        // base colour would throw away the only thing that says where you are.
        tinted = render::modulate(base, 0.72f + 0.55f * static_cast<core::f32>(sampled & 0xFFu) / 255.0f);
    }
    return render::applyAerialPerspective(render::modulate(tinted, lit), _haze, distance, fogDensity());
}

inline core::u32 TerrainSurface::shadePhysical(core::f32 worldX, core::f32 worldZ, core::u32 base, core::f32 nx,
                                               core::f32 nz, core::f32 occlusion, core::f32 distance, bool rocky,
                                               const render::CameraBasis &basis) const noexcept
{
    render::PbrMaterial material;
    material.albedo = render::Vec3f(static_cast<core::f32>((base >> 16) & 0xFFu) / 255.0f,
                                    static_cast<core::f32>((base >> 8) & 0xFFu) / 255.0f,
                                    static_cast<core::f32>(base & 0xFFu) / 255.0f);
    material.metallic = 0.0f;
    material.roughness = rocky ? 0.55f : 0.85f;
    material.ao = 1.0f - 0.6f * occlusion;

    render::Light sun;
    sun.type = render::LightType::Directional;
    sun.direction = render::Vec3f(-_sun.x, -_sun.y, -_sun.z);
    sun.color = render::Vec3f(1.0f, 0.96f, 0.89f);
    sun.intensity = 2.6f * _sun.intensity * (1.0f - occlusion);

    const core::f32 inverse = render::inverseSqrtNewton(nx * nx + 1.0f + nz * nz);
    const render::Vec3f normal(nx * inverse, inverse, nz * inverse);

    // The environment term carries the whole shaded side of the world, so it cannot
    // be a token value: at 0.35 the slopes facing away came out black, because ACES
    // maps a small radiance to a smaller one.
    //
    // It used to be the sky read STRAIGHT UP — one colour for every normal, which
    // made a slope facing the sunset and a slope facing away from it receive the
    // same ambient. Now it is the sky integrated over the hemisphere the normal
    // actually sees, so the two differ by exactly what the sky differs by.
    constexpr core::f32 kSkyIrradiance = 1.15f;
    render::Vec3f ambient = render::evaluateIrradiance(_irradiance, normal.x, normal.y, normal.z);
    ambient.x *= kSkyIrradiance;
    ambient.y *= kSkyIrradiance;
    ambient.z *= kSkyIrradiance;
    const render::Vec3f fragment(worldX, _params.seaLevel, worldZ);
    const core::u32 shaded =
        render::pbrShadeToRgb(material, &sun, 1u, normal, fragment, basis.eye, ambient, render::ToneMap::Aces);
    return render::applyAerialPerspective(shaded, _haze, distance, fogDensity());
}

inline core::u32 TerrainSurface::applyLamp(core::u32 colour, core::f32 worldX, core::f32 worldZ) const noexcept
{
    if (!_lamp.on)
        return colour;

    const core::f32 dx = worldX - _lamp.x;
    const core::f32 dz = worldZ - _lamp.z;
    const core::f32 distance = render::approximateLength(dx, dz);
    const core::f32 inverse = 1.0f / (distance + 0.001f);
    // A cosine straight from the dot product. No angle is ever formed, which is what
    // keeps a lamp off any transcendental on a target that has none.
    const core::f32 offAxis = (dx * inverse) * _lamp.headingX + (dz * inverse) * _lamp.headingZ;

    const core::f32 band = _lamp.coneInner - _lamp.coneOuter;
    core::f32 cone = band > 0.0001f ? (offAxis - _lamp.coneOuter) / band : 1.0f;
    cone = cone < 0.0f ? 0.0f : (cone > 1.0f ? 1.0f : cone);
    cone *= cone; // squared, so the edge of the beam softens instead of ending on a line

    core::f32 reach = _lamp.reach > 0.0001f ? 1.0f - distance / _lamp.reach : 0.0f;
    reach = reach < 0.0f ? 0.0f : reach;

    const core::f32 beam = cone * reach;
    if (beam <= 0.0f)
        return colour;

    // BRIGHTENS what is there rather than replacing it: a torch on grass is warm-lit
    // grass, and mixing straight to the lamp's own colour would paint every surface in
    // the world the same beige and throw away the only thing that says where you are.
    // The small blend toward the tint is the warmth; the multiply is the light.
    return render::mixColours(render::modulate(colour, 1.0f + beam * 2.5f), _lamp.tint, beam * 0.25f);
}

inline core::u32 TerrainSurface::shadeSurface(core::f32 worldX, core::f32 worldZ, core::u32 base, core::f32 lit,
                                              core::f32 nx, core::f32 nz, core::f32 occlusion,
                                              const render::CameraBasis &basis) const noexcept
{
    const core::f32 distance = render::approximateLength(worldX - basis.eye.x, worldZ - basis.eye.z);
    const bool rocky = (nx * nx + nz * nz) > 1.2f;
    // One funnel, so the lamp is applied ONCE rather than in each of the three shading
    // paths — three copies of a light are three chances for one of them to stay dark.
    core::u32 colour;
    if (_physicallyBased)
        colour = shadePhysical(worldX, worldZ, base, nx, nz, occlusion, distance, rocky, basis);
    else if (_perPixel)
        colour = shade(worldX, worldZ, base, lit, distance, rocky);
    else
        colour = render::applyAerialPerspective(render::modulate(base, lit), _haze, distance, fogDensity());
    return applyLamp(colour, worldX, worldZ);
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_SURFACE_INL
