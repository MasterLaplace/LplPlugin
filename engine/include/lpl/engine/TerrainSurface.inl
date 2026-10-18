#ifndef LPL_ENGINE_TERRAIN_SURFACE_INL
#define LPL_ENGINE_TERRAIN_SURFACE_INL

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
        const core::u32 sampled =
            grain.sample(wrapQ16(worldX * _params.grainTiles), wrapQ16(worldZ * _params.grainTiles),
                         render::MipTexture::levelForFootprint(distance * 0.06f));
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
    const core::f32 offAxis = (dx * inverse) * _lamp.headingX + (dz * inverse) * _lamp.headingZ;

    const core::f32 band = _lamp.coneInner - _lamp.coneOuter;
    core::f32 cone = band > 0.0001f ? (offAxis - _lamp.coneOuter) / band : 1.0f;
    cone = cone < 0.0f ? 0.0f : (cone > 1.0f ? 1.0f : cone);
    cone *= cone;

    core::f32 reach = _lamp.reach > 0.0001f ? 1.0f - distance / _lamp.reach : 0.0f;
    reach = reach < 0.0f ? 0.0f : reach;

    const core::f32 beam = cone * reach;
    if (beam <= 0.0f)
        return colour;

    return render::mixColours(render::modulate(colour, 1.0f + beam * 2.5f), _lamp.tint, beam * 0.25f);
}

inline core::u32 TerrainSurface::shadeSurface(core::f32 worldX, core::f32 worldZ, core::u32 base, core::f32 lit,
                                              core::f32 nx, core::f32 nz, core::f32 occlusion,
                                              const render::CameraBasis &basis) const noexcept
{
    const core::f32 distance = render::approximateLength(worldX - basis.eye.x, worldZ - basis.eye.z);
    const bool rocky = (nx * nx + nz * nz) > 1.2f;
    core::u32 colour;
    if (_physicallyBased)
        colour = shadePhysical(worldX, worldZ, base, nx, nz, occlusion, distance, rocky, basis);
    else if (_perPixel)
        colour = shade(worldX, worldZ, base, lit, distance, rocky);
    else
        colour = render::applyAerialPerspective(render::modulate(base, lit), _haze, distance, fogDensity());
    return applyLamp(colour, worldX, worldZ);
}

inline core::f32 effectiveCurvature(const TerrainSurfaceParams &params, core::f32 eyeY) noexcept
{
    const core::f32 altitude = eyeY - params.seaLevel;
    return params.curvatureFactor + (altitude > 0.0f ? altitude : 0.0f) * params.curvatureAltitudeGain;
}

inline void TerrainSurface::applyLook(const render::SkyParams &sky, const render::WaterParams &water,
                                      core::f32 dayFraction) noexcept
{
    _skyParams = sky;
    const core::f32 phase = _water.phase;
    _water = water;
    _water.phase = phase;
    _dayFraction = dayFraction;
    _sun = render::sunAt(_dayFraction);
    _irradianceElevation = -1.0e9f;
}

template <typename BedDepthAt>
inline core::u32 TerrainSurface::drawWaterWith(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                               const render::CameraBasis &basis, const core::f32 *quad,
                                               const render::WaterParams &water, core::f32 latticePitch,
                                               BedDepthAt &&bedDepthAt) const
{
    const auto shade = [&](core::f32 wx, core::f32 wy, core::f32 wz) {
        core::u32 colour = render::waterColour(wx, wy, wz, basis.eye, _sun, _skyParams, water, bedDepthAt(wx, wz));
        core::u32 reflected = 0u;
        if (_reflection && render::sampleProbe(_probe, _probePixels, wx, wy, wz, reflected))
            colour = render::mixColours(colour, reflected, 0.45f);
        return render::applyAerialPerspective(
            colour, _haze, render::approximateLength(wx - basis.eye.x, wz - basis.eye.z), _params.fogDensity);
    };

    const core::u32 divisions = (water.swellHeight > 0.0f && _waterTessellation > 1u) ? _waterTessellation : 1u;
    if (divisions == 1u)
        return render::fillPolygonShadedClipped(rt, mvp, quad, 4u, shade);

    const core::f32 y = quad[1];
    if (quad[4] != y || quad[7] != y || quad[10] != y)
        return render::fillPolygonShadedClipped(rt, mvp, quad, 4u, shade);

    const core::f32 x0 = quad[0];
    const core::f32 z0 = quad[2];
    const core::f32 x1 = quad[6];
    const core::f32 z1 = quad[8];
    if (latticePitch <= 0.0f || x1 <= x0 || z1 <= z0)
        return render::fillPolygonShadedClipped(rt, mvp, quad, 4u, shade);

    const core::f32 fade = 1.0f / (2.0f * water.swellHeight);
    const auto surfaceY = [&](core::f32 wx, core::f32 wz) {
        const core::f32 depth = bedDepthAt(wx, wz);
        core::f32 damp = depth * fade;
        damp = damp < 0.0f ? 0.0f : (damp > 1.0f ? 1.0f : damp);
        return y + render::waterHeight(wx, wz, water) * damp;
    };

    const auto latticeFloor = [latticePitch](core::f32 v) {
        const core::f32 scaled = v / latticePitch;
        const core::f32 truncated = static_cast<core::f32>(static_cast<core::i32>(scaled));
        return (scaled < truncated ? truncated - 1.0f : truncated) * latticePitch;
    };

    core::u32 triangles = 0u;
    for (core::f32 az = latticeFloor(z0); az < z1; az += latticePitch)
    {
        const core::f32 lowZ = az < z0 ? z0 : az;
        const core::f32 highZ = az + latticePitch > z1 ? z1 : az + latticePitch;
        if (highZ <= lowZ)
            continue;
        for (core::f32 ax = latticeFloor(x0); ax < x1; ax += latticePitch)
        {
            const core::f32 lowX = ax < x0 ? x0 : ax;
            const core::f32 highX = ax + latticePitch > x1 ? x1 : ax + latticePitch;
            if (highX <= lowX)
                continue;
            const core::f32 cell[12] = {lowX,  surfaceY(lowX, lowZ),   lowZ,  highX, surfaceY(highX, lowZ), lowZ,
                                        highX, surfaceY(highX, highZ), highZ, lowX,  surfaceY(lowX, highZ), highZ};
            triangles += render::fillPolygonShadedClipped(rt, mvp, cell, 4u, shade);
        }
    }
    return triangles;
}

template <typename DrawWorld>
inline void TerrainSurface::refreshProbe(core::u32 frame, const render::CameraBasis &basis, bool waterInSight,
                                         DrawWorld &&drawWorld)
{
    if (!_reflection || _probeColour == nullptr || _probeDepth == nullptr || !waterInSight)
    {
        _probe.valid = false;
        return;
    }
    if ((frame & 3u) != 0u && _probe.valid)
        return;

    const render::CameraBasis mirrored = render::mirrorBasisAboutPlane(basis, _params.seaLevel);
    const core::f32 aspect = static_cast<core::f32>(_probe.width) / static_cast<core::f32>(_probe.height);
    _probe.planeY = _params.seaLevel;
    _probe.mirrorMvp = render::mirrorViewProjection(mirrored, aspect, render::CameraLens{1.04719755f, 0.4f, 600.0f});

    const render::RenderTarget target{_probeColour, _probeDepth, _probe.width, _probe.height};
    render::drawSky(target, mirrored, _sun, _skyParams, 0.5773502692f, 3u);
    drawWorld(target, _probe.mirrorMvp, mirrored);

    _probe.valid = render::copyTargetToTexture(target, _probePixels);
}

template <typename HeightAt, typename ForEachProp>
inline void TerrainSurface::fillShadowMask(const render::HeightfieldPatchParams &patch, HeightAt &&heightAt,
                                           ForEachProp &&forEachProp, procgen::Grid<core::u8> &mask) const
{
    if (!_shadows || mask.empty())
        return;

    for (core::u32 z = 0u; z < patch.size; ++z)
        for (core::u32 x = 0u; x < patch.size; ++x)
        {
            const core::i32 worldX = patch.originX + static_cast<core::i32>(x);
            const core::i32 worldZ = patch.originZ + static_cast<core::i32>(z);
            const core::f32 occlusion =
                render::terrainShadow([&heightAt](core::i32 sx, core::i32 sz) { return heightAt(sx, sz); }, worldX,
                                      worldZ, heightAt(worldX, worldZ), _sun, _params.shadowSteps);
            mask.at(x, z) = static_cast<core::u8>(occlusion * 255.0f);
        }

    if (_sun.y <= 0.12f)
        return;

    const core::f32 reach = 1.0f / _sun.y;
    forEachProp([&](core::i32 cellX, core::i32 cellZ, core::f32 height, core::f32 spread) {
        const core::u32 steps = static_cast<core::u32>(height * reach);
        for (core::u32 step = 0u; step <= steps && step < _params.shadowSteps; ++step)
        {
            const core::f32 t = static_cast<core::f32>(step);
            const core::i32 shadowX = cellX - static_cast<core::i32>(_sun.x * t * reach);
            const core::i32 shadowZ = cellZ - static_cast<core::i32>(_sun.z * t * reach);
            const core::i32 localX = shadowX - patch.originX;
            const core::i32 localZ = shadowZ - patch.originZ;
            if (localX < 0 || localZ < 0 || localX >= static_cast<core::i32>(patch.size) ||
                localZ >= static_cast<core::i32>(patch.size))
                continue;

            const core::f32 along = steps == 0u ? 0.0f : t / static_cast<core::f32>(steps);
            const core::f32 strength = (0.85f - 0.45f * along) * (spread > 0.6f ? 1.0f : 0.7f);
            core::u8 &cell = mask.at(static_cast<core::u32>(localX), static_cast<core::u32>(localZ));
            const core::u32 added = static_cast<core::u32>(cell) + static_cast<core::u32>(strength * 255.0f);
            cell = static_cast<core::u8>(added > 255u ? 255u : added);
        }
    });
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_SURFACE_INL
