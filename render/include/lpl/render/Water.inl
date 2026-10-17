#ifndef LPL_RENDER_WATER_INL
#define LPL_RENDER_WATER_INL

namespace lpl::render {

inline void WaterParams::setDrift(core::f32 dx, core::f32 dz) noexcept
{
    constexpr core::f32 kCos103 = -0.22495f;
    constexpr core::f32 kSin103 = 0.97437f;
    const core::f32 length = dx * dx + dz * dz;
    if (length > 1.0e-8f)
    {
        const core::f32 inverse = inverseSqrtNewton(length);
        dx *= inverse;
        dz *= inverse;
    }
    else
    {
        dx = 1.0f;
        dz = 0.0f;
    }
    driftX = dx;
    driftZ = dz;
    crossX = dx * kCos103 - dz * kSin103;
    crossZ = dx * kSin103 + dz * kCos103;
}

inline WaveSample sampleWater(core::f32 worldX, core::f32 worldZ, const WaterParams &params) noexcept
{
    const auto foldWave = [](core::f32 v, core::f32 &outSlope) {
        const core::f32 wrapped = v - 2.0f * static_cast<core::f32>(static_cast<core::i32>(v * 0.5f));
        const core::f32 t = wrapped < 0.0f ? wrapped + 2.0f : wrapped;
        if (t < 1.0f)
        {
            outSlope = 2.0f;
            return 2.0f * t - 1.0f;
        }
        outSlope = -2.0f;
        return 3.0f - 2.0f * t;
    };

    const core::f32 sharp =
        params.crestSharpness < 0.0f ? 0.0f : (params.crestSharpness > 1.0f ? 1.0f : params.crestSharpness);
    const auto skew = [sharp](core::f32 t, core::f32 &ioSlope) {
        const core::f32 u = (t + 1.0f) * 0.5f;
        const core::f32 squared = 2.0f * u * u - 1.0f;
        ioSlope *= 1.0f + sharp * (2.0f * u - 1.0f);
        return t + sharp * (squared - t);
    };

    struct Octave {
        core::f32 dirX;
        core::f32 dirZ;
        core::f32 frequency;
        core::f32 amplitude;
        core::f32 speed;
    };
    const core::f32 chop = params.chopStrength < 0.0f ? 0.0f : params.chopStrength;
    const Octave octaves[3] = {
        {params.driftX,                               params.driftZ,                               params.rippleScale,        1.0f,         1.0f },
        {params.crossX,                               params.crossZ,                               params.rippleScale * 2.3f, 0.42f * chop, -1.3f},
        {params.driftX * 0.6f + params.crossX * 0.8f, params.driftZ * 0.6f + params.crossZ * 0.8f,
         params.rippleScale * 6.7f,                                                                                           0.15f * chop, 2.1f },
    };

    WaveSample sample;
    core::f32 amplitudeSum = 0.0f;
    core::f32 slopeSum = 0.0f;
    for (const Octave &octave : octaves)
    {
        if (octave.amplitude <= 0.0f)
            continue;
        const core::f32 v = (worldX * octave.dirX + worldZ * octave.dirZ) * octave.frequency -
                            params.phase * octave.speed * params.flowSpeed;
        core::f32 slope = 0.0f;
        const core::f32 value = skew(foldWave(v, slope), slope);
        sample.height += octave.amplitude * value;
        sample.slopeX += octave.amplitude * slope * octave.dirX * octave.frequency;
        sample.slopeZ += octave.amplitude * slope * octave.dirZ * octave.frequency;
        amplitudeSum += octave.amplitude;
        slopeSum += octave.amplitude * 2.0f * octave.frequency;
    }

    if (amplitudeSum > 0.0f)
        sample.height /= amplitudeSum;
    if (slopeSum > 0.0f)
    {
        sample.slopeX /= slopeSum;
        sample.slopeZ /= slopeSum;
    }

    const core::f32 threshold = params.foamCrest < 0.0f ? 0.0f : (params.foamCrest > 0.99f ? 0.99f : params.foamCrest);
    const core::f32 over = sample.height - threshold;
    sample.crest = over <= 0.0f ? 0.0f : over / (1.0f - threshold);
    return sample;
}

inline core::f32 waterHeight(core::f32 worldX, core::f32 worldZ, const WaterParams &params) noexcept
{
    if (params.swellHeight <= 0.0f)
        return 0.0f;
    return sampleWater(worldX, worldZ, params).height * params.swellHeight;
}

inline void waterNormal(core::f32 worldX, core::f32 worldZ, const WaterParams &params, core::f32 &outNx,
                        core::f32 &outNz) noexcept
{
    const WaveSample sample = sampleWater(worldX, worldZ, params);
    outNx = -sample.slopeX * params.rippleAmplitude;
    outNz = -sample.slopeZ * params.rippleAmplitude;
}

inline core::u32 waterColour(core::f32 worldX, core::f32 worldY, core::f32 worldZ, const math::Vec3<core::f32> &eye,
                             const SunState &sun, const SkyParams &skyParams, const WaterParams &params,
                             core::f32 bedDepth) noexcept
{
    const WaveSample wave = sampleWater(worldX, worldZ, params);
    core::f32 nx = -wave.slopeX * params.rippleAmplitude;
    core::f32 nz = -wave.slopeZ * params.rippleAmplitude;

    core::f32 vx = worldX - eye.x;
    core::f32 vy = worldY - eye.y;
    core::f32 vz = worldZ - eye.z;
    const core::f32 inverse = inverseSqrtNewton(vx * vx + vy * vy + vz * vz);
    vx *= inverse;
    vy *= inverse;
    vz *= inverse;

    core::f32 ny = 1.0f;
    const core::f32 normalInverse = inverseSqrtNewton(nx * nx + ny * ny + nz * nz);
    nx *= normalInverse;
    ny *= normalInverse;
    nz *= normalInverse;
    const core::f32 dot = vx * nx + vy * ny + vz * nz;
    const core::f32 rx = vx - 2.0f * dot * nx;
    const core::f32 ry = vy - 2.0f * dot * ny;
    const core::f32 rz = vz - 2.0f * dot * nz;

    const core::u32 mirrored = skyColour(rx, ry < 0.0f ? -ry : ry, rz, sun, skyParams);

    core::f32 depthMix = bedDepth * params.depthScale;
    depthMix = depthMix < 0.0f ? 0.0f : (depthMix > 1.0f ? 1.0f : depthMix);
    core::u32 body = mixColours(params.shallow, params.deep, depthMix);

    if (params.scatterStrength > 0.0f && wave.crest > 0.0f)
    {
        const core::f32 towardsSun = -(vx * sun.x + vz * sun.z);
        if (towardsSun > 0.0f)
            body = mixColours(body, params.scatter, wave.crest * towardsSun * params.scatterStrength);
    }

    const core::f32 cosine = dot < 0.0f ? -dot : dot;
    const core::f32 oneMinus = 1.0f - cosine;
    const core::f32 squared = oneMinus * oneMinus;
    core::f32 fresnel = 0.02f + 0.98f * squared * squared * oneMinus;
    fresnel = fresnel < 0.0f ? 0.0f : (fresnel > 1.0f ? 1.0f : fresnel);

    core::u32 colour = mixColours(body, mirrored, fresnel);

    const core::f32 alignment = rx * sun.x + ry * sun.y + rz * sun.z;
    if (alignment > 0.0f)
    {
        const core::f32 glint = detail::intPow(alignment, static_cast<core::u32>(params.glintPower)) * sun.intensity;
        if (glint > 0.004f)
            colour = modulate(colour, 1.0f + 5.0f * glint);
    }

    if (params.foamGain > 0.0f)
    {
        core::f32 foam = 0.0f;
        if (params.foamWidth > 0.0f)
        {
            const core::f32 shore = 1.0f - bedDepth / params.foamWidth;
            if (shore > 0.0f)
                foam = (shore > 1.0f ? 1.0f : shore) * (0.65f + 0.35f * wave.height);
        }
        if (wave.crest > foam)
            foam = wave.crest;
        if (foam > 0.0f)
        {
            const core::f32 amount = foam * params.foamGain;
            colour = mixColours(colour, params.foam, amount > 1.0f ? 1.0f : amount);
        }
    }
    return colour;
}

} // namespace lpl::render

#endif // LPL_RENDER_WATER_INL
