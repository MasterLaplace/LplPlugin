#ifndef LPL_ENGINE_VIEW_PROFILE_INL
#define LPL_ENGINE_VIEW_PROFILE_INL

namespace lpl::engine {

inline ViewProfile toEngineView(const pack::ViewV1 &wire) noexcept
{
    ViewProfile out{};

    out.sky.zenithR = wire.zenithR;
    out.sky.zenithG = wire.zenithG;
    out.sky.zenithB = wire.zenithB;
    out.sky.horizonR = wire.horizonR;
    out.sky.horizonG = wire.horizonG;
    out.sky.horizonB = wire.horizonB;
    out.sky.duskR = wire.duskR;
    out.sky.duskG = wire.duskG;
    out.sky.duskB = wire.duskB;
    out.sky.groundR = wire.groundR;
    out.sky.groundG = wire.groundG;
    out.sky.groundB = wire.groundB;
    out.sky.sunSize = wire.sunSize;
    out.sky.mieStrength = wire.mieStrength;
    out.sky.mieSharpness = wire.mieSharpness;
    out.sky.nightFloor = wire.nightFloor;

    out.dayFraction = wire.dayFraction;

    out.surface.seaLevel = wire.seaLevel;
    out.surface.fogDensity = wire.fogDensity;
    out.surface.ambient = wire.ambient;
    out.surface.grainTiles = wire.grainTiles;
    out.surface.shadowSteps = wire.shadowSteps;
    out.surface.curvatureFactor = wire.curvatureFactor;
    out.surface.curvatureAltitudeGain = wire.curvatureAltitudeGain;

    out.water.shallow = wire.waterShallow;
    out.water.deep = wire.waterDeep;
    out.water.rippleScale = wire.rippleScale;
    out.water.rippleAmplitude = wire.rippleAmplitude;
    out.water.glintPower = wire.glintPower;
    out.water.depthScale = wire.depthScale;
    out.water.swellHeight = wire.swellHeight;
    out.water.crestSharpness = wire.crestSharpness;
    out.water.chopStrength = wire.chopStrength;
    out.water.foam = wire.foamColour;
    out.water.scatter = wire.scatterColour;
    out.water.foamGain = wire.foamGain;
    out.water.foamWidth = wire.foamWidth;
    out.water.foamCrest = wire.foamCrest;
    out.water.scatterStrength = wire.scatterStrength;

    out.grazerTint = wire.grazerTint;
    out.hunterTint = wire.hunterTint;
    out.bodyScale = wire.bodyScale;

    if ((wire.flags & pack::kViewFlagOverridePalette) != 0u)
    {
        const core::u32 count =
            wire.biomeColourCount > pack::kWireBiomeColours ? pack::kWireBiomeColours : wire.biomeColourCount;
        for (core::u32 i = 0u; i < count; ++i)
            out.palette[i] = wire.biomeColour[i];
        out.paletteCount = count;
    }
    return out;
}

inline pack::ViewV1 toWireView(const ViewProfile &profile) noexcept
{
    pack::ViewV1 wire{};

    wire.zenithR = profile.sky.zenithR;
    wire.zenithG = profile.sky.zenithG;
    wire.zenithB = profile.sky.zenithB;
    wire.horizonR = profile.sky.horizonR;
    wire.horizonG = profile.sky.horizonG;
    wire.horizonB = profile.sky.horizonB;
    wire.duskR = profile.sky.duskR;
    wire.duskG = profile.sky.duskG;
    wire.duskB = profile.sky.duskB;
    wire.groundR = profile.sky.groundR;
    wire.groundG = profile.sky.groundG;
    wire.groundB = profile.sky.groundB;
    wire.sunSize = profile.sky.sunSize;
    wire.mieStrength = profile.sky.mieStrength;
    wire.mieSharpness = profile.sky.mieSharpness;
    wire.nightFloor = profile.sky.nightFloor;

    wire.dayFraction = profile.dayFraction;

    wire.seaLevel = profile.surface.seaLevel;
    wire.fogDensity = profile.surface.fogDensity;
    wire.ambient = profile.surface.ambient;
    wire.grainTiles = profile.surface.grainTiles;
    wire.shadowSteps = profile.surface.shadowSteps;
    wire.curvatureFactor = profile.surface.curvatureFactor;
    wire.curvatureAltitudeGain = profile.surface.curvatureAltitudeGain;

    wire.waterShallow = profile.water.shallow;
    wire.waterDeep = profile.water.deep;
    wire.rippleScale = profile.water.rippleScale;
    wire.rippleAmplitude = profile.water.rippleAmplitude;
    wire.glintPower = profile.water.glintPower;
    wire.depthScale = profile.water.depthScale;
    wire.swellHeight = profile.water.swellHeight;
    wire.crestSharpness = profile.water.crestSharpness;
    wire.chopStrength = profile.water.chopStrength;
    wire.foamColour = profile.water.foam;
    wire.scatterColour = profile.water.scatter;
    wire.foamGain = profile.water.foamGain;
    wire.foamWidth = profile.water.foamWidth;
    wire.foamCrest = profile.water.foamCrest;
    wire.scatterStrength = profile.water.scatterStrength;

    wire.grazerTint = profile.grazerTint;
    wire.hunterTint = profile.hunterTint;
    wire.bodyScale = profile.bodyScale;

    if (profile.paletteCount != 0u)
    {
        const core::u32 count =
            profile.paletteCount > pack::kWireBiomeColours ? pack::kWireBiomeColours : profile.paletteCount;
        for (core::u32 i = 0u; i < count; ++i)
            wire.biomeColour[i] = profile.palette[i];
        wire.biomeColourCount = count;
        wire.flags |= pack::kViewFlagOverridePalette;
    }
    return wire;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_VIEW_PROFILE_INL
