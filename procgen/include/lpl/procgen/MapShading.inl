#ifndef LPL_PROCGEN_MAP_SHADING_INL
#define LPL_PROCGEN_MAP_SHADING_INL

namespace lpl::procgen {

inline const char *mapShadingName(MapShading shading) noexcept
{
    switch (shading)
    {
    case MapShading::Biome: return "biome";
    case MapShading::Height: return "height";
    case MapShading::Moisture: return "moisture";
    case MapShading::Drainage: return "drainage";
    case MapShading::Region: return "region";
    case MapShading::Slope: return "slope";
    case MapShading::Climate: return "climate";
    case MapShading::Count: break;
    }
    return "?";
}

inline const char *climateAxisName(core::u32 axis) noexcept
{
    switch (axis % kClimateAxisCount)
    {
    case 0u: return "temperature";
    case 1u: return "moisture";
    case 2u: return "continentalness";
    case 3u: return "erosion";
    case 4u: return "depth";
    default: return "weirdness";
    }
}

inline Rgb biomeColour(BiomeId biome) noexcept
{
    switch (biome)
    {
    case BiomeId::Ocean: return {0.07f, 0.20f, 0.42f};
    case BiomeId::Beach: return {0.83f, 0.77f, 0.55f};
    case BiomeId::Snow: return {0.94f, 0.95f, 0.97f};
    case BiomeId::Tundra: return {0.60f, 0.62f, 0.56f};
    case BiomeId::Taiga: return {0.20f, 0.38f, 0.31f};
    case BiomeId::Rock: return {0.44f, 0.42f, 0.40f};
    case BiomeId::Desert: return {0.85f, 0.72f, 0.42f};
    case BiomeId::Savanna: return {0.70f, 0.68f, 0.34f};
    case BiomeId::Grassland: return {0.42f, 0.60f, 0.30f};
    case BiomeId::Forest: return {0.20f, 0.45f, 0.22f};
    case BiomeId::Rainforest: return {0.11f, 0.36f, 0.18f};
    case BiomeId::Marsh: return {0.31f, 0.42f, 0.33f};
    case BiomeId::Lake: return {0.16f, 0.35f, 0.55f};
    case BiomeId::Count: break;
    }
    return {1.0f, 0.0f, 1.0f};
}

inline Rgb rampColour(float t) noexcept
{
    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;
    return {0.05f + 0.95f * t, 0.05f + 0.62f * t * t, 0.10f + 0.15f * t * t * t};
}

inline Rgb regionColour(core::u16 region) noexcept
{
    const core::u32 h = ValueNoise2D::hash2(static_cast<core::i32>(region), 7, 0x9E37u);
    return {0.25f + 0.7f * static_cast<float>((h >> 0) & 0xFFu) / 255.0f,
            0.25f + 0.7f * static_cast<float>((h >> 8) & 0xFFu) / 255.0f,
            0.25f + 0.7f * static_cast<float>((h >> 16) & 0xFFu) / 255.0f};
}

inline float drainageRamp(core::u32 flow, core::u32 maxAccumulation) noexcept
{
    if (maxAccumulation == 0u)
        return 0.0f;
    const float span = math::fixedLog2(maxAccumulation + 1u).toFloat();
    if (span <= 0.0f)
        return 0.0f;
    return math::fixedLog2(flow + 1u).toFloat() / span;
}

inline Rgb surfaceColour(const WorldAtlas &atlas, MapShading shading, core::u32 axis, core::u32 x, core::u32 z) noexcept
{
    switch (shading)
    {
    case MapShading::Height: {
        const float span = (atlas.highest - atlas.lowest).toFloat();
        return rampColour(span > 0.0f ? (atlas.height.at(x, z) - atlas.lowest).toFloat() / span : 0.5f);
    }
    case MapShading::Moisture:
        return atlas.moisture.empty() ? Rgb{0.5f, 0.5f, 0.5f} : rampColour(atlas.moisture.at(x, z).toFloat());
    case MapShading::Drainage:
        if (atlas.drainage.maxAccumulation == 0u)
            return {0.1f, 0.1f, 0.1f};
        return rampColour(drainageRamp(atlas.drainage.accumulation.at(x, z), atlas.drainage.maxAccumulation));
    case MapShading::Region:
        if (atlas.regions.regions.empty())
            return {0.2f, 0.2f, 0.2f};
        return regionColour(atlas.regions.regions.at(x, z));
    case MapShading::Slope: return rampColour(slopeAt(atlas.height, x, z).toFloat() * 0.6f);
    case MapShading::Climate:
        if (atlas.climate.empty())
            return {0.2f, 0.2f, 0.2f};
        return rampColour(atlas.climate.axes[axis % kClimateAxisCount].at(x, z).toFloat());
    case MapShading::Biome:
    case MapShading::Count: break;
    }
    return biomeColour(atlas.biomes.at(x, z));
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_MAP_SHADING_INL
