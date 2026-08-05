/**
 * @file MapShading.hpp
 * @brief How a generated world is coloured when the point is to inspect it.
 *
 * Not a world's palette — that belongs to the document (`pack::ViewV1`) and is
 * what a *game* looks like. This is the instrument: seven ways of colouring the
 * same cell so that a pass which came out flat, saturated, empty or upside down is
 * visible at a glance rather than absent from a signature. The two are genuinely
 * different things, and unifying them would be a mistake rather than a cleanup:
 * one lights a 3D heightfield, the other reads as a flat map, and the same biome
 * legitimately wants a different base colour under those two treatments.
 *
 * It lives here, next to the grids it reads, because the vocabulary is entirely
 * this module's: @ref BiomeId, @ref DrainageNetwork, @ref VoronoiDiagram,
 * @ref ClimateField. It was written twice before it was written once — the map
 * viewer had all of it in an anonymous namespace, and the editor was about to
 * grow a second set.
 *
 * Two rules this header keeps, which the copy it replaces did not:
 *  - **No libm.** The drainage ramp needs a logarithm and this module already has
 *    one, @ref fixedLog2, whose own documentation is about flow accumulation
 *    spanning four orders of magnitude. The app-side copy called `std::log`
 *    instead, which is both a second answer and a function no code in this module
 *    is allowed to reach for. The ratio is unaffected: a change of logarithm base
 *    cancels in @f$\log(1+f)/\log(1+m)@f$.
 *  - **Float is presentation only.** Every value here is a colour or a normalised
 *    ramp coordinate, never state. Nothing in this file may feed an authoritative
 *    path, which is why nothing in it returns Fixed32.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_MAP_SHADING_HPP
#    define LPL_PROCGEN_MAP_SHADING_HPP

#    include <lpl/math/FixedMath.hpp>
#    include <lpl/procgen/ValueNoise.hpp>
#    include <lpl/procgen/WorldAtlas.hpp>

namespace lpl::procgen {

/** @brief A colour in the viewer's own units: three channels, 0 to 1. */
struct Rgb {
    float r;
    float g;
    float b;
};

/**
 * @enum MapShading
 * @brief Which quantity colours a surface cell.
 *
 * @c Climate is one mode rather than six, and the axis it shows is a separate
 * option: the six axes are the same kind of quantity, and cycling shading through
 * six near-identical entries would bury the five modes that are not.
 */
enum class MapShading : int {
    Biome = 0, ///< The classification itself — the only non-scalar mode.
    Height,
    Moisture,
    Drainage, ///< Flow accumulation, logarithmically compressed.
    Region,   ///< Province id, hashed to a colour.
    Slope,
    Climate, ///< One of the six axes, chosen by @c axis.
    Count
};

/// @return A stable lowercase name, for a HUD line or a combo box.
[[nodiscard]] inline const char *mapShadingName(MapShading shading) noexcept
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

/// @return The name of climate axis @p axis, in @ref ClimateField order.
[[nodiscard]] inline const char *climateAxisName(core::u32 axis) noexcept
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

/**
 * @brief The biome palette, chosen so a glance reads as a map, not a debug ramp.
 *
 * A magenta answer means an id outside the enumeration reached here, which is a
 * bug worth seeing rather than a colour worth picking.
 */
[[nodiscard]] inline Rgb biomeColour(BiomeId biome) noexcept
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

/** @brief Amber-on-abyss ramp for the scalar views, so they read as instrumentation. */
[[nodiscard]] inline Rgb rampColour(float t) noexcept
{
    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;
    return {0.05f + 0.95f * t, 0.05f + 0.62f * t * t, 0.10f + 0.15f * t * t * t};
}

/** @brief Hashes a province id so neighbouring regions never share a shade. */
[[nodiscard]] inline Rgb regionColour(core::u16 region) noexcept
{
    const core::u32 h = ValueNoise2D::hash2(static_cast<core::i32>(region), 7, 0x9E37u);
    return {0.25f + 0.7f * static_cast<float>((h >> 0) & 0xFFu) / 255.0f,
            0.25f + 0.7f * static_cast<float>((h >> 8) & 0xFFu) / 255.0f,
            0.25f + 0.7f * static_cast<float>((h >> 16) & 0xFFu) / 255.0f};
}

/**
 * @brief Where a flow accumulation sits on a logarithmic ramp, in [0, 1].
 *
 * Accumulation spans four orders of magnitude across one map — a ridge cell
 * drains itself, the river mouth drains half the world — so a linear ramp shows
 * only the trunk. @ref fixedLog2 is the module's own compression for exactly this
 * quantity; the base cancels in the ratio, so this is the wetness index's
 * logarithm and not an approximation of a different one.
 */
[[nodiscard]] inline float drainageRamp(core::u32 flow, core::u32 maxAccumulation) noexcept
{
    if (maxAccumulation == 0u)
        return 0.0f;
    const float span = math::fixedLog2(maxAccumulation + 1u).toFloat();
    if (span <= 0.0f)
        return 0.0f;
    return math::fixedLog2(flow + 1u).toFloat() / span;
}

/**
 * @brief The colour of one surface cell under @p shading.
 *
 * A mode whose grid the recipe never asked for answers grey rather than guessing:
 * "this pass did not run" and "this pass produced zero" must not look alike, or
 * the view stops being an instrument.
 *
 * @param atlas  The world to read.
 * @param shading Which quantity to show.
 * @param axis   Which climate axis, when @p shading is @c MapShading::Climate.
 * @param x      Cell X.
 * @param z      Cell Z.
 */
[[nodiscard]] inline Rgb surfaceColour(const WorldAtlas &atlas, MapShading shading, core::u32 axis, core::u32 x,
                                       core::u32 z) noexcept
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
        // Every axis is normalised to [0, 1] by the climate pass, so one ramp
        // reads all six — and the fact that it does is itself the claim being
        // looked at: an axis that came out flat or saturated is a bug you can see
        // here and nowhere in a signature.
        return rampColour(atlas.climate.axes[axis % kClimateAxisCount].at(x, z).toFloat());
    case MapShading::Biome:
    case MapShading::Count: break;
    }
    return biomeColour(atlas.biomes.at(x, z));
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_MAP_SHADING_HPP
