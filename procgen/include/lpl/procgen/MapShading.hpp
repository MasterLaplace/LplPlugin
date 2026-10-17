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
    Biome = 0, /**< The classification itself — the only non-scalar mode. */
    Height,
    Moisture,
    Drainage, /**< Flow accumulation, logarithmically compressed. */
    Region,   /**< Province id, hashed to a colour. */
    Slope,
    Climate, /**< One of the six axes, chosen by @c axis. */
    Count
};

/**
 * @brief Names a shading mode.
 * @return A stable lowercase name, for a HUD line or a combo box.
 */
[[nodiscard]] inline const char *mapShadingName(MapShading shading) noexcept;

/**
 * @brief Names a climate axis.
 * @return The name of climate axis @p axis, in @ref ClimateField order.
 */
[[nodiscard]] inline const char *climateAxisName(core::u32 axis) noexcept;

/**
 * @brief The biome palette, chosen so a glance reads as a map, not a debug ramp.
 *
 * A magenta answer means an id outside the enumeration reached here, which is a
 * bug worth seeing rather than a colour worth picking.
 */
[[nodiscard]] inline Rgb biomeColour(BiomeId biome) noexcept;

/**
 * @brief Amber-on-abyss ramp for the scalar views, so they read as instrumentation.
 */
[[nodiscard]] inline Rgb rampColour(float t) noexcept;

/**
 * @brief Hashes a province id so neighbouring regions never share a shade.
 */
[[nodiscard]] inline Rgb regionColour(core::u16 region) noexcept;

/**
 * @brief Where a flow accumulation sits on a logarithmic ramp, in [0, 1].
 *
 * Accumulation spans four orders of magnitude across one map — a ridge cell
 * drains itself, the river mouth drains half the world — so a linear ramp shows
 * only the trunk. @ref fixedLog2 is the module's own compression for exactly this
 * quantity; the base cancels in the ratio, so this is the wetness index's
 * logarithm and not an approximation of a different one.
 */
[[nodiscard]] inline float drainageRamp(core::u32 flow, core::u32 maxAccumulation) noexcept;

/**
 * @brief The colour of one surface cell under @p shading.
 *
 * A mode whose grid the recipe never asked for answers grey rather than guessing:
 * "this pass did not run" and "this pass produced zero" must not look alike, or
 * the view stops being an instrument.
 *
 * Every climate axis is normalised to [0, 1] by the climate pass, so one ramp
 * reads all six — and the fact that it does is itself the claim being
 * looked at: an axis that came out flat or saturated is a bug you can see
 * here and nowhere in a signature.
 *
 * @param atlas  The world to read.
 * @param shading Which quantity to show.
 * @param axis   Which climate axis, when @p shading is @c MapShading::Climate.
 * @param x      Cell X.
 * @param z      Cell Z.
 */
[[nodiscard]] inline Rgb surfaceColour(const WorldAtlas &atlas, MapShading shading, core::u32 axis, core::u32 x,
                                       core::u32 z) noexcept;

} // namespace lpl::procgen

#    include "MapShading.inl"

#endif // LPL_PROCGEN_MAP_SHADING_HPP
