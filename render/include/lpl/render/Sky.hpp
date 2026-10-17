/**
 * @file Sky.hpp
 * @brief A sky computed rather than loaded, and the sun that lights the world.
 *
 * No cube map, no gradient texture, no asset of any kind: the colour of a pixel
 * of sky is a function of the direction it looks at and where the sun is. That is
 * not a stylistic preference here — a freestanding kernel has no filesystem to
 * load a sky from, so a sky that cannot be computed is a sky that cannot exist.
 *
 * The model is a deliberately cheap reading of atmospheric scattering, and it is
 * worth being precise about which parts are physics and which are taste:
 *
 *  - **Rayleigh** — short wavelengths scatter more, so a long path through the
 *    atmosphere loses its blue and leaves red. That is why the zenith is blue and
 *    the horizon is pale, and why the whole sky turns amber when the sun sits low.
 *    Modelled here as a path-length term, not an integral.
 *  - **Mie** — the bright halo hugging the sun, from scattering off large
 *    particles. A forward-scattering lobe, approximated by a power of the cosine
 *    between the view ray and the sun.
 *  - **The ground half** is not sky at all: below the horizon the same function
 *    returns a haze the terrain fades into, which is what makes a distant ridge
 *    dissolve instead of ending against a hard blue wall.
 *
 * The same @ref SunState drives the sky, the directional light and the shadows.
 * One source for all three is the point: a sky painted at dawn over a world lit
 * from noon is the single most obvious way to make a scene look assembled rather
 * than observed.
 *
 * @warning Everything here is float and non-authoritative. It reads the clock and
 *          produces pixels; no value computed in this file ever flows back into
 *          the simulation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_SKY_HPP
#    define LPL_RENDER_SKY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::render {

/**
 * @struct SkyParams
 * @brief The dials of the model. Defaults are a clear day at sea level.
 */
struct SkyParams {
    core::f32 zenithR{0.20f}; /**< Zenith colour, the blue Rayleigh leaves behind. */
    core::f32 zenithG{0.40f};
    core::f32 zenithB{0.85f};

    core::f32 horizonR{0.70f}; /**< Horizon colour at midday: a long path, so pale. */
    core::f32 horizonG{0.80f};
    core::f32 horizonB{0.95f};

    core::f32 duskR{1.00f}; /**< What a low sun turns the horizon into. */
    core::f32 duskG{0.45f};
    core::f32 duskB{0.18f};

    core::f32 groundR{0.16f}; /**< Below the horizon: haze, not sky. */
    core::f32 groundG{0.17f};
    core::f32 groundB{0.20f};

    core::f32 sunSize{0.9985f};   /**< cos of the solar disc's angular radius. */
    core::f32 mieStrength{0.55f}; /**< How bright the halo around the sun is. */
    core::f32 mieSharpness{8.0f}; /**< How tightly it hugs the sun. */
    core::f32 nightFloor{0.05f};  /**< Ambient that survives after sunset. */
};

/**
 * @struct SunState
 * @brief Where the sun is, and how strong. Drives sky, light and shadow alike.
 */
struct SunState {
    core::f32 x{0.38f}; /**< Unit direction TOWARD the sun. */
    core::f32 y{0.86f};
    core::f32 z{0.34f};
    core::f32 elevation{1.0f}; /**< Sine of the altitude angle; negative at night. */
    core::f32 intensity{1.0f}; /**< 0 at night, 1 at noon; the light's strength. */
};

/**
 * @brief Places the sun for a time of day.
 *
 * CORDIC, not libm: this is compiled into a freestanding kernel where sinf
 * does not exist, and the same call is what the camera basis already uses.
 * Intensity fades through the horizon rather than switching off: the last
 * minutes of light are what make a sunset read as one.
 *
 * @param dayFraction Time in [0, 1): 0 is midnight, 0.25 sunrise, 0.5 noon.
 * @param tilt        Azimuth of the sun's arc, in radians; the season, roughly.
 * @return Direction, elevation and intensity.
 */
[[nodiscard]] inline SunState sunAt(core::f32 dayFraction, core::f32 tilt = 0.4f) noexcept;

/**
 * @brief Colour of the sky along one view direction.
 *
 * Looking up is a short path through the atmosphere; looking at the horizon is
 * a long one, which is the whole reason the two ends differ. How low the sun is
 * decides how red the long path comes out. The Mie halo is forward scattering,
 * so it only shows near the sun's own direction, and it reddens with the same
 * path the sky does. The disc itself has a hard edge on purpose: a smooth blob
 * reads as a lens flare, and this is meant to be the sun.
 *
 * At night the model goes black, and a black sky with no stars is a hole rather
 * than a night: a floor keeps the shapes readable. Below the horizon, the haze
 * the terrain fades into is blended rather than switched, so a ridge on the
 * skyline dissolves instead of being cut out.
 *
 * @param dirX View direction, normalised.
 * @param dirY Its vertical component; negative looks below the horizon.
 * @param dirZ
 * @param sun    Where the sun is.
 * @param params The dials.
 * @return Packed 0x00RRGGBB.
 */
[[nodiscard]] inline core::u32 skyColour(core::f32 dirX, core::f32 dirY, core::f32 dirZ, const SunState &sun,
                                         const SkyParams &params) noexcept;

/**
 * @brief Blends a surface colour toward the sky, by distance.
 *
 * Aerial perspective, and the cheapest large improvement a scene like this can
 * get: distance reads as haze, the far edge of a streamed world dissolves instead
 * of ending against a wall, and the seams a level of detail leaves behind stop
 * being the most visible thing on screen.
 *
 * The falloff is exponential-squared, evaluated as the rational approximation
 * t^2 / (1 + t^2) with t = distance * density: no expf in a freestanding build,
 * and the shape only has to be monotonic and smooth.
 *
 * @param colour   Packed surface colour.
 * @param skyTint  Packed sky colour along the same direction.
 * @param distance Distance to the surface, in world units.
 * @param density  Fog density; larger means a shorter view.
 * @return The blended colour.
 */
[[nodiscard]] inline core::u32 applyAerialPerspective(core::u32 colour, core::u32 skyTint, core::f32 distance,
                                                      core::f32 density) noexcept;

/**
 * @brief Marches a ray toward the sun over a height field and reports occlusion.
 *
 * Horizon mapping, and for a height field it is both the cheapest correct shadow
 * and the only one that needs no second buffer: walk from the cell toward the
 * sun, and if the terrain ever rises above the ray, the cell is in shadow. No
 * shadow map, no second pass over the scene, no resolution to pick — the terrain
 * IS the occluder and it is already in memory.
 *
 * @warning It shadows the terrain against ITSELF only. A tree does not cast one,
 *          because a tree is not in this height field. Saying that plainly is
 *          better than a reader assuming the scene has general shadows and
 *          wondering why a forest sits on flat light.
 *
 * A sun at or below the horizon casts no shadow — everything is already in
 * night, and marching would divide by a vanishing vertical component — and
 * neither does a sun overhead: nothing can occlude it. The march steps along the
 * sun's horizontal direction; the ray climbs by the tangent of the solar
 * altitude, which is elevation over the horizontal length. How far the terrain
 * rises above the ray decides how hard the shadow is: a ridge that barely clips
 * the sun gives a soft edge, which is what a penumbra looks like without
 * computing one.
 *
 * @param sampleHeight Callable: (worldX, worldZ) -> elevation.
 * @param worldX  Cell to test.
 * @param worldZ
 * @param height  Its own elevation.
 * @param sun     Where the sun is.
 * @param steps   Cells to march; the shadow's maximum length.
 * @return 0 fully lit, up to 1 fully shadowed.
 */
template <typename Sampler>
[[nodiscard]] inline core::f32 terrainShadow(Sampler &&sampleHeight, core::i32 worldX, core::i32 worldZ,
                                             core::f32 height, const SunState &sun, core::u32 steps) noexcept;

} // namespace lpl::render

#    include "Sky.inl"

#endif // LPL_RENDER_SKY_HPP
