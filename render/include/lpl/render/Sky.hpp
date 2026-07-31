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
    core::f32 zenithR{0.20f}; ///< Zenith colour, the blue Rayleigh leaves behind.
    core::f32 zenithG{0.40f};
    core::f32 zenithB{0.85f};

    core::f32 horizonR{0.70f}; ///< Horizon colour at midday: a long path, so pale.
    core::f32 horizonG{0.80f};
    core::f32 horizonB{0.95f};

    core::f32 duskR{1.00f}; ///< What a low sun turns the horizon into.
    core::f32 duskG{0.45f};
    core::f32 duskB{0.18f};

    core::f32 groundR{0.16f}; ///< Below the horizon: haze, not sky.
    core::f32 groundG{0.17f};
    core::f32 groundB{0.20f};

    core::f32 sunSize{0.9985f};   ///< cos of the solar disc's angular radius.
    core::f32 mieStrength{0.55f}; ///< How bright the halo around the sun is.
    core::f32 mieSharpness{8.0f}; ///< How tightly it hugs the sun.
    core::f32 nightFloor{0.05f};  ///< Ambient that survives after sunset.
};

/**
 * @struct SunState
 * @brief Where the sun is, and how strong. Drives sky, light and shadow alike.
 */
struct SunState {
    core::f32 x{0.38f}; ///< Unit direction TOWARD the sun.
    core::f32 y{0.86f};
    core::f32 z{0.34f};
    core::f32 elevation{1.0f}; ///< Sine of the altitude angle; negative at night.
    core::f32 intensity{1.0f}; ///< 0 at night, 1 at noon; the light's strength.
};

/**
 * @brief Places the sun for a time of day.
 *
 * @param dayFraction Time in [0, 1): 0 is midnight, 0.25 sunrise, 0.5 noon.
 * @param tilt        Azimuth of the sun's arc, in radians; the season, roughly.
 * @return Direction, elevation and intensity.
 */
[[nodiscard]] inline SunState sunAt(core::f32 dayFraction, core::f32 tilt = 0.4f) noexcept
{
    // CORDIC, not libm: this is compiled into a freestanding kernel where sinf
    // does not exist, and the same call is what the camera basis already uses.
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

    // Intensity fades through the horizon rather than switching off: the last
    // minutes of light are what make a sunset read as one.
    core::f32 strength = sun.elevation * 4.0f;
    if (strength > 1.0f)
        strength = 1.0f;
    if (strength < 0.0f)
        strength = 0.0f;
    sun.intensity = strength;
    return sun;
}

/**
 * @brief Colour of the sky along one view direction.
 *
 * @param dirX View direction, normalised.
 * @param dirY Its vertical component; negative looks below the horizon.
 * @param dirZ
 * @param sun    Where the sun is.
 * @param params The dials.
 * @return Packed 0x00RRGGBB.
 */
[[nodiscard]] inline core::u32 skyColour(core::f32 dirX, core::f32 dirY, core::f32 dirZ, const SunState &sun,
                                         const SkyParams &params) noexcept
{
    const auto clamp01 = [](core::f32 v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };

    // Height in the sky, and the path length that follows from it. Looking up is
    // a short path through the atmosphere; looking at the horizon is a long one,
    // which is the whole reason the two ends differ.
    const core::f32 up = clamp01(dirY);
    const core::f32 path = 1.0f - up;
    const core::f32 pathSquared = path * path;

    // How low the sun is decides how red the long path comes out.
    const core::f32 lowSun = clamp01(1.0f - sun.elevation * 3.0f);

    core::f32 horizonR = params.horizonR + (params.duskR - params.horizonR) * lowSun;
    core::f32 horizonG = params.horizonG + (params.duskG - params.horizonG) * lowSun;
    core::f32 horizonB = params.horizonB + (params.duskB - params.horizonB) * lowSun;

    core::f32 r = params.zenithR + (horizonR - params.zenithR) * pathSquared;
    core::f32 g = params.zenithG + (horizonG - params.zenithG) * pathSquared;
    core::f32 b = params.zenithB + (horizonB - params.zenithB) * pathSquared;

    // Mie: the halo. Forward scattering, so it only shows near the sun's own
    // direction, and it reddens with the same path the sky does.
    const core::f32 alignment = clamp01(dirX * sun.x + dirY * sun.y + dirZ * sun.z);
    core::f32 lobe = alignment;
    for (core::u32 i = 1u; i < static_cast<core::u32>(params.mieSharpness); ++i)
        lobe *= alignment;
    const core::f32 halo = lobe * params.mieStrength;
    r += halo * (0.9f + 0.1f * lowSun);
    g += halo * (0.7f - 0.2f * lowSun);
    b += halo * (0.5f - 0.3f * lowSun);

    // The disc itself. A hard edge on purpose: a smooth blob reads as a lens
    // flare, and this is meant to be the sun.
    if (alignment > params.sunSize)
    {
        r += 1.4f;
        g += 1.2f;
        b += 0.8f;
    }

    // Night: the model above goes black, and a black sky with no stars is a hole
    // rather than a night. A floor keeps the shapes readable.
    const core::f32 lit = params.nightFloor + (1.0f - params.nightFloor) * sun.intensity;
    r *= lit;
    g *= lit;
    b *= lit;

    if (dirY < 0.0f)
    {
        // Below the horizon: the haze the terrain fades into. Blended rather than
        // switched, so a ridge on the skyline dissolves instead of being cut out.
        const core::f32 down = clamp01(-dirY * 4.0f);
        r = r + (params.groundR * lit - r) * down;
        g = g + (params.groundG * lit - g) * down;
        b = b + (params.groundB * lit - b) * down;
    }

    const auto channel = [&clamp01](core::f32 v) { return static_cast<core::u32>(clamp01(v) * 255.0f); };
    return (channel(r) << 16) | (channel(g) << 8) | channel(b);
}

/**
 * @brief Blends a surface colour toward the sky, by distance.
 *
 * Aerial perspective, and the cheapest large improvement a scene like this can
 * get: distance reads as haze, the far edge of a streamed world dissolves instead
 * of ending against a wall, and the seams a level of detail leaves behind stop
 * being the most visible thing on screen.
 *
 * @param colour   Packed surface colour.
 * @param skyTint  Packed sky colour along the same direction.
 * @param distance Distance to the surface, in world units.
 * @param density  Fog density; larger means a shorter view.
 * @return The blended colour.
 */
[[nodiscard]] inline core::u32 applyAerialPerspective(core::u32 colour, core::u32 skyTint, core::f32 distance,
                                                      core::f32 density) noexcept
{
    // Exponential-squared falloff, evaluated as a rational approximation: no expf
    // in a freestanding build, and the shape only has to be monotonic and smooth.
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
                                             core::f32 height, const SunState &sun, core::u32 steps) noexcept
{
    // A sun at or below the horizon casts no shadow — everything is already in
    // night, and marching would divide by a vanishing vertical component.
    if (sun.elevation < 0.05f)
        return 0.0f;

    // Step along the sun's horizontal direction; the ray climbs by the tangent of
    // the solar altitude, which is elevation over the horizontal length.
    const core::f32 horizontal = sun.x * sun.x + sun.z * sun.z;
    if (horizontal < 0.0001f)
        return 0.0f; // sun overhead: nothing can occlude it
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
            // How far above the ray decides how hard the shadow is: a ridge that
            // barely clips the sun gives a soft edge, which is what a penumbra
            // looks like without computing one.
            const core::f32 excess = (terrain - rayHeight) * 0.5f;
            occlusion = excess > 1.0f ? 1.0f : excess;
            break;
        }
    }
    return occlusion;
}

} // namespace lpl::render

#endif // LPL_RENDER_SKY_HPP
