/**
 * @file Water.hpp
 * @brief A water surface shaded per pixel: swell, foam, scatter, reflection, glint.
 *
 * Water is the one surface a flat colour cannot fake, and the reason is not
 * artistic: what you see in water depends on the direction from your eye to THAT
 * POINT, so a per-primitive colour is answering a question that was never asked.
 * A lake drawn as one flat blue quad reads as painted cardboard from every angle,
 * and cutting it into a thousand quads does not help — each one is still flat.
 *
 * With a per-pixel shader the mirror is exact and costs no extra pass: reflect the
 * view ray about the (rippled) surface normal and ask the sky what is in that
 * direction. There is no render-to-texture, no second camera, and nothing to keep
 * in step — the reflection is the sky function this file already shares with the
 * dome overhead, so water and sky cannot disagree about the weather.
 *
 * What makes a sea read as a SEA rather than as a rippling mirror is three things
 * this file used not to have, and each of them is a separate observation:
 *
 *  - a swell has SCALE. One ripple frequency is a puddle at any size; a sea is a
 *    long swell with chop riding on it and fine texture riding on that.
 *  - a crest is SHARP and a trough is BROAD. That asymmetry is what a Gerstner wave
 *    buys over a sine, and it is the difference between water and corrugated iron.
 *  - the white is where the water BREAKS: along a shore, and on the crests. Foam is
 *    not a decoration on top of water, it is the only part of a sea whose colour
 *    does not come from what is behind or above it.
 *
 * All three are computed from ONE evaluation, @ref sampleWater — deliberately, and
 * this is the part worth keeping: height, slope and crest-ness come out of the same
 * octave sum, so the foam lands on the crest the geometry actually has. Three
 * separate functions would be three chances to disagree about where a crest is, and
 * foam floating beside a wave is more obviously wrong than no foam at all.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_WATER_HPP
#    define LPL_RENDER_WATER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/Sky.hpp>

namespace lpl::render {

/**
 * @struct WaterParams
 * @brief What the surface is made of, and how it moves.
 */
struct WaterParams {
    core::u32 shallow{0x00246E8Cu}; ///< Body colour where the bed is near.
    core::u32 deep{0x000C1E3Cu};    ///< Body colour where it is not.
    core::f32 rippleScale{0.85f};   ///< Spatial frequency of the ripples.
    core::f32 rippleAmplitude{0.16f};
    core::f32 phase{0.0f};       ///< Advances with time: the ripples travel.
    core::f32 glintPower{48.0f}; ///< Tightness of the sun's reflection.
    core::f32 depthScale{0.22f}; ///< How fast the body colour goes to @c deep.

    /**
     * @brief Which way the swell travels, and the second crest that crosses it.
     *
     * These were CONSTANTS in the ripple function, which is why every body of water in the
     * world had its waves running the same way regardless of the weather or the ground: a
     * lake and a river flowing north to south rippled identically, and both ignored a wind
     * the world already models for its rain shadow.
     *
     * The defaults are the values that were hardcoded, to the digit, so a caller that says
     * nothing gets exactly the surface it got before. Use @ref setDrift to point them.
     */
    core::f32 driftX{1.0f};
    core::f32 driftZ{0.4f};

    /**
     * @brief How fast the surface travels along its drift, as a multiple of the base rate.
     *
     * The other half of @ref setDrift. One is the still-water rate the phase already
     * advances at; a river's current raises it, and a wind-only surface — a lake — sits
     * near it. Kept apart from the direction because a magnitude folded into the direction
     * stretches the WAVELENGTH instead of raising the SPEED, which is the opposite of what
     * "this water moves faster" should look like.
     */
    core::f32 flowSpeed{1.0f};
    core::f32 crossX{-0.7f};
    core::f32 crossZ{1.0f};

    // ── The swell ────────────────────────────────────────────────────────────
    //
    // What separates a sea from a rippling mirror. Every field below defaults to
    // OFF or to the value that reproduces the two-crossed-ripples surface this file
    // had, so a caller that says nothing is unaffected — the sea is something a
    // world asks for, not something that happens to it.

    /**
     * @brief Peak height of the swell in world units, and the switch for all of it.
     *
     * Zero means no swell: the octave sum collapses to the two crossed ripples that
     * were here before, and @ref waterHeight returns a flat surface. That is the
     * default because a lake in a courtyard should not heave.
     */
    core::f32 swellHeight{0.0f};

    /**
     * @brief How much sharper a crest is than a trough, in [0, 1].
     *
     * Zero is a triangle wave: symmetric, and it reads as corrugated iron. One
     * squares the wave's rising profile, which broadens the troughs and narrows the
     * peaks — the visible half of what a Gerstner wave does. The other half is
     * horizontal bunching towards the crest, which needs the surface to be a mesh
     * rather than a plane; @ref waterHeight is what a displaced mesh samples.
     */
    core::f32 crestSharpness{0.0f};

    /**
     * @brief Weight of the two octaves above the swell.
     *
     * The frequency ratios themselves are constants in @ref sampleWater and not
     * knobs: they are the shape of a wind-sea spectrum, not a per-world choice, and
     * a caller free to set them is a caller free to make two octaves beat against
     * each other at a visible period.
     */
    core::f32 chopStrength{1.0f};

    // ── Foam and scatter ─────────────────────────────────────────────────────

    core::u32 foam{0x00E8F4F8u};    ///< Broken water: barely blue, never pure white.
    core::u32 scatter{0x0048D0B0u}; ///< Sunlight coming THROUGH a crest.

    /**
     * @brief How white broken water gets, and the switch for ALL foam.
     *
     * Zero — the default — means no surf and no whitecaps. One switch rather than one
     * per kind, because "does this world have foam" is one decision and two switches
     * is two things to forget: a caller that sets a shore width and leaves the gain at
     * zero has asked for foam and been given none, with nothing to say why.
     */
    core::f32 foamGain{0.0f};

    /**
     * @brief Bed depth over which shoreline surf fades out, in world units.
     *
     * The single most legible cue that a body of water has a bottom: a hard waterline
     * reads as a texture boundary, a band of white reads as a beach. Driven by the
     * same bed depth the body colour uses, so foam cannot appear where the water is
     * deep. Zero keeps whitecaps and drops the shore band.
     */
    core::f32 foamWidth{1.6f};

    /**
     * @brief Normalised swell height above which a crest breaks, in (0, 1).
     *
     * Not a fraction of the foam, a fraction of the WAVE: only the top slice of the
     * swell whitens. Push it to zero and the whole sea is milk.
     */
    core::f32 foamCrest{0.78f};

    /**
     * @brief Strength of the light that comes through a wave rather than off it.
     *
     * Zero disables it. A crest between the eye and the sun is thin enough to pass
     * light, which is why the top of a wave glows green while its face stays dark —
     * and it is the cue that says "this is water and it is deep" rather than "this
     * is a shiny surface". Cheap here: crest height times how much the view faces
     * the sun, tinted, added.
     */
    core::f32 scatterStrength{0.0f};

    /**
     * @brief Points the swell along a direction, and derives the crossing crest from it.
     *
     * One call rather than four fields, because the relationship between the two crests is
     * the part that matters and the part a caller would get wrong: they must NOT be
     * perpendicular. Two waves at right angles interfere into a checkerboard, which reads as
     * a tiled floor — the original comment said so, and setting the pair by hand is how that
     * property gets lost.
     *
     * The angle between them is the one the hardcoded pair had, 103 degrees, applied as a
     * fixed rotation. Cosine and sine of a constant are constants, so this needs no libm and
     * is safe on the kernel side.
     *
     * Two callers are intended and they answer different questions: open water takes the
     * PREVAILING WIND, which the world already has as `MoistureParams::windDirection`; a
     * river takes its own DOWNSLOPE, because a current does not care which way the air goes.
     *
     * @param dx Direction along x. Need not be normalised.
     * @param dz Direction along z.
     */
    void setDrift(core::f32 dx, core::f32 dz) noexcept
    {
        constexpr core::f32 kCos103 = -0.22495f;
        constexpr core::f32 kSin103 = 0.97437f;
        // ⚠ NORMALISED, because the drift enters the wave as `(x·dir) * frequency` — so its
        // magnitude silently multiplies the SPATIAL FREQUENCY. Feed it a velocity, as the
        // river current does, and slow water does not ripple slowly: its wavelength is
        // stretched by the same factor and the surface goes flat. Measured on screen as
        // "between two flows there is no animation at all, just cyan".
        //
        // A direction says WHERE, a magnitude says HOW FAST, and they belong in different
        // terms. See @ref flowSpeed for the other half.
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
};

/**
 * @struct WaveSample
 * @brief One evaluation of the surface: how high, which way it tilts, how broken.
 */
struct WaveSample {
    core::f32 height{0.0f}; ///< Normalised, in [-1, 1]. Multiply by @c swellHeight.
    core::f32 slopeX{0.0f}; ///< Normalised surface gradient along x.
    core::f32 slopeZ{0.0f}; ///< Along z.
    core::f32 crest{0.0f};  ///< 0 away from a crest, 1 at a breaking one.
};

/**
 * @brief The swell, its gradient and its crests, in one pass and without a sine.
 *
 * A triangle wave folded and then skewed is close enough to a swell that nobody
 * looking at water can tell, and it needs no libm — which on the kernel side is not
 * a preference. Three octaves along three directions, none of them perpendicular to
 * another: two waves at right angles interfere into a checkerboard, which reads as a
 * tiled floor.
 *
 * The gradient is the DERIVATIVE of the same sum rather than a second guess at it.
 * The previous version used the wave's value as the normal's tilt, which is off by a
 * quarter period — harmless while nothing else read the height, and wrong the moment
 * foam had to land on a crest.
 *
 * @param worldX World x.
 * @param worldZ World z.
 * @param params The surface.
 * @return Height, slope and crest-ness, all normalised.
 */
[[nodiscard]] inline WaveSample sampleWater(core::f32 worldX, core::f32 worldZ, const WaterParams &params) noexcept
{
    // Triangle wave of period 2 in v, and its derivative. Both are needed at the
    // same v, so they are computed together rather than by differencing.
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

    // Broadens the troughs and narrows the peaks. u is the wave mapped to [0, 1];
    // squaring u pushes mass towards the trough, so the crest arrives late and
    // leaves early. d/dt of the blend comes out as a single multiply-add.
    const core::f32 sharp =
        params.crestSharpness < 0.0f ? 0.0f : (params.crestSharpness > 1.0f ? 1.0f : params.crestSharpness);
    const auto skew = [sharp](core::f32 t, core::f32 &ioSlope) {
        const core::f32 u = (t + 1.0f) * 0.5f;
        const core::f32 squared = 2.0f * u * u - 1.0f;
        ioSlope *= 1.0f + sharp * (2.0f * u - 1.0f);
        return t + sharp * (squared - t);
    };

    // The spectrum. Ratios are fixed: a long swell, chop at a little over twice its
    // frequency running across it, and fine texture at nearly seven times, back along
    // the swell. Each travels at its own speed — a sea whose octaves drift together
    // reads as one wave with a decorated surface.
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
        // ⚠ MINUS the phase, not plus. A crest sits where v is constant, so as the phase
        // rises the projection x·dir must FALL to keep it there — with a plus, every crest
        // travels along −dir and the water runs backwards up its own slope. The field is
        // called `drift` and every caller points it where the water goes; the arithmetic
        // was doing the opposite, which is exactly what a river flowing uphill looks like.
        const core::f32 v = (worldX * octave.dirX + worldZ * octave.dirZ) * octave.frequency -
                            params.phase * octave.speed * params.flowSpeed;
        core::f32 slope = 0.0f;
        const core::f32 value = skew(foldWave(v, slope), slope);
        sample.height += octave.amplitude * value;
        sample.slopeX += octave.amplitude * slope * octave.dirX * octave.frequency;
        sample.slopeZ += octave.amplitude * slope * octave.dirZ * octave.frequency;
        amplitudeSum += octave.amplitude;
        // Normalising the gradient by its own worst case rather than by the height's
        // keeps a fine octave from tilting the normal past vertical: a short wave of
        // small amplitude still has a steep face.
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

/**
 * @brief Vertical displacement of the surface at a point, in world units.
 *
 * What a displaced water mesh samples. A plane with a normal map is a mirror that
 * ripples; the horizon of a sea has to actually move, and that is geometry.
 *
 * @param worldX World x.
 * @param worldZ World z.
 * @param params The surface.
 * @return Offset to add to the still-water level.
 */
[[nodiscard]] inline core::f32 waterHeight(core::f32 worldX, core::f32 worldZ, const WaterParams &params) noexcept
{
    if (params.swellHeight <= 0.0f)
        return 0.0f;
    return sampleWater(worldX, worldZ, params).height * params.swellHeight;
}

/**
 * @brief The surface's tilt at a point, in the convention the shader wants.
 *
 * Kept as its own entry point because it is what every existing caller asks for, and
 * because "which way does it tilt" is a smaller question than @ref sampleWater's.
 */
inline void waterNormal(core::f32 worldX, core::f32 worldZ, const WaterParams &params, core::f32 &outNx,
                        core::f32 &outNz) noexcept
{
    const WaveSample sample = sampleWater(worldX, worldZ, params);
    // Downhill is the direction the normal leans, so the gradient enters negated.
    outNx = -sample.slopeX * params.rippleAmplitude;
    outNz = -sample.slopeZ * params.rippleAmplitude;
}

/**
 * @brief Colour of one point on the water surface, seen from @p eye.
 *
 * @param bedDepth How far the bed is below the surface here, in world units.
 *                 Zero at the shoreline, which is what makes a beach fade instead
 *                 of ending on a hard line.
 */
[[nodiscard]] inline core::u32 waterColour(core::f32 worldX, core::f32 worldY, core::f32 worldZ,
                                           const math::Vec3<core::f32> &eye, const SunState &sun,
                                           const SkyParams &skyParams, const WaterParams &params,
                                           core::f32 bedDepth) noexcept
{
    const WaveSample wave = sampleWater(worldX, worldZ, params);
    core::f32 nx = -wave.slopeX * params.rippleAmplitude;
    core::f32 nz = -wave.slopeZ * params.rippleAmplitude;

    // View ray, from the eye to this point.
    core::f32 vx = worldX - eye.x;
    core::f32 vy = worldY - eye.y;
    core::f32 vz = worldZ - eye.z;
    const core::f32 inverse = inverseSqrtNewton(vx * vx + vy * vy + vz * vz);
    vx *= inverse;
    vy *= inverse;
    vz *= inverse;

    // Reflect about the rippled normal, normalised the same cheap way.
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

    // Body colour: deeper water keeps less of what is under it.
    core::f32 depthMix = bedDepth * params.depthScale;
    depthMix = depthMix < 0.0f ? 0.0f : (depthMix > 1.0f ? 1.0f : depthMix);
    core::u32 body = mixColours(params.shallow, params.deep, depthMix);

    // Light through the wave rather than off it. A crest between the eye and the sun
    // is thin enough to pass light, which is why the top of a wave glows and its face
    // stays dark. Added to the BODY, not to the final colour: it is light coming out
    // of the water, so the mirror still wins at a grazing angle.
    if (params.scatterStrength > 0.0f && wave.crest > 0.0f)
    {
        const core::f32 towardsSun = -(vx * sun.x + vz * sun.z);
        if (towardsSun > 0.0f)
            body = mixColours(body, params.scatter, wave.crest * towardsSun * params.scatterStrength);
    }

    // Fresnel by Schlick: at a grazing angle water is a mirror, from overhead it
    // is a window. Getting this backwards is what makes CG water look like blue
    // plastic — the reflection has to arrive with the angle, not uniformly.
    const core::f32 cosine = dot < 0.0f ? -dot : dot;
    const core::f32 oneMinus = 1.0f - cosine;
    const core::f32 squared = oneMinus * oneMinus;
    core::f32 fresnel = 0.02f + 0.98f * squared * squared * oneMinus;
    fresnel = fresnel < 0.0f ? 0.0f : (fresnel > 1.0f ? 1.0f : fresnel);

    core::u32 colour = mixColours(body, mirrored, fresnel);

    // The sun's own reflection: a tight lobe about the mirror direction, which is
    // the highlight that tells an eye the surface is wet and moving.
    const core::f32 alignment = rx * sun.x + ry * sun.y + rz * sun.z;
    if (alignment > 0.0f)
    {
        const core::f32 glint = detail::intPow(alignment, static_cast<core::u32>(params.glintPower)) * sun.intensity;
        if (glint > 0.004f)
            colour = modulate(colour, 1.0f + 5.0f * glint);
    }

    // Foam LAST, and that ordering is the point: broken water is opaque, so it has no
    // reflection, no Fresnel and no glint. Mixing it in earlier would let the sky shine
    // through the surf.
    if (params.foamGain > 0.0f)
    {
        core::f32 foam = 0.0f;
        if (params.foamWidth > 0.0f)
        {
            // Surf along the shore, where the bed comes up to meet the surface. Modulated by
            // the swell so the line breathes instead of sitting there as a painted ring.
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

#endif // LPL_RENDER_WATER_HPP
