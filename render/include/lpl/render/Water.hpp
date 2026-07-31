/**
 * @file Water.hpp
 * @brief A water surface shaded per pixel: reflection, Fresnel, sun glint.
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
    core::f32 phase{0.0f};          ///< Advances with time: the ripples travel.
    core::f32 glintPower{48.0f};    ///< Tightness of the sun's reflection.
    core::f32 depthScale{0.22f};    ///< How fast the body colour goes to @c deep.
};

/**
 * @brief Two crossed sine ripples, without a sine.
 *
 * A triangle wave folded twice is close enough to a swell that nobody looking at
 * water can tell, and it needs no libm — which on the kernel side is not a
 * preference. The two directions are deliberately not perpendicular: two waves at
 * right angles interfere into a checkerboard, which reads as a tiled floor.
 */
inline void waterNormal(core::f32 worldX, core::f32 worldZ, const WaterParams &params, core::f32 &outNx,
                                      core::f32 &outNz) noexcept
{
    const auto fold = [](core::f32 v) {
        // v mod 2, mapped to a triangle in [-1, 1].
        const core::f32 wrapped = v - 2.0f * static_cast<core::f32>(static_cast<core::i32>(v * 0.5f));
        const core::f32 t = wrapped < 0.0f ? wrapped + 2.0f : wrapped;
        return t < 1.0f ? (2.0f * t - 1.0f) : (3.0f - 2.0f * t);
    };

    const core::f32 a = fold((worldX + worldZ * 0.4f) * params.rippleScale + params.phase);
    const core::f32 b = fold((worldZ - worldX * 0.7f) * params.rippleScale * 1.7f - params.phase * 1.3f);
    outNx = (a * 0.7f + b * 0.3f) * params.rippleAmplitude;
    outNz = (b * 0.7f - a * 0.3f) * params.rippleAmplitude;
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
    core::f32 nx = 0.0f;
    core::f32 nz = 0.0f;
    waterNormal(worldX, worldZ, params, nx, nz);

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
    const core::u32 body = mixColours(params.shallow, params.deep, depthMix);

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
    return colour;
}

} // namespace lpl::render

#endif // LPL_RENDER_WATER_HPP
