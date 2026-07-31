/**
 * @file Irradiance.hpp
 * @brief Image-based lighting: the sky, integrated, as nine coefficients.
 *
 * What the physically based path was missing. Its environment term was the sky
 * evaluated STRAIGHT UP, one colour used for every normal — which is the honest
 * placeholder for image-based lighting when there is no image, and it is wrong in a
 * way that shows: a north face and a south face receive identical ambient light, so
 * a valley at sunset is lit as though the orange half of the sky were not there.
 *
 * What a surface actually receives is the sky's radiance integrated over the whole
 * hemisphere above it, weighted by the cosine of the angle to its normal. Computing
 * that per pixel is out of the question; the standard answer is that the cosine
 * kernel is so smooth that projecting the environment onto the first NINE spherical
 * harmonics loses almost nothing — the reconstruction is accurate to about one
 * percent for a diffuse surface (Ramamoorthi & Hanrahan, 2001). Nine coefficients is
 * a handful of multiply-adds per pixel, and the projection runs once per frame.
 *
 * Two constraints shaped the implementation:
 *
 *  - No transcendentals. This links into the freestanding kernel, where libm does
 *    not exist and a builtin @c sinf is a link error. The sample directions are
 *    therefore the texels of a cube map, which are pure arithmetic — an angular
 *    parameterisation would have needed sines to enumerate.
 *  - The solid angle of a cube texel is NOT uniform. A texel at a face corner
 *    subtends far less sky than one at the centre, and weighting them equally
 *    biases the whole integral towards the eight corners. The weight below is the
 *    exact Jacobian of the cube-to-sphere map.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-31
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_IRRADIANCE_HPP
#    define LPL_RENDER_IRRADIANCE_HPP

#    include <lpl/render/OrbitCamera.hpp> // inverseSqrtNewton
#    include <lpl/render/Pbr.hpp>
#    include <lpl/render/Sky.hpp>

namespace lpl::render {

/**
 * @struct IrradianceProbe
 * @brief The environment projected onto nine spherical-harmonic coefficients.
 *
 * Band 0 is the average sky (one colour), band 1 its directional bias (where the
 * bright half is), band 2 its second moment (how sharply). Beyond band 2 the cosine
 * kernel has already fallen to nothing, which is why nine is the number.
 */
struct IrradianceProbe {
    Vec3f coefficient[9];

    /** @brief A probe that reproduces one flat colour, for a caller with no sky. */
    static IrradianceProbe uniform(const Vec3f &radiance) noexcept
    {
        IrradianceProbe probe{};
        // Y00 = 0.282095, and the l=0 convolution weight is pi. The band-0 term of a
        // constant environment L is L / Y00, so that E = pi * L00 * Y00 = pi * L.
        probe.coefficient[0] = Vec3f(radiance.x / 0.282095f, radiance.y / 0.282095f, radiance.z / 0.282095f);
        return probe;
    }
};

namespace detail {

/** @brief The nine real spherical-harmonic basis functions at a direction. */
inline void shBasis(core::f32 x, core::f32 y, core::f32 z, core::f32 *out) noexcept
{
    out[0] = 0.282095f;
    out[1] = 0.488603f * y;
    out[2] = 0.488603f * z;
    out[3] = 0.488603f * x;
    out[4] = 1.092548f * x * y;
    out[5] = 1.092548f * y * z;
    out[6] = 0.315392f * (3.0f * z * z - 1.0f);
    out[7] = 1.092548f * x * z;
    out[8] = 0.546274f * (x * x - y * y);
}

} // namespace detail

/**
 * @brief Projects the sky onto an irradiance probe.
 *
 * @param resolution Texels per cube-face edge. Eight is plenty: the target is a
 *                   band-2 projection, and everything finer than a band-2 lobe is
 *                   integrated away regardless of how well it was sampled. Six
 *                   faces at eight squared is 384 sky evaluations, once per frame.
 */
[[nodiscard]] inline IrradianceProbe projectSky(const SunState &sun, const SkyParams &params,
                                                core::u32 resolution = 8u) noexcept
{
    IrradianceProbe probe{};
    if (resolution == 0u)
        resolution = 1u;

    const core::f32 step = 2.0f / static_cast<core::f32>(resolution);
    core::f32 totalWeight = 0.0f;

    for (core::u32 face = 0u; face < 6u; ++face)
        for (core::u32 v = 0u; v < resolution; ++v)
            for (core::u32 u = 0u; u < resolution; ++u)
            {
                // Face coordinates at texel centres, in [-1, 1].
                const core::f32 a = -1.0f + (static_cast<core::f32>(u) + 0.5f) * step;
                const core::f32 b = -1.0f + (static_cast<core::f32>(v) + 0.5f) * step;

                core::f32 dx = 0.0f;
                core::f32 dy = 0.0f;
                core::f32 dz = 0.0f;
                switch (face)
                {
                case 0u:
                    dx = 1.0f;
                    dy = -b;
                    dz = -a;
                    break;
                case 1u:
                    dx = -1.0f;
                    dy = -b;
                    dz = a;
                    break;
                case 2u:
                    dx = a;
                    dy = 1.0f;
                    dz = b;
                    break;
                case 3u:
                    dx = a;
                    dy = -1.0f;
                    dz = -b;
                    break;
                case 4u:
                    dx = a;
                    dy = -b;
                    dz = 1.0f;
                    break;
                default:
                    dx = -a;
                    dy = -b;
                    dz = -1.0f;
                    break;
                }

                // Solid angle of the texel: the Jacobian of the cube-to-sphere map.
                // Uniform weights would pull the whole integral towards the corners,
                // where a texel subtends barely a third of what a centre one does.
                const core::f32 lengthSquared = dx * dx + dy * dy + dz * dz;
                const core::f32 inverse = inverseSqrtNewton(lengthSquared);
                const core::f32 weight = step * step * inverse * inverse * inverse;

                const core::f32 nx = dx * inverse;
                const core::f32 ny = dy * inverse;
                const core::f32 nz = dz * inverse;

                const core::u32 packed = skyColour(nx, ny, nz, sun, params);
                const Vec3f radiance(static_cast<core::f32>((packed >> 16) & 0xFFu) / 255.0f,
                                     static_cast<core::f32>((packed >> 8) & 0xFFu) / 255.0f,
                                     static_cast<core::f32>(packed & 0xFFu) / 255.0f);

                core::f32 basis[9];
                detail::shBasis(nx, ny, nz, basis);
                for (core::u32 i = 0u; i < 9u; ++i)
                {
                    probe.coefficient[i].x += radiance.x * basis[i] * weight;
                    probe.coefficient[i].y += radiance.y * basis[i] * weight;
                    probe.coefficient[i].z += radiance.z * basis[i] * weight;
                }
                totalWeight += weight;
            }

    // The cube's texel weights sum to 4*pi only in the limit; normalising by what
    // was actually accumulated keeps a coarse resolution from darkening the result.
    if (totalWeight > 0.0f)
    {
        const core::f32 correction = 12.566370614f / totalWeight;
        for (core::u32 i = 0u; i < 9u; ++i)
        {
            probe.coefficient[i].x *= correction;
            probe.coefficient[i].y *= correction;
            probe.coefficient[i].z *= correction;
        }
    }
    return probe;
}

/**
 * @brief Irradiance arriving at a surface with a given normal, divided by pi.
 *
 * Returns RADIANCE, not irradiance: the caller's shading model multiplies by the
 * albedo and expects an incoming radiance, so the 1/pi of the Lambertian BRDF is
 * folded in here rather than left for every call site to remember.
 *
 * The three constants are the cosine kernel's own spherical-harmonic coefficients.
 * Band 0 passes through, band 1 is attenuated to two thirds, band 2 to a quarter,
 * and band 3 vanishes — which is the reason the projection stops at nine.
 */
[[nodiscard]] inline Vec3f evaluateIrradiance(const IrradianceProbe &probe, core::f32 nx, core::f32 ny,
                                              core::f32 nz) noexcept
{
    constexpr core::f32 kBand0 = 3.141592654f;
    constexpr core::f32 kBand1 = 2.094395102f;
    constexpr core::f32 kBand2 = 0.785398163f;
    constexpr core::f32 kWeight[9] = {kBand0, kBand1, kBand1, kBand1, kBand2, kBand2, kBand2, kBand2, kBand2};

    core::f32 basis[9];
    detail::shBasis(nx, ny, nz, basis);

    Vec3f irradiance(0.0f, 0.0f, 0.0f);
    for (core::u32 i = 0u; i < 9u; ++i)
    {
        const core::f32 term = kWeight[i] * basis[i];
        irradiance.x += probe.coefficient[i].x * term;
        irradiance.y += probe.coefficient[i].y * term;
        irradiance.z += probe.coefficient[i].z * term;
    }

    // A band-limited reconstruction can dip below zero where the environment has a
    // sharp edge — the horizon is exactly such an edge. Negative light is not a
    // subtlety to preserve, it is black pixels on the shaded side.
    constexpr core::f32 kInversePi = 0.318309886f;
    irradiance.x = irradiance.x > 0.0f ? irradiance.x * kInversePi : 0.0f;
    irradiance.y = irradiance.y > 0.0f ? irradiance.y * kInversePi : 0.0f;
    irradiance.z = irradiance.z > 0.0f ? irradiance.z * kInversePi : 0.0f;
    return irradiance;
}

} // namespace lpl::render

#endif // LPL_RENDER_IRRADIANCE_HPP
