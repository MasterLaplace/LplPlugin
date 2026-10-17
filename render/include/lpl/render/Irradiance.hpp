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

#    include <lpl/render/OrbitCamera.hpp>
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

    /**
     * @brief A probe that reproduces one flat colour, for a caller with no sky.
     *
     * Y00 = 0.282095, and the l=0 convolution weight is pi. The band-0 term of a
     * constant environment L is L / Y00, so that E = pi * L00 * Y00 = pi * L.
     */
    static IrradianceProbe uniform(const Vec3f &radiance) noexcept;
};

namespace detail {

/**
 * @brief The nine real spherical-harmonic basis functions at a direction.
 */
inline void shBasis(core::f32 x, core::f32 y, core::f32 z, core::f32 *out) noexcept;

} // namespace detail

/**
 * @brief Projects the sky onto an irradiance probe.
 *
 * Each cube face is sampled at its texel centres, face coordinates in [-1, 1].
 * A texel is weighted by its solid angle, the Jacobian of the cube-to-sphere map
 * (step^2 / |d|^3): uniform weights would pull the whole integral towards the
 * corners, where a texel subtends barely a third of what a centre one does. The
 * cube's texel weights sum to 4*pi only in the limit; normalising by what was
 * actually accumulated keeps a coarse resolution from darkening the result.
 *
 * @param resolution Texels per cube-face edge. Eight is plenty: the target is a
 *                   band-2 projection, and everything finer than a band-2 lobe is
 *                   integrated away regardless of how well it was sampled. Six
 *                   faces at eight squared is 384 sky evaluations, once per frame.
 */
[[nodiscard]] inline IrradianceProbe projectSky(const SunState &sun, const SkyParams &params,
                                                core::u32 resolution = 8u) noexcept;

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
 *
 * A band-limited reconstruction can dip below zero where the environment has a
 * sharp edge — the horizon is exactly such an edge. Negative light is not a
 * subtlety to preserve, it is black pixels on the shaded side: it is clamped to 0.
 */
[[nodiscard]] inline Vec3f evaluateIrradiance(const IrradianceProbe &probe, core::f32 nx, core::f32 ny,
                                              core::f32 nz) noexcept;

} // namespace lpl::render

#    include "Irradiance.inl"

#endif // LPL_RENDER_IRRADIANCE_HPP
