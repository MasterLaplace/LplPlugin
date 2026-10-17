/**
 * @file Pbr.hpp
 * @brief Cook-Torrance metallic/roughness PBR + HDRI tone mapping.
 *
 * The full BRDF (GGX normal distribution, Smith-Schlick geometry, Schlick
 * Fresnel) needs only +,-,*,/, the hardware sqrt and an INTEGER-exponent pow
 * (Fresnel's (1-cosTheta)^5) — never powf/expf, which are neither
 * freestanding-linkable nor deterministic. Lighting runs in float (SSE,
 * -ffp-contract=off), bit-identical across the Linux oracle and the i686 kernel.
 * Tone mapping (Reinhard and ACES-fitted, both rational/polynomial) maps the HDR
 * radiance to [0,1]. The folded shaded colors are the cross-target signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_PBR_HPP
#    define LPL_RENDER_PBR_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/std/cmath.hpp>

namespace lpl::render {

/** @brief Metallic/roughness PBR material. */
struct PbrMaterial {
    Vec3f albedo{0.8f, 0.6f, 0.4f};
    core::f32 metallic{0.0f};  /**< 0 = dielectric, 1 = metal. */
    core::f32 roughness{0.5f}; /**< Perceptual roughness in (0,1]. */
    core::f32 ao{1.0f};        /**< Ambient occlusion. */
};

/** @brief Tone-mapping operator for HDR -> LDR. */
enum class ToneMap : core::u32 {
    Reinhard = 0u, /**< x / (1 + x), per channel. */
    Aces = 1u,     /**< Narkowicz ACES fit (polynomial). */
};

namespace detail {

constexpr core::f32 kPi = 3.14159265358979323846f;

/**
 * @brief GGX / Trowbridge-Reitz normal distribution.
 */
[[nodiscard]] inline core::f32 distributionGGX(core::f32 ndoth, core::f32 roughness) noexcept;

/**
 * @brief Smith geometry term with Schlick-GGX, one direction.
 */
[[nodiscard]] inline core::f32 geometrySchlickGGX(core::f32 ndotv, core::f32 k) noexcept
{
    return ndotv / (ndotv * (1.0f - k) + k + 1e-7f);
}

/**
 * @brief Fresnel-Schlick; (1-cosTheta)^5 uses the integer-exponent pow.
 */
[[nodiscard]] inline Vec3f fresnelSchlick(core::f32 cosTheta, const Vec3f &f0) noexcept;

} // namespace detail

/**
 * @brief Evaluates Cook-Torrance for one light, returning HDR radiance (linear).
 *
 * F0 is 0.04 for dielectrics and the albedo for metals. The geometry term takes
 * the direct-lighting remap k = (roughness + 1)^2 / 8. Energy conservation: the
 * diffuse term is only the non-reflected, non-metal part.
 *
 * @param mat     PBR material.
 * @param N       Unit surface normal.
 * @param V       Unit view direction (surface -> camera).
 * @param L       Unit light direction (surface -> light).
 * @param radiance Incoming light radiance (color * intensity * attenuation).
 */
[[nodiscard]] inline Vec3f pbrDirect(const PbrMaterial &mat, const Vec3f &N, const Vec3f &V, const Vec3f &L,
                                     const Vec3f &radiance) noexcept;

/**
 * @brief Tone-maps an HDR radiance to [0,1] per channel.
 *
 * The Narkowicz ACES fit is (x(ax+b)) / (x(cx+d)+e).
 */
[[nodiscard]] inline Vec3f toneMap(ToneMap op, const Vec3f &hdr) noexcept;

/**
 * @brief Full PBR shade of a fragment: ambient (HDRI constant) + N lights,
 *        tone-mapped and packed to 0x00RRGGBB.
 *
 * @param mat      PBR material.
 * @param lights   Classical Light array (reused for direction/color/intensity).
 * @param count    Number of lights.
 * @param N        Unit normal.
 * @param fragPos  World-space fragment position.
 * @param viewPos  World-space camera position.
 * @param hdriAmbient Constant HDRI irradiance (the "environment" term).
 * @param op       Tone-mapping operator.
 */
[[nodiscard]] inline core::u32 pbrShadeToRgb(const PbrMaterial &mat, const Light *lights, core::u32 count,
                                             const Vec3f &N, const Vec3f &fragPos, const Vec3f &viewPos,
                                             const Vec3f &hdriAmbient, ToneMap op) noexcept;

} // namespace lpl::render

#    include "Pbr.inl"

#endif // LPL_RENDER_PBR_HPP
