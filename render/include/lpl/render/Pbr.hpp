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
    core::f32 metallic{0.0f};  ///< 0 = dielectric, 1 = metal.
    core::f32 roughness{0.5f}; ///< Perceptual roughness in (0,1].
    core::f32 ao{1.0f};        ///< Ambient occlusion.
};

/** @brief Tone-mapping operator for HDR -> LDR. */
enum class ToneMap : core::u32 {
    Reinhard = 0u, ///< x / (1 + x), per channel.
    Aces = 1u,     ///< Narkowicz ACES fit (polynomial).
};

namespace detail {

constexpr core::f32 kPi = 3.14159265358979323846f;

/** @brief GGX / Trowbridge-Reitz normal distribution. */
[[nodiscard]] inline core::f32 distributionGGX(core::f32 ndoth, core::f32 roughness) noexcept
{
    const core::f32 a = roughness * roughness;
    const core::f32 a2 = a * a;
    const core::f32 d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * d * d + 1e-7f);
}

/** @brief Smith geometry term with Schlick-GGX, one direction. */
[[nodiscard]] inline core::f32 geometrySchlickGGX(core::f32 ndotv, core::f32 k) noexcept
{
    return ndotv / (ndotv * (1.0f - k) + k + 1e-7f);
}

/** @brief Fresnel-Schlick; (1-cosTheta)^5 uses the integer-exponent pow. */
[[nodiscard]] inline Vec3f fresnelSchlick(core::f32 cosTheta, const Vec3f &f0) noexcept
{
    const core::f32 f = intPow(saturate(1.0f - cosTheta), 5u);
    return f0 + (Vec3f(1.0f, 1.0f, 1.0f) - f0) * f;
}

} // namespace detail

/**
 * @brief Evaluates Cook-Torrance for one light, returning HDR radiance (linear).
 *
 * @param mat     PBR material.
 * @param N       Unit surface normal.
 * @param V       Unit view direction (surface -> camera).
 * @param L       Unit light direction (surface -> light).
 * @param radiance Incoming light radiance (color * intensity * attenuation).
 */
[[nodiscard]] inline Vec3f pbrDirect(const PbrMaterial &mat, const Vec3f &N, const Vec3f &V, const Vec3f &L,
                                     const Vec3f &radiance) noexcept
{
    const Vec3f H = (V + L).normalize();
    const core::f32 ndotv = detail::saturate(N.dot(V));
    const core::f32 ndotl = detail::saturate(N.dot(L));
    const core::f32 ndoth = detail::saturate(N.dot(H));
    const core::f32 hdotv = detail::saturate(H.dot(V));

    // F0: 0.04 for dielectrics, albedo for metals.
    const Vec3f dielectric(0.04f, 0.04f, 0.04f);
    const Vec3f f0 = dielectric + (mat.albedo - dielectric) * mat.metallic;

    const core::f32 D = detail::distributionGGX(ndoth, mat.roughness);
    const core::f32 r1 = mat.roughness + 1.0f;
    const core::f32 k = (r1 * r1) / 8.0f; // direct-lighting remap
    const core::f32 G = detail::geometrySchlickGGX(ndotv, k) * detail::geometrySchlickGGX(ndotl, k);
    const Vec3f F = detail::fresnelSchlick(hdotv, f0);

    const Vec3f numerator = F * (D * G);
    const core::f32 denom = 4.0f * ndotv * ndotl + 1e-7f;
    const Vec3f specular = numerator / denom;

    // Energy conservation: diffuse only for the non-reflected, non-metal part.
    const Vec3f kd = (Vec3f(1.0f, 1.0f, 1.0f) - F) * (1.0f - mat.metallic);
    const Vec3f diffuse = Vec3f(mat.albedo.x * kd.x, mat.albedo.y * kd.y, mat.albedo.z * kd.z) / detail::kPi;

    const Vec3f brdf = diffuse + specular;
    return Vec3f(brdf.x * radiance.x, brdf.y * radiance.y, brdf.z * radiance.z) * ndotl;
}

/** @brief Tone-maps an HDR radiance to [0,1] per channel. */
[[nodiscard]] inline Vec3f toneMap(ToneMap op, const Vec3f &hdr) noexcept
{
    if (op == ToneMap::Reinhard)
        return Vec3f(hdr.x / (1.0f + hdr.x), hdr.y / (1.0f + hdr.y), hdr.z / (1.0f + hdr.z));

    // Narkowicz ACES fit: (x(ax+b)) / (x(cx+d)+e).
    const auto aces = [](core::f32 x) {
        constexpr core::f32 a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
        return detail::saturate((x * (a * x + b)) / (x * (c * x + d) + e));
    };
    return Vec3f(aces(hdr.x), aces(hdr.y), aces(hdr.z));
}

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
[[nodiscard]] inline core::u32 pbrShadeToRgb(const PbrMaterial &mat, const Light *lights, core::u32 count, const Vec3f &N,
                                             const Vec3f &fragPos, const Vec3f &viewPos, const Vec3f &hdriAmbient,
                                             ToneMap op) noexcept
{
    const Vec3f V = (viewPos - fragPos).normalize();
    Vec3f Lo = Vec3f(mat.albedo.x * hdriAmbient.x, mat.albedo.y * hdriAmbient.y, mat.albedo.z * hdriAmbient.z) * mat.ao;

    for (core::u32 i = 0; i < count; ++i)
    {
        const Light &lt = lights[i];
        Vec3f L;
        core::f32 attenuation = 1.0f;
        if (lt.type == LightType::Directional)
        {
            L = (-lt.direction).normalize();
        }
        else
        {
            const Vec3f toLight = lt.position - fragPos;
            const core::f32 dist = lpl::pmr::sqrt(toLight.lengthSquared());
            L = (dist > 0.0f) ? (toLight / dist) : Vec3f(0.0f, 1.0f, 0.0f);
            const core::f32 dd = dist / lt.range;
            attenuation = detail::saturate(1.0f - dd * dd);
        }
        const Vec3f radiance = lt.color * (lt.intensity * attenuation);
        Lo = Lo + pbrDirect(mat, N, V, L, radiance);
    }

    const Vec3f mapped = toneMap(op, Lo);
    const core::u32 r = static_cast<core::u32>(detail::saturate(mapped.x) * 255.0f + 0.5f);
    const core::u32 g = static_cast<core::u32>(detail::saturate(mapped.y) * 255.0f + 0.5f);
    const core::u32 b = static_cast<core::u32>(detail::saturate(mapped.z) * 255.0f + 0.5f);
    return (r << 16) | (g << 8) | b;
}

} // namespace lpl::render

#endif // LPL_RENDER_PBR_HPP
