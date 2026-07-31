/**
 * @file Lighting.hpp
 * @brief Classical lighting models: Lambert, Phong, Blinn-Phong + light types.
 *
 * Shading runs in float (SSE, -ffp-contract=off) which is bit-identical across
 * the Linux oracle and the i686 kernel. Specular uses an INTEGER exponent
 * evaluated by repeated multiplication, never pow()/powf() — those lower to a
 * libm call that is both non-freestanding-linkable and not guaranteed
 * deterministic. Normalization uses the hardware sqrt (IEEE-exact). Colors are
 * Vec3<f32> in [0,1] and packed to 0x00RRGGBB by toRgb().
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_LIGHTING_HPP
#    define LPL_RENDER_LIGHTING_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/std/cmath.hpp>

namespace lpl::render {

using Vec3f = math::Vec3<core::f32>;

/** @brief Light category. */
enum class LightType : core::u32 {
    Directional = 0u, ///< Infinitely far (sun); uses direction only.
    Point = 1u,       ///< Omnidirectional with distance attenuation.
    Spot = 2u,        ///< Cone-limited point light.
};

/** @brief Specular evaluation model. */
enum class ShadingModel : core::u32 {
    Lambert = 0u,    ///< Diffuse only.
    Phong = 1u,      ///< Reflection-vector specular.
    BlinnPhong = 2u, ///< Half-vector specular.
};

/** @brief Surface material parameters. */
struct Material {
    Vec3f albedo{1.0f, 1.0f, 1.0f};
    core::f32 ambient{0.1f};          ///< Ambient term scale.
    core::f32 specularStrength{0.5f}; ///< Specular contribution scale.
    core::u32 shininess{32u};         ///< Integer specular exponent.
};

/** @brief A single light source. */
struct Light {
    LightType type{LightType::Directional};
    Vec3f direction{0.0f, -1.0f, 0.0f}; ///< Direction the light travels (for Directional/Spot).
    Vec3f position{0.0f, 0.0f, 0.0f};   ///< Position (for Point/Spot).
    Vec3f color{1.0f, 1.0f, 1.0f};
    core::f32 intensity{1.0f};
    core::f32 range{50.0f};        ///< Attenuation range (Point/Spot).
    core::f32 spotCosCutoff{0.8f}; ///< cos(half-angle) cone cutoff (Spot).
};

namespace detail {

[[nodiscard]] inline core::f32 saturate(core::f32 x) noexcept { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

/** @brief base^exp with an integer exponent (deterministic, no libm). */
[[nodiscard]] inline core::f32 intPow(core::f32 base, core::u32 exp) noexcept
{
    core::f32 result = 1.0f;
    core::f32 b = base;
    while (exp != 0u)
    {
        if (exp & 1u)
            result *= b;
        b *= b;
        exp >>= 1u;
    }
    return result;
}

} // namespace detail

/**
 * @brief Shades one fragment with a single light using the chosen model.
 *
 * @param model      Diffuse/specular model.
 * @param mat        Surface material.
 * @param light      Light source.
 * @param normal     Unit surface normal.
 * @param fragPos    World-space fragment position.
 * @param viewPos    World-space camera position.
 * @return Linear color contribution (un-clamped; caller sums + clamps).
 */
[[nodiscard]] inline Vec3f shadeFragment(ShadingModel model, const Material &mat, const Light &light, Vec3f normal,
                                         Vec3f fragPos, Vec3f viewPos) noexcept
{
    const Vec3f N = normal.normalize();

    // Direction from fragment TOWARD the light, plus attenuation.
    Vec3f L{};
    core::f32 attenuation = 1.0f;
    if (light.type == LightType::Directional)
    {
        L = (-light.direction).normalize();
    }
    else
    {
        const Vec3f toLight = light.position - fragPos;
        const core::f32 dist = lpl::pmr::sqrt(toLight.lengthSquared());
        L = (dist > 0.0f) ? (toLight / dist) : Vec3f(0.0f, 1.0f, 0.0f);
        const core::f32 d = dist / light.range;
        attenuation = detail::saturate(1.0f - d * d);
        if (light.type == LightType::Spot)
        {
            const core::f32 spotCos = (-L).dot(light.direction.normalize());
            if (spotCos < light.spotCosCutoff)
                attenuation = 0.0f;
        }
    }

    const Vec3f radiance = light.color * (light.intensity * attenuation);

    // Diffuse (Lambert).
    const core::f32 ndotl = detail::saturate(N.dot(L));
    Vec3f color = mat.albedo * ndotl;

    // Specular.
    if (model != ShadingModel::Lambert && ndotl > 0.0f)
    {
        const Vec3f V = (viewPos - fragPos).normalize();
        core::f32 specBase = 0.0f;
        if (model == ShadingModel::Phong)
        {
            const Vec3f R = N * (2.0f * N.dot(L)) - L; // reflect(-L, N) toward viewer
            specBase = detail::saturate(R.dot(V));
        }
        else // Blinn-Phong
        {
            const Vec3f H = (L + V).normalize();
            specBase = detail::saturate(N.dot(H));
        }
        const core::f32 spec = detail::intPow(specBase, mat.shininess) * mat.specularStrength;
        color = color + Vec3f(spec, spec, spec);
    }

    return Vec3f(color.x * radiance.x, color.y * radiance.y, color.z * radiance.z);
}

/** @brief Sums ambient + every light, clamps, packs to 0x00RRGGBB. */
[[nodiscard]] inline core::u32 shadeToRgb(ShadingModel model, const Material &mat, const Light *lights,
                                          core::u32 lightCount, Vec3f normal, Vec3f fragPos, Vec3f viewPos) noexcept
{
    Vec3f acc = mat.albedo * mat.ambient;
    for (core::u32 i = 0; i < lightCount; ++i)
        acc = acc + shadeFragment(model, mat, lights[i], normal, fragPos, viewPos);

    const core::u32 r = static_cast<core::u32>(detail::saturate(acc.x) * 255.0f + 0.5f);
    const core::u32 g = static_cast<core::u32>(detail::saturate(acc.y) * 255.0f + 0.5f);
    const core::u32 b = static_cast<core::u32>(detail::saturate(acc.z) * 255.0f + 0.5f);
    return (r << 16) | (g << 8) | b;
}

/**
 * @brief Scales a packed colour by a lighting term, SATURATING each channel.
 *
 * Masking with 0xFF instead of clamping is the difference between a lit
 * highlight and a wrap-around: a lighting term reaches past 1.0 on a slope
 * facing the light, and 0xF0 (snow) times 1.25 is 0x12C, which masked becomes
 * 0x2C — dark. On screen that is a snowfield covered in black, red and yellow
 * blocks, each one a channel that overflowed at a different point, and it reads
 * as corrupt memory rather than as the arithmetic mistake it is.
 */
[[nodiscard]] inline core::u32 modulate(core::u32 colour, core::f32 lit) noexcept
{
    const core::u32 scale = static_cast<core::u32>((lit < 0.0f ? 0.0f : lit) * 256.0f);
    const auto channel = [scale](core::u32 value) -> core::u32 {
        const core::u32 lit8 = (value * scale) >> 8;
        return lit8 > 255u ? 255u : lit8;
    };
    return (channel((colour >> 16) & 0xFFu) << 16) | (channel((colour >> 8) & 0xFFu) << 8) | channel(colour & 0xFFu);
}

/**
 * @brief Linear mix of two packed colours, channel by channel.
 *
 * Written once here because it was written twice inside the water shader and
 * would have been written a third time for the reflection probe. Mixing packed
 * colours by adding them and masking is the tempting shortcut and it is wrong:
 * two channels that sum past 255 carry into the next one, so a bright blend turns
 * a red highlight green.
 */
[[nodiscard]] inline core::u32 mixColours(core::u32 from, core::u32 to, core::f32 t) noexcept
{
    const core::f32 clamped = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const core::u32 weight = static_cast<core::u32>(clamped * 256.0f);
    core::u32 out = 0u;
    for (core::u32 shift = 0u; shift <= 16u; shift += 8u)
    {
        const core::u32 a = (from >> shift) & 0xFFu;
        const core::u32 b = (to >> shift) & 0xFFu;
        out |= (((a * (256u - weight) + b * weight) >> 8) & 0xFFu) << shift;
    }
    return out;
}

/**
 * @brief Amber-on-abyss ramp for a scalar field, so a data view reads as
 *        instrumentation rather than as a picture.
 */
[[nodiscard]] inline core::u32 heatRamp(core::f32 t) noexcept
{
    t = detail::saturate(t);
    const core::u32 r = static_cast<core::u32>((0.05f + 0.95f * t) * 255.0f);
    const core::u32 g = static_cast<core::u32>((0.05f + 0.62f * t * t) * 255.0f);
    const core::u32 b = static_cast<core::u32>((0.10f + 0.15f * t * t * t) * 255.0f);
    return (r << 16) | (g << 8) | b;
}

} // namespace lpl::render

#endif // LPL_RENDER_LIGHTING_HPP
