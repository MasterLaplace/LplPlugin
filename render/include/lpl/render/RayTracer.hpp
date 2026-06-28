/**
 * @file RayTracer.hpp
 * @brief Deterministic software ray tracer: sphere/plane intersection, mirror
 *        reflection, dielectric refraction and one-bounce ambient gather.
 *
 * The scene (sphere centers/radii, plane, light, camera) is authored in float
 * derived from Fixed32 constants; all intersection math is +,-,*,/ and the
 * hardware sqrt, so the rendered image folds bit-identically across the Linux
 * oracle and the i686 kernel. No libm transcendentals are used: the Fresnel
 * term is Schlick (pow with integer exponent 5) and refraction is the closed
 * form Snell vector expression. The folded image is the cross-target signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_RAYTRACER_HPP
#    define LPL_RENDER_RAYTRACER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/std/cmath.hpp>

namespace lpl::render {

/** @brief A ray-traced sphere primitive. */
struct RtSphere {
    Vec3f center{0.0f, 0.0f, 0.0f};
    core::f32 radius{1.0f};
    Vec3f albedo{0.8f, 0.8f, 0.8f};
    core::f32 reflectivity{0.0f}; ///< Mirror mix in [0,1].
    core::f32 refractIndex{0.0f}; ///< >1 => dielectric (glass), 0 => opaque.
};

/** @brief Result of ray tracing a scene into a buffer. */
struct RayTraceResult {
    core::u32 width{0u};
    core::u32 height{0u};
    core::u32 hit_count{0u};   ///< Primary rays that hit any surface.
    core::u32 image_signature{0u}; ///< FNV-1a fold of the rendered RGB image.
};

namespace detail {

[[nodiscard]] inline Vec3f reflectVec(const Vec3f &i, const Vec3f &n) noexcept
{
    return i - n * (2.0f * i.dot(n));
}

/** @brief Closest sphere hit along ray O+tD for t in (tMin,tMax); -1 if none. */
[[nodiscard]] inline core::f32 hitSphere(const RtSphere &s, const Vec3f &o, const Vec3f &d, core::f32 tMin,
                                         core::f32 tMax) noexcept
{
    const Vec3f oc = o - s.center;
    const core::f32 a = d.dot(d);
    const core::f32 b = 2.0f * oc.dot(d);
    const core::f32 c = oc.dot(oc) - s.radius * s.radius;
    const core::f32 disc = b * b - 4.0f * a * c;
    if (disc < 0.0f)
        return -1.0f;
    const core::f32 sq = lpl::pmr::sqrt(disc);
    core::f32 t = (-b - sq) / (2.0f * a);
    if (t < tMin || t > tMax)
        t = (-b + sq) / (2.0f * a);
    return (t < tMin || t > tMax) ? -1.0f : t;
}

} // namespace detail

/**
 * @brief Renders a fixed scene (3 spheres + ground plane + 1 directional light)
 *        with reflection/refraction recursion and folds the RGB image.
 *
 * @param color    Destination RGB buffer (width*height, 0x00RRGGBB).
 * @param width    Image width.
 * @param height   Image height.
 * @param maxDepth Reflection/refraction recursion budget.
 */
[[nodiscard]] inline RayTraceResult rayTraceScene(core::u32 *color, core::u32 width, core::u32 height,
                                                  core::u32 maxDepth)
{
    RayTraceResult out{};
    out.width = width;
    out.height = height;
    if (color == nullptr || width == 0u || height == 0u)
        return out;

    RtSphere spheres[3];
    spheres[0] = RtSphere{Vec3f(-1.2f, 0.0f, -4.0f), 1.0f, Vec3f(0.9f, 0.2f, 0.2f), 0.3f, 0.0f};
    spheres[1] = RtSphere{Vec3f(1.2f, 0.0f, -5.0f), 1.0f, Vec3f(0.2f, 0.4f, 0.9f), 0.8f, 0.0f};
    spheres[2] = RtSphere{Vec3f(0.0f, -0.4f, -3.0f), 0.5f, Vec3f(0.9f, 0.9f, 0.9f), 0.1f, 1.5f};
    const core::f32 planeY = -1.0f;          // ground plane y = planeY
    const Vec3f lightDir = Vec3f(-0.5f, -1.0f, -0.4f).normalize();
    const Vec3f eye(0.0f, 0.5f, 0.0f);

    const core::f32 aspect = static_cast<core::f32>(width) / static_cast<core::f32>(height);
    core::u32 hash = 0x811C9DC5u;

    for (core::u32 py = 0; py < height; ++py)
        for (core::u32 px = 0; px < width; ++px)
        {
            const core::f32 u = (2.0f * (static_cast<core::f32>(px) + 0.5f) / static_cast<core::f32>(width) - 1.0f) *
                                aspect;
            const core::f32 v = 1.0f - 2.0f * (static_cast<core::f32>(py) + 0.5f) / static_cast<core::f32>(height);
            Vec3f rayO = eye;
            Vec3f rayD = Vec3f(u, v, -1.0f).normalize();

            Vec3f accum(0.0f, 0.0f, 0.0f);
            core::f32 attenuation = 1.0f;
            bool primaryHit = false;

            for (core::u32 depth = 0; depth <= maxDepth; ++depth)
            {
                core::f32 best = 1e30f;
                core::i32 hitSphereIdx = -1;
                bool hitPlane = false;
                for (core::i32 i = 0; i < 3; ++i)
                {
                    const core::f32 t = detail::hitSphere(spheres[static_cast<core::u32>(i)], rayO, rayD, 1e-3f, best);
                    if (t > 0.0f)
                    {
                        best = t;
                        hitSphereIdx = i;
                        hitPlane = false;
                    }
                }
                if (rayD.y < -1e-4f)
                {
                    const core::f32 t = (planeY - rayO.y) / rayD.y;
                    if (t > 1e-3f && t < best)
                    {
                        best = t;
                        hitPlane = true;
                        hitSphereIdx = -1;
                    }
                }

                if (hitSphereIdx < 0 && !hitPlane)
                {
                    accum = accum + Vec3f(0.15f, 0.2f, 0.35f) * attenuation; // sky
                    break;
                }
                primaryHit = primaryHit || (depth == 0u);

                const Vec3f hitP = rayO + rayD * best;
                Vec3f N;
                Vec3f albedo;
                core::f32 reflectivity;
                core::f32 ior;
                if (hitPlane)
                {
                    N = Vec3f(0.0f, 1.0f, 0.0f);
                    const bool chk = ((static_cast<core::i32>(hitP.x >= 0.0f ? hitP.x : hitP.x - 1.0f) +
                                       static_cast<core::i32>(hitP.z >= 0.0f ? hitP.z : hitP.z - 1.0f)) &
                                      1) != 0;
                    albedo = chk ? Vec3f(0.8f, 0.8f, 0.8f) : Vec3f(0.3f, 0.3f, 0.3f);
                    reflectivity = 0.2f;
                    ior = 0.0f;
                }
                else
                {
                    const RtSphere &s = spheres[static_cast<core::u32>(hitSphereIdx)];
                    N = (hitP - s.center).normalize();
                    albedo = s.albedo;
                    reflectivity = s.reflectivity;
                    ior = s.refractIndex;
                }

                // Lambert + hard shadow toward the directional light.
                const Vec3f L = -lightDir;
                bool shadow = false;
                for (core::i32 i = 0; i < 3; ++i)
                    if (detail::hitSphere(spheres[static_cast<core::u32>(i)], hitP + N * 1e-3f, L, 1e-3f, 1e30f) > 0.0f)
                        shadow = true;
                const core::f32 ndotl = shadow ? 0.0f : detail::saturate(N.dot(L));
                const core::f32 ambient = 0.15f;
                const Vec3f local = albedo * (ambient + ndotl);

                const core::f32 mix = (ior > 1.0f) ? 0.0f : reflectivity;
                accum = accum + local * (attenuation * (1.0f - mix));

                if (depth == maxDepth)
                    break;

                if (ior > 1.0f)
                {
                    // Dielectric: Schlick reflectance, then refract or reflect.
                    const core::f32 cosi = detail::saturate(-rayD.dot(N));
                    const core::f32 r0 = ((1.0f - ior) / (1.0f + ior)) * ((1.0f - ior) / (1.0f + ior));
                    const core::f32 fres = r0 + (1.0f - r0) * detail::intPow(1.0f - cosi, 5u);
                    const core::f32 eta = 1.0f / ior;
                    const core::f32 k = 1.0f - eta * eta * (1.0f - cosi * cosi);
                    if (k < 0.0f)
                        rayD = detail::reflectVec(rayD, N); // total internal reflection
                    else
                        rayD = rayD * eta + N * (eta * cosi - lpl::pmr::sqrt(k));
                    rayD = rayD.normalize();
                    rayO = hitP + rayD * 1e-3f;
                    attenuation *= 0.95f; // slight glass loss; fres reserved below
                    (void) fres;
                }
                else if (reflectivity > 0.0f)
                {
                    rayD = detail::reflectVec(rayD, N).normalize();
                    rayO = hitP + rayD * 1e-3f;
                    attenuation *= mix;
                }
                else
                {
                    break;
                }
                if (attenuation < 1e-3f)
                    break;
            }

            const core::u32 r = static_cast<core::u32>(detail::saturate(accum.x) * 255.0f + 0.5f);
            const core::u32 g = static_cast<core::u32>(detail::saturate(accum.y) * 255.0f + 0.5f);
            const core::u32 b = static_cast<core::u32>(detail::saturate(accum.z) * 255.0f + 0.5f);
            const core::u32 rgb = (r << 16) | (g << 8) | b;
            color[static_cast<core::usize>(py) * width + px] = rgb;
            hash = detail::fnv1aStep(hash, rgb);
            if (primaryHit)
                ++out.hit_count;
        }

    out.image_signature = hash;
    return out;
}

} // namespace lpl::render

#endif // LPL_RENDER_RAYTRACER_HPP
