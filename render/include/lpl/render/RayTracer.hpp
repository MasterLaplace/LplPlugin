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
    core::f32 reflectivity{0.0f}; /**< Mirror mix in [0,1]. */
    core::f32 refractIndex{0.0f}; /**< >1 => dielectric (glass), 0 => opaque. */
};

/** @brief Result of ray tracing a scene into a buffer. */
struct RayTraceResult {
    core::u32 width{0u};
    core::u32 height{0u};
    core::u32 hit_count{0u};       /**< Primary rays that hit any surface. */
    core::u32 image_signature{0u}; /**< FNV-1a fold of the rendered RGB image. */
};

namespace detail {

[[nodiscard]] inline Vec3f reflectVec(const Vec3f &i, const Vec3f &n) noexcept { return i - n * (2.0f * i.dot(n)); }

/**
 * @brief Closest sphere hit along ray O+tD for t in (tMin,tMax); -1 if none.
 */
[[nodiscard]] inline core::f32 hitSphere(const RtSphere &s, const Vec3f &o, const Vec3f &d, core::f32 tMin,
                                         core::f32 tMax) noexcept;

} // namespace detail

/**
 * @brief Renders a fixed scene (3 spheres + ground plane + 1 directional light)
 *        with reflection/refraction recursion and folds the RGB image.
 *
 * The ground plane is y = -1. Each hit is lit by Lambert with a hard shadow
 * toward the directional light; a ray that hits nothing takes the sky colour.
 * A dielectric refracts the ray, or reflects it on total internal reflection,
 * and loses 5% of its weight per bounce: its Schlick reflectance is computed
 * but not used.
 *
 * @param color    Destination RGB buffer (width*height, 0x00RRGGBB).
 * @param width    Image width.
 * @param height   Image height.
 * @param maxDepth Reflection/refraction recursion budget.
 */
[[nodiscard]] inline RayTraceResult rayTraceScene(core::u32 *color, core::u32 width, core::u32 height,
                                                  core::u32 maxDepth);

} // namespace lpl::render

#    include "RayTracer.inl"

#endif // LPL_RENDER_RAYTRACER_HPP
