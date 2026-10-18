/**
 * @file Revolve.hpp
 * @brief Solids of revolution from a 2D profile, and instanced drawing of them.
 *
 * The cheapest way to get a rounded solid out of code with no assets: take a
 * closed curve as a silhouette and sweep it around the vertical axis. A boulder, a
 * pot, a trunk, a tower — all the same operation with a different profile, and the
 * profile can come from @ref Topology.hpp's Catmull-Rom tessellation, which is
 * what finally puts that module's output on screen instead of only in a signature.
 *
 * The mesh is a QUAD STRIP grid, stored flat: rings along the sweep, samples along
 * the profile. Drawing it goes through the clipper, so a boulder the walker is
 * standing next to behaves like the terrain does.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_REVOLVE_HPP
#    define LPL_RENDER_REVOLVE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

/**
 * @struct RevolvedMesh
 * @brief A solid of revolution as a ring x profile grid of positions.
 *
 * Positions only. Normals are computed at draw time from the quad itself, which
 * for a shape this size costs less than storing them and cannot fall out of step
 * with the geometry it describes.
 */
struct RevolvedMesh {
    pmr::vector<core::f32> positions; /**< 3 floats per vertex, ring-major. */
    core::u32 rings{0u};              /**< Sweep steps around the axis. */
    core::u32 samples{0u};            /**< Points along the profile. */
    core::f32 height{0.0f};           /**< Model-space height, for culling. */
    core::f32 radius{0.0f};           /**< Model-space horizontal reach. */

    [[nodiscard]] bool empty() const noexcept { return rings < 3u || samples < 2u; }
};

/**
 * @brief Sweeps a profile around the Y axis.
 *
 * One jitter per ring, so the silhouette stays a closed curve: jittering per
 * vertex instead tears the surface into facets that do not meet.
 *
 * @param profileRadius Radius at each profile sample.
 * @param profileHeight Height at each profile sample.
 * @param samples       Profile samples.
 * @param rings         Sweep steps; 8 is plenty for a rock at world scale.
 * @param wobble        Per-ring radial jitter in [0, 1): what stops a revolved
 *                      solid from looking turned on a lathe. A boulder that is
 *                      perfectly circular in plan reads as a vase.
 */
[[nodiscard]] inline RevolvedMesh revolveProfile(const core::f32 *profileRadius, const core::f32 *profileHeight,
                                                 core::u32 samples, core::u32 rings, core::u32 seed,
                                                 core::f32 wobble = 0.25f);

/**
 * @brief Draws one instance of a revolved mesh, lit by a directional light.
 *
 * Each quad's normal is the cross product of two of its edges; the quad may face
 * either way, so the Lambert term takes the absolute value of N.L. The light is
 * not normalised: it arrives as a unit vector.
 *
 * @param light Direction TOWARDS the light (the sun vector), for a flat Lambert
 *              term per quad — enough on a rock, which has no smooth shading to
 *              lose.
 * @return Triangles submitted.
 */
inline core::u32 drawRevolved(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, const RevolvedMesh &mesh,
                              core::f32 worldX, core::f32 worldY, core::f32 worldZ, core::f32 scale, core::u32 albedo,
                              const math::Vec3<core::f32> &light, core::f32 ambient) noexcept;

} // namespace lpl::render

#    include "Revolve.inl"

#endif // LPL_RENDER_REVOLVE_HPP
