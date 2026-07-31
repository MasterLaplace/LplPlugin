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
    pmr::vector<core::f32> positions; ///< 3 floats per vertex, ring-major.
    core::u32 rings{0u};              ///< Sweep steps around the axis.
    core::u32 samples{0u};            ///< Points along the profile.
    core::f32 height{0.0f};           ///< Model-space height, for culling.
    core::f32 radius{0.0f};           ///< Model-space horizontal reach.

    [[nodiscard]] bool empty() const noexcept { return rings < 3u || samples < 2u; }
};

/**
 * @brief Sweeps a profile around the Y axis.
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
                                                 core::f32 wobble = 0.25f)
{
    RevolvedMesh mesh;
    if (profileRadius == nullptr || profileHeight == nullptr || samples < 2u || rings < 3u)
        return mesh;

    mesh.rings = rings;
    mesh.samples = samples;
    mesh.positions.resize(static_cast<core::usize>(rings) * samples * 3u, 0.0f);

    for (core::u32 ring = 0u; ring < rings; ++ring)
    {
        const core::f32 angle = (6.2831853f * static_cast<core::f32>(ring)) / static_cast<core::f32>(rings);
        const core::f32 cosine = OrbitCamera::cosOf(angle);
        const core::f32 sine = OrbitCamera::sinOf(angle);

        // One jitter per ring, so the silhouette stays a closed curve: jittering
        // per vertex instead tears the surface into facets that do not meet.
        core::u32 hash = 0x811C9DC5u ^ seed;
        hash = (hash ^ ring) * 0x01000193u;
        hash = (hash ^ 0x9E3779B9u) * 0x01000193u;
        const core::f32 jitter = 1.0f + wobble * ((static_cast<core::f32>((hash >> 8) & 0xFFu) / 255.0f) - 0.5f);

        for (core::u32 sample = 0u; sample < samples; ++sample)
        {
            const core::f32 r = profileRadius[sample] * jitter;
            const core::f32 y = profileHeight[sample];
            const core::usize base = (static_cast<core::usize>(ring) * samples + sample) * 3u;
            mesh.positions[base + 0u] = r * cosine;
            mesh.positions[base + 1u] = y;
            mesh.positions[base + 2u] = r * sine;
            if (y > mesh.height)
                mesh.height = y;
            if (r > mesh.radius)
                mesh.radius = r;
        }
    }
    return mesh;
}

/**
 * @brief Draws one instance of a revolved mesh, lit by a directional light.
 *
 * @param light Direction TOWARDS the light (the sun vector), for a flat Lambert
 *              term per quad — enough on a rock, which has no smooth shading to
 *              lose.
 * @return Triangles submitted.
 */
inline core::u32 drawRevolved(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, const RevolvedMesh &mesh,
                              core::f32 worldX, core::f32 worldY, core::f32 worldZ, core::f32 scale, core::u32 albedo,
                              const math::Vec3<core::f32> &light, core::f32 ambient) noexcept
{
    if (mesh.empty())
        return 0u;

    core::u32 triangles = 0u;
    for (core::u32 ring = 0u; ring < mesh.rings; ++ring)
    {
        const core::u32 next = (ring + 1u) % mesh.rings;
        for (core::u32 sample = 0u; sample + 1u < mesh.samples; ++sample)
        {
            const auto at = [&](core::u32 r, core::u32 sIndex, core::u32 axis) {
                return mesh.positions[(static_cast<core::usize>(r) * mesh.samples + sIndex) * 3u + axis];
            };
            const core::f32 quad[12] = {
                worldX + at(ring, sample, 0u) * scale,          worldY + at(ring, sample, 1u) * scale,
                worldZ + at(ring, sample, 2u) * scale,          worldX + at(next, sample, 0u) * scale,
                worldY + at(next, sample, 1u) * scale,          worldZ + at(next, sample, 2u) * scale,
                worldX + at(next, sample + 1u, 0u) * scale,     worldY + at(next, sample + 1u, 1u) * scale,
                worldZ + at(next, sample + 1u, 2u) * scale,     worldX + at(ring, sample + 1u, 0u) * scale,
                worldY + at(ring, sample + 1u, 1u) * scale,     worldZ + at(ring, sample + 1u, 2u) * scale};

            // Normal from two edges of this quad: cross product, then the Lambert
            // term. No normalisation of the light — it arrives as a unit vector.
            const core::f32 e1x = quad[3] - quad[0];
            const core::f32 e1y = quad[4] - quad[1];
            const core::f32 e1z = quad[5] - quad[2];
            const core::f32 e2x = quad[9] - quad[0];
            const core::f32 e2y = quad[10] - quad[1];
            const core::f32 e2z = quad[11] - quad[2];
            core::f32 nx = e1y * e2z - e1z * e2y;
            core::f32 ny = e1z * e2x - e1x * e2z;
            core::f32 nz = e1x * e2y - e1y * e2x;
            const core::f32 lengthSquared = nx * nx + ny * ny + nz * nz;
            core::f32 lambert = ambient;
            if (lengthSquared > 1.0e-9f)
            {
                const core::f32 inverse = inverseSqrtNewton(lengthSquared);
                nx *= inverse;
                ny *= inverse;
                nz *= inverse;
                core::f32 ndl = nx * light.x + ny * light.y + nz * light.z;
                ndl = ndl < 0.0f ? -ndl : ndl; // the quad may face either way
                lambert = ambient + (1.0f - ambient) * ndl;
            }

            triangles += fillPolygonClipped(rt, mvp, quad, 4u, modulate(albedo, lambert));
        }
    }
    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_REVOLVE_HPP
