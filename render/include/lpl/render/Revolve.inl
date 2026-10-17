#ifndef LPL_RENDER_REVOLVE_INL
#define LPL_RENDER_REVOLVE_INL

namespace lpl::render {

inline RevolvedMesh revolveProfile(const core::f32 *profileRadius, const core::f32 *profileHeight, core::u32 samples,
                                   core::u32 rings, core::u32 seed, core::f32 wobble)
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
                worldX + at(ring, sample, 0u) * scale,      worldY + at(ring, sample, 1u) * scale,
                worldZ + at(ring, sample, 2u) * scale,      worldX + at(next, sample, 0u) * scale,
                worldY + at(next, sample, 1u) * scale,      worldZ + at(next, sample, 2u) * scale,
                worldX + at(next, sample + 1u, 0u) * scale, worldY + at(next, sample + 1u, 1u) * scale,
                worldZ + at(next, sample + 1u, 2u) * scale, worldX + at(ring, sample + 1u, 0u) * scale,
                worldY + at(ring, sample + 1u, 1u) * scale, worldZ + at(ring, sample + 1u, 2u) * scale};

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
                ndl = ndl < 0.0f ? -ndl : ndl;
                lambert = ambient + (1.0f - ambient) * ndl;
            }

            triangles += fillPolygonClipped(rt, mvp, quad, 4u, modulate(albedo, lambert));
        }
    }
    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_REVOLVE_INL
