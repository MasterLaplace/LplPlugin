/**
 * @file Surface.cpp
 * @brief A depth-only rasteriser, so the marcher can composite a surface without testing triangles.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Surface.hpp>

namespace lpl::voxel {

namespace {

[[nodiscard]] core::f32 squareRoot(core::f32 v) noexcept
{
    if (v <= 0.0f)
        return 0.0f;
    core::f32 g = v > 1.0f ? v : 1.0f;
    for (int i = 0; i < 24; ++i)
        g = 0.5f * (g + v / g);
    return g;
}

struct Projected final {
    core::f32 x{0.0f};
    core::f32 y{0.0f};
    core::f32 depth{0.0f}; ///< Along the eye's forward axis, in metres.
    bool behind{true};
};

} // namespace

core::u32 patchIndices(const SheetPatch &patch, core::u32 *out, core::u32 capacity) noexcept
{
    if (out == nullptr || patch.points == nullptr || patch.rows < 2u || patch.columns < 2u)
        return 0u;

    core::u32 written = 0u;
    for (core::u32 r = 0u; r + 1u < patch.rows; ++r)
    {
        for (core::u32 c = 0u; c + 1u < patch.columns; ++c)
        {
            const core::u32 a = r * patch.columns + c;
            const core::u32 b = a + 1u;
            const core::u32 d = a + patch.columns;
            const core::u32 e = d + 1u;

            // A row that stopped short was padded with its last point, so its quads collapse.
            // Dropping them leaves a missing corner, which is where the trace actually failed --
            // filling it would invent surface exactly where the tool could not follow one.
            const auto degenerate = [&](core::u32 i, core::u32 j, core::u32 k) noexcept {
                const math::Vec3<core::f32> &p = patch.points[i];
                const math::Vec3<core::f32> &q = patch.points[j];
                const math::Vec3<core::f32> &s = patch.points[k];
                const math::Vec3<core::f32> u{q.x - p.x, q.y - p.y, q.z - p.z};
                const math::Vec3<core::f32> v{s.x - p.x, s.y - p.y, s.z - p.z};
                const math::Vec3<core::f32> n{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
                return n.lengthSquared() < 1e-8f;
            };

            if (written + 6u > capacity)
                return written;
            if (!degenerate(a, b, d))
            {
                out[written++] = a;
                out[written++] = b;
                out[written++] = d;
            }
            if (!degenerate(b, e, d))
            {
                out[written++] = b;
                out[written++] = e;
                out[written++] = d;
            }
        }
    }
    return written;
}

void clearSurfaceDepth(const SurfaceDepth &depth) noexcept
{
    if (!depth.valid())
        return;
    const core::usize n = static_cast<core::usize>(depth.width) * depth.height;
    for (core::usize i = 0u; i < n; ++i)
    {
        depth.metres[i] = kNoSurface;
        depth.facing[i] = 0.0f;
        if (depth.u != nullptr)
            depth.u[i] = 0.0f;
        if (depth.v != nullptr)
            depth.v[i] = 0.0f;
    }
}

void computeVertexNormals(const SurfaceMesh &mesh, math::Vec3<core::f32> *out) noexcept
{
    if (!mesh.valid() || out == nullptr)
        return;
    for (core::u32 i = 0u; i < mesh.pointCount; ++i)
        out[i] = math::Vec3<core::f32>{0.0f, 0.0f, 0.0f};

    for (core::u32 t = 0u; t + 2u < mesh.indexCount; t += 3u)
    {
        const core::u32 ia = mesh.indices[t];
        const core::u32 ib = mesh.indices[t + 1u];
        const core::u32 ic = mesh.indices[t + 2u];
        if (ia >= mesh.pointCount || ib >= mesh.pointCount || ic >= mesh.pointCount)
            continue;
        const math::Vec3<core::f32> &a = mesh.points[ia];
        const math::Vec3<core::f32> &b = mesh.points[ib];
        const math::Vec3<core::f32> &c = mesh.points[ic];
        const math::Vec3<core::f32> u{b.x - a.x, b.y - a.y, b.z - a.z};
        const math::Vec3<core::f32> v{c.x - a.x, c.y - a.y, c.z - a.z};
        // Left unnormalised on purpose: the cross product's length is twice the triangle's area,
        // so summing them IS the area weighting.
        const math::Vec3<core::f32> n{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};

        // ⚠ A sheet has no outside, so neighbouring triangles can be wound opposite ways and their
        // normals would cancel -- leaving a zero normal and a black band along every such seam.
        // Aligning each contribution with what the vertex has so far keeps the accumulation
        // meaningful; the resulting sign is arbitrary, which is exactly what two-sided lighting
        // expects.
        for (core::u32 k = 0u; k < 3u; ++k)
        {
            const core::u32 index = mesh.indices[t + k];
            math::Vec3<core::f32> &acc = out[index];
            const core::f32 align = acc.x * n.x + acc.y * n.y + acc.z * n.z;
            const core::f32 sign = align < 0.0f ? -1.0f : 1.0f;
            acc.x += n.x * sign;
            acc.y += n.y * sign;
            acc.z += n.z * sign;
        }
    }

    for (core::u32 i = 0u; i < mesh.pointCount; ++i)
    {
        const core::f32 len2 = out[i].lengthSquared();
        if (len2 < 1e-20f)
        {
            // A vertex no triangle reached, or one whose contributions cancelled exactly. Facing
            // the eye is the honest default: it shades flat rather than black, and black would
            // read as a hole in the surface.
            out[i] = math::Vec3<core::f32>{0.0f, 0.0f, 1.0f};
            continue;
        }
        const core::f32 inv = 1.0f / squareRoot(len2);
        out[i].x *= inv;
        out[i].y *= inv;
        out[i].z *= inv;
    }
}

core::u32 rasteriseSurface(const SurfaceMesh &mesh, const VolumeGeometry &geometry, const Eye &eye,
                           const SurfaceDepth &depth) noexcept
{
    if (!mesh.valid() || !depth.valid() || !geometry.valid())
        return 0u;

    const core::f32 mps = geometry.metresPerSample();
    const core::f32 aspect = static_cast<core::f32>(depth.height) / static_cast<core::f32>(depth.width);
    const core::f32 h = eye.horizontalFieldOfView * 0.5f;
    const core::f32 h2 = h * h;
    const core::f32 tanHalf = h * (1.0f + h2 * (1.0f / 3.0f + h2 * (2.0f / 15.0f)));
    const core::f32 halfW = static_cast<core::f32>(depth.width) * 0.5f;
    const core::f32 halfH = static_cast<core::f32>(depth.height) * 0.5f;

    // The same basis the marcher builds its rays from, so a triangle lands where the ray that
    // should hit it goes. Deriving the projection separately is how a surface ends up offset from
    // the scan by a pixel and nobody can say why.
    const auto project = [&](const math::Vec3<core::f32> &samplePoint) noexcept {
        const math::Vec3<core::f32> world{samplePoint.x * mps, samplePoint.y * mps, samplePoint.z * mps};
        const math::Vec3<core::f32> rel{world.x - eye.position.x, world.y - eye.position.y, world.z - eye.position.z};
        Projected p{};
        p.depth = rel.x * eye.forward.x + rel.y * eye.forward.y + rel.z * eye.forward.z;
        if (p.depth <= 1e-4f)
            return p;
        const core::f32 right = rel.x * eye.right.x + rel.y * eye.right.y + rel.z * eye.right.z;
        const core::f32 up = rel.x * eye.up.x + rel.y * eye.up.y + rel.z * eye.up.z;
        p.x = halfW + (right / (p.depth * tanHalf)) * halfW;
        p.y = halfH - (up / (p.depth * tanHalf * aspect)) * halfH;
        p.behind = false;
        return p;
    };

    core::u32 drawn = 0u;
    for (core::u32 i = 0u; i + 2u < mesh.indexCount; i += 3u)
    {
        const math::Vec3<core::f32> &a = mesh.points[mesh.indices[i]];
        const math::Vec3<core::f32> &b = mesh.points[mesh.indices[i + 1u]];
        const math::Vec3<core::f32> &c = mesh.points[mesh.indices[i + 2u]];

        const Projected pa = project(a);
        const Projected pb = project(b);
        const Projected pc = project(c);
        // Any vertex behind the eye drops the triangle. Clipping properly would matter for a scene
        // the camera flies through; a patch is small and dropping it at the edge of view costs a
        // sliver rather than producing the wrap-around smear an unclipped projection gives.
        if (pa.behind || pb.behind || pc.behind)
            continue;

        // Two-sided: a sheet has no outside, so a triangle facing away is the same surface seen
        // from behind and must still be drawn.
        const math::Vec3<core::f32> u{b.x - a.x, b.y - a.y, b.z - a.z};
        const math::Vec3<core::f32> v{c.x - a.x, c.y - a.y, c.z - a.z};
        math::Vec3<core::f32> n{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
        const core::f32 len2 = n.lengthSquared();
        core::f32 facing = 0.0f;
        if (len2 > 1e-12f)
        {
            const core::f32 inv = 1.0f / squareRoot(len2);
            facing = (n.x * eye.forward.x + n.y * eye.forward.y + n.z * eye.forward.z) * inv;
            if (facing < 0.0f)
                facing = -facing;
        }

        core::f32 minX = pa.x < pb.x ? (pa.x < pc.x ? pa.x : pc.x) : (pb.x < pc.x ? pb.x : pc.x);
        core::f32 maxX = pa.x > pb.x ? (pa.x > pc.x ? pa.x : pc.x) : (pb.x > pc.x ? pb.x : pc.x);
        core::f32 minY = pa.y < pb.y ? (pa.y < pc.y ? pa.y : pc.y) : (pb.y < pc.y ? pb.y : pc.y);
        core::f32 maxY = pa.y > pb.y ? (pa.y > pc.y ? pa.y : pc.y) : (pb.y > pc.y ? pb.y : pc.y);
        if (minX < 0.0f)
            minX = 0.0f;
        if (minY < 0.0f)
            minY = 0.0f;
        if (maxX > static_cast<core::f32>(depth.width - 1u))
            maxX = static_cast<core::f32>(depth.width - 1u);
        if (maxY > static_cast<core::f32>(depth.height - 1u))
            maxY = static_cast<core::f32>(depth.height - 1u);
        if (maxX < minX || maxY < minY)
            continue;

        const core::f32 area = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
        if (area > -1e-6f && area < 1e-6f)
            continue;
        const core::f32 invArea = 1.0f / area;

        bool covered = false;
        for (core::u32 py = static_cast<core::u32>(minY); py <= static_cast<core::u32>(maxY); ++py)
        {
            for (core::u32 px = static_cast<core::u32>(minX); px <= static_cast<core::u32>(maxX); ++px)
            {
                const core::f32 sx = static_cast<core::f32>(px) + 0.5f;
                const core::f32 sy = static_cast<core::f32>(py) + 0.5f;
                const core::f32 w0 = ((pb.x - sx) * (pc.y - sy) - (pb.y - sy) * (pc.x - sx)) * invArea;
                const core::f32 w1 = ((pc.x - sx) * (pa.y - sy) - (pc.y - sy) * (pa.x - sx)) * invArea;
                const core::f32 w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
                    continue;

                // Depth along the FORWARD axis, then corrected to distance along the ray, because
                // that is what the marcher's parameter measures. Comparing the two directly would
                // put the surface progressively too near towards the edges of the frame.
                const core::f32 forwardDepth = w0 * pa.depth + w1 * pb.depth + w2 * pc.depth;
                const core::f32 ndcX = (sx / static_cast<core::f32>(depth.width) * 2.0f - 1.0f) * tanHalf;
                const core::f32 ndcY = (1.0f - 2.0f * sy / static_cast<core::f32>(depth.height)) * aspect * tanHalf;
                const core::f32 stretch = squareRoot(1.0f + ndcX * ndcX + ndcY * ndcY);
                const core::f32 along = forwardDepth * stretch;

                const core::usize at = static_cast<core::usize>(py) * depth.width + px;
                if (along < depth.metres[at])
                {
                    depth.metres[at] = along;

                    // Per-pixel from interpolated vertex normals when the mesh has them; the flat
                    // face normal only when it does not. This is the difference between a sheet
                    // and a heap of shards.
                    core::f32 pixelFacing = facing;
                    if (mesh.normals != nullptr)
                    {
                        const core::f32 ia = w0 / pa.depth;
                        const core::f32 ib = w1 / pb.depth;
                        const core::f32 ic = w2 / pc.depth;
                        const core::f32 sum = ia + ib + ic;
                        if (sum > 1e-12f)
                        {
                            const math::Vec3<core::f32> &na = mesh.normals[mesh.indices[i]];
                            const math::Vec3<core::f32> &nb = mesh.normals[mesh.indices[i + 1u]];
                            const math::Vec3<core::f32> &nc = mesh.normals[mesh.indices[i + 2u]];
                            const core::f32 inv = 1.0f / sum;
                            math::Vec3<core::f32> shaded{(ia * na.x + ib * nb.x + ic * nc.x) * inv,
                                                         (ia * na.y + ib * nb.y + ic * nc.y) * inv,
                                                         (ia * na.z + ib * nb.z + ic * nc.z) * inv};
                            const core::f32 l2 = shaded.lengthSquared();
                            if (l2 > 1e-12f)
                            {
                                const core::f32 il = 1.0f / squareRoot(l2);
                                pixelFacing =
                                    (shaded.x * eye.forward.x + shaded.y * eye.forward.y + shaded.z * eye.forward.z) *
                                    il;
                                if (pixelFacing < 0.0f)
                                    pixelFacing = -pixelFacing;
                            }
                        }
                    }
                    depth.facing[at] = pixelFacing;
                    if (depth.u != nullptr && depth.v != nullptr && mesh.textured())
                    {
                        // Perspective-correct: weights divided by depth, renormalised. A sheet is
                        // usually seen edge-on, so a single triangle spans a huge depth range and
                        // an affine interpolation slides the texture visibly along the surface.
                        const core::f32 ia = w0 / pa.depth;
                        const core::f32 ib = w1 / pb.depth;
                        const core::f32 ic = w2 / pc.depth;
                        const core::f32 sum = ia + ib + ic;
                        if (sum > 1e-12f)
                        {
                            const core::u32 ta = mesh.textureIndices[i];
                            const core::u32 tb = mesh.textureIndices[i + 1u];
                            const core::u32 tc = mesh.textureIndices[i + 2u];
                            if (ta < mesh.textureCount && tb < mesh.textureCount && tc < mesh.textureCount)
                            {
                                const core::f32 inv = 1.0f / sum;
                                depth.u[at] = (ia * mesh.texture[ta * 2u] + ib * mesh.texture[tb * 2u] +
                                               ic * mesh.texture[tc * 2u]) *
                                              inv;
                                depth.v[at] = (ia * mesh.texture[ta * 2u + 1u] + ib * mesh.texture[tb * 2u + 1u] +
                                               ic * mesh.texture[tc * 2u + 1u]) *
                                              inv;
                            }
                        }
                    }
                    covered = true;
                }
            }
        }
        if (covered)
            ++drawn;
    }
    return drawn;
}

} // namespace lpl::voxel
