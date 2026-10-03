/**
 * @file Surface.cpp
 * @brief A depth-only rasteriser, so the marcher can composite a surface without testing triangles.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/std/cmath.hpp>
#include <lpl/voxel/Surface.hpp>

#include <optional>

namespace lpl::voxel {

namespace {

[[nodiscard]] bool isDegenerate(const math::Vec3<core::f32> &p, const math::Vec3<core::f32> &q,
                                const math::Vec3<core::f32> &s) noexcept
{
    return (q - p).cross(s - p).lengthSquared() < 1e-8f;
}

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

            if (written + 6u > capacity)
                return written;
            // A row that stopped short was padded with its last point, so its quads collapse.
            // Dropping them leaves a missing corner, which is where the trace actually failed --
            // filling it would invent surface exactly where the tool could not follow one.
            if (!isDegenerate(patch.points[a], patch.points[b], patch.points[d]))
            {
                out[written++] = a;
                out[written++] = b;
                out[written++] = d;
            }
            if (!isDegenerate(patch.points[b], patch.points[e], patch.points[d]))
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
    if (!mesh.valid() || !mesh.indicesInRange() || out == nullptr)
        return;
    for (core::u32 i = 0u; i < mesh.pointCount; ++i)
        out[i] = math::Vec3<core::f32>::zero();

    for (core::u32 t = 0u; t + 2u < mesh.indexCount; t += 3u)
    {
        const math::Vec3<core::f32> &a = mesh.points[mesh.indices[t]];
        const math::Vec3<core::f32> &b = mesh.points[mesh.indices[t + 1u]];
        const math::Vec3<core::f32> &c = mesh.points[mesh.indices[t + 2u]];
        // Left unnormalised on purpose: the cross product's length is twice the triangle's area,
        // so summing them IS the area weighting.
        const math::Vec3<core::f32> n = (b - a).cross(c - a);

        // ⚠ A sheet has no outside, so neighbouring triangles can be wound opposite ways and their
        // normals would cancel -- leaving a zero normal and a black band along every such seam.
        // Aligning each contribution with what the vertex has so far keeps the accumulation
        // meaningful; the resulting sign is arbitrary, which is exactly what two-sided lighting
        // expects.
        for (core::u32 k = 0u; k < 3u; ++k)
        {
            math::Vec3<core::f32> &accumulated = out[mesh.indices[t + k]];
            accumulated += accumulated.dot(n) < 0.0f ? -n : n;
        }
    }

    for (core::u32 i = 0u; i < mesh.pointCount; ++i)
    {
        // A vertex no triangle reached, or one whose contributions cancelled exactly, has no
        // direction to normalise. +Z is no better a direction than any other: it only keeps every
        // entry a unit vector.
        if (out[i].lengthSquared() < 1e-20f)
            out[i] = math::Vec3<core::f32>::unitZ();
        else
            out[i] = out[i].normalize();
    }
}

namespace {

struct Projected final {
    core::f32 x{0.0f};
    core::f32 y{0.0f};
    core::f32 depth{0.0f}; ///< Along the eye's forward axis, in metres.
    bool behind{true};
};

/// How a point in level-0 samples lands on the frame: the marcher's @ref ImagePlane, read backwards.
struct Projection final {
    Eye eye;
    ImagePlane plane;
    core::f32 metresPerSample;

    [[nodiscard]] Projected project(const math::Vec3<core::f32> &samplePoint) const noexcept
    {
        const math::Vec3<core::f32> relative = samplePoint * metresPerSample - eye.position;
        Projected p{};
        p.depth = relative.dot(eye.forward);
        if (p.depth <= 1e-4f)
            return p;
        p.x = plane.column(relative.dot(eye.right) / p.depth);
        p.y = plane.row(relative.dot(eye.up) / p.depth);
        p.behind = false;
        return p;
    }

    /// Depth along the FORWARD axis, corrected to distance along the ray through (@p x, @p y),
    /// because that is what the marcher's parameter measures. Comparing the two directly would put
    /// the surface progressively too near towards the edges of the frame.
    [[nodiscard]] core::f32 alongRay(core::f32 forwardDepth, core::f32 x, core::f32 y) const noexcept
    {
        const core::f32 right = plane.rightSlope(x);
        const core::f32 up = plane.upSlope(y);
        return forwardDepth * pmr::sqrt(1.0f + right * right + up * up);
    }
};

/// Weights of a triangle's three corners at one point.
struct Barycentric final {
    core::f32 a{0.0f};
    core::f32 b{0.0f};
    core::f32 c{0.0f};

    [[nodiscard]] constexpr bool inside() const noexcept { return a >= 0.0f && b >= 0.0f && c >= 0.0f; }
};

template <typename Value>
[[nodiscard]] constexpr Value blend(const Barycentric &w, const Value &a, const Value &b, const Value &c) noexcept
{
    return a * w.a + b * w.b + c * w.c;
}

/// Twice the signed area of the projected triangle: its sign is its winding on the screen.
[[nodiscard]] core::f32 twiceSignedArea(const Projected (&corner)[3]) noexcept
{
    return (corner[1].x - corner[0].x) * (corner[2].y - corner[0].y) -
           (corner[1].y - corner[0].y) * (corner[2].x - corner[0].x);
}

/// The weights of a pixel centre in screen space; @p inverseArea is one over @ref twiceSignedArea.
[[nodiscard]] Barycentric screenWeights(const Projected (&corner)[3], core::f32 inverseArea, core::f32 sx,
                                        core::f32 sy) noexcept
{
    const Projected &pa = corner[0];
    const Projected &pb = corner[1];
    const Projected &pc = corner[2];
    Barycentric w{};
    w.a = ((pb.x - sx) * (pc.y - sy) - (pb.y - sy) * (pc.x - sx)) * inverseArea;
    w.b = ((pc.x - sx) * (pa.y - sy) - (pc.y - sy) * (pa.x - sx)) * inverseArea;
    w.c = 1.0f - w.a - w.b;
    return w;
}

/**
 * Screen weights divided by each corner's depth and renormalised: what interpolates a quantity that
 * belongs to the surface rather than to the screen. A sheet is usually seen edge-on, so a single
 * triangle spans a huge depth range and screen weights slide the quantity visibly along it.
 */
[[nodiscard]] std::optional<Barycentric> perspectiveWeights(const Barycentric &screen,
                                                            const Projected (&corner)[3]) noexcept
{
    const Barycentric divided{screen.a / corner[0].depth, screen.b / corner[1].depth, screen.c / corner[2].depth};
    const core::f32 sum = divided.a + divided.b + divided.c;
    if (!(sum > 1e-12f))
        return std::nullopt;
    const core::f32 inverse = 1.0f / sum;
    return Barycentric{divided.a * inverse, divided.b * inverse, divided.c * inverse};
}

/// |N . V| for @p normal of any length; zero for one with no direction.
[[nodiscard]] core::f32 facingOf(const math::Vec3<core::f32> &normal, const math::Vec3<core::f32> &forward) noexcept
{
    const core::f32 lengthSquared = normal.lengthSquared();
    if (!(lengthSquared > 1e-12f))
        return 0.0f;
    const core::f32 cosine = normal.dot(forward) / pmr::sqrt(lengthSquared);
    return cosine < 0.0f ? -cosine : cosine;
}

/// A position in a surface's own flattening.
struct TexturePoint final {
    core::f32 u{0.0f};
    core::f32 v{0.0f};
};

/// The triangle whose first corner is mesh.indices[@p first], as the rasteriser reads it.
struct MeshTriangle final {
    const SurfaceMesh &mesh;
    core::u32 first;

    [[nodiscard]] core::u32 vertexIndex(core::u32 corner) const noexcept { return mesh.indices[first + corner]; }

    [[nodiscard]] const math::Vec3<core::f32> &point(core::u32 corner) const noexcept
    {
        return mesh.points[vertexIndex(corner)];
    }

    /// Per-pixel from interpolated vertex normals when the mesh has them; @p faceFacing when it does
    /// not. This is the difference between a sheet and a heap of shards.
    [[nodiscard]] core::f32 facingAt(const std::optional<Barycentric> &weights, core::f32 faceFacing,
                                     const math::Vec3<core::f32> &forward) const noexcept
    {
        if (mesh.normals == nullptr || !weights.has_value())
            return faceFacing;
        const math::Vec3<core::f32> shaded = blend(*weights, mesh.normals[vertexIndex(0u)],
                                                   mesh.normals[vertexIndex(1u)], mesh.normals[vertexIndex(2u)]);
        return shaded.lengthSquared() > 1e-12f ? facingOf(shaded, forward) : faceFacing;
    }

    /// @pre The mesh is textured.
    [[nodiscard]] TexturePoint textureAt(const Barycentric &weights) const noexcept
    {
        const auto pair = [&](core::u32 corner) noexcept {
            return &mesh.texture[mesh.textureIndices[first + corner] * 2u];
        };
        const core::f32 *a = pair(0u);
        const core::f32 *b = pair(1u);
        const core::f32 *c = pair(2u);
        return TexturePoint{blend(weights, a[0], b[0], c[0]), blend(weights, a[1], b[1], c[1])};
    }
};

[[nodiscard]] constexpr core::f32 lowest(core::f32 a, core::f32 b, core::f32 c) noexcept
{
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

[[nodiscard]] constexpr core::f32 highest(core::f32 a, core::f32 b, core::f32 c) noexcept
{
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

/// The pixels a triangle's bounding box covers, clipped to the frame; inclusive bounds.
struct PixelBox final {
    core::u32 left{0u};
    core::u32 right{0u};
    core::u32 top{0u};
    core::u32 bottom{0u};
};

[[nodiscard]] std::optional<PixelBox> pixelsAround(const Projected (&corner)[3], core::u32 width,
                                                   core::u32 height) noexcept
{
    const core::f32 lastColumn = static_cast<core::f32>(width - 1u);
    const core::f32 lastRow = static_cast<core::f32>(height - 1u);
    const core::f32 left = lowest(corner[0].x, corner[1].x, corner[2].x);
    const core::f32 right = highest(corner[0].x, corner[1].x, corner[2].x);
    const core::f32 top = lowest(corner[0].y, corner[1].y, corner[2].y);
    const core::f32 bottom = highest(corner[0].y, corner[1].y, corner[2].y);
    const core::f32 clippedLeft = left < 0.0f ? 0.0f : left;
    const core::f32 clippedRight = right > lastColumn ? lastColumn : right;
    const core::f32 clippedTop = top < 0.0f ? 0.0f : top;
    const core::f32 clippedBottom = bottom > lastRow ? lastRow : bottom;
    if (clippedRight < clippedLeft || clippedBottom < clippedTop)
        return std::nullopt;
    return PixelBox{static_cast<core::u32>(clippedLeft), static_cast<core::u32>(clippedRight),
                    static_cast<core::u32>(clippedTop), static_cast<core::u32>(clippedBottom)};
}

/// Writes the triangle into every pixel it covers nearer than what is there. @return Whether it
/// wrote any.
[[nodiscard]] bool rasteriseTriangle(const MeshTriangle &triangle, const Projection &projection,
                                     const SurfaceDepth &depth) noexcept
{
    const Projected corner[3]{projection.project(triangle.point(0u)), projection.project(triangle.point(1u)),
                              projection.project(triangle.point(2u))};
    // Any vertex behind the eye drops the triangle. Clipping properly would matter for a scene
    // the camera flies through; a patch is small and dropping it at the edge of view costs a
    // sliver rather than producing the wrap-around smear an unclipped projection gives.
    if (corner[0].behind || corner[1].behind || corner[2].behind)
        return false;

    const std::optional<PixelBox> box = pixelsAround(corner, depth.width, depth.height);
    const core::f32 area = twiceSignedArea(corner);
    if (!box.has_value() || (area > -1e-6f && area < 1e-6f))
        return false;
    const core::f32 inverseArea = 1.0f / area;

    // Two-sided: a sheet has no outside, so a triangle facing away is the same surface seen from
    // behind and must still be drawn.
    const math::Vec3<core::f32> &a = triangle.point(0u);
    const core::f32 faceFacing =
        facingOf((triangle.point(1u) - a).cross(triangle.point(2u) - a), projection.eye.forward);
    const bool paintsTexture = depth.u != nullptr && depth.v != nullptr && triangle.mesh.textured();

    bool covered = false;
    for (core::u32 py = box->top; py <= box->bottom; ++py)
    {
        for (core::u32 px = box->left; px <= box->right; ++px)
        {
            const core::f32 sx = static_cast<core::f32>(px) + 0.5f;
            const core::f32 sy = static_cast<core::f32>(py) + 0.5f;
            const Barycentric screen = screenWeights(corner, inverseArea, sx, sy);
            if (!screen.inside())
                continue;

            const core::f32 along =
                projection.alongRay(blend(screen, corner[0].depth, corner[1].depth, corner[2].depth), sx, sy);
            const core::usize at = static_cast<core::usize>(py) * depth.width + px;
            if (!(along < depth.metres[at]))
                continue;

            const std::optional<Barycentric> surface = perspectiveWeights(screen, corner);
            depth.metres[at] = along;
            depth.facing[at] = triangle.facingAt(surface, faceFacing, projection.eye.forward);
            if (paintsTexture && surface.has_value())
            {
                const TexturePoint texture = triangle.textureAt(*surface);
                depth.u[at] = texture.u;
                depth.v[at] = texture.v;
            }
            covered = true;
        }
    }
    return covered;
}

} // namespace

core::u32 rasteriseSurface(const SurfaceMesh &mesh, const VolumeGeometry &geometry, const Eye &eye,
                           const SurfaceDepth &depth) noexcept
{
    if (!mesh.valid() || !mesh.indicesInRange() || !depth.valid() || !geometry.valid())
        return 0u;

    const Projection projection{.eye = eye,
                                .plane = imagePlane(eye, depth.width, depth.height),
                                .metresPerSample = geometry.metresPerSample()};
    core::u32 drawn = 0u;
    for (core::u32 first = 0u; first + 2u < mesh.indexCount; first += 3u)
    {
        if (rasteriseTriangle(MeshTriangle{mesh, first}, projection, depth))
            ++drawn;
    }
    return drawn;
}

} // namespace lpl::voxel
