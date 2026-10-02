/**
 * @file Surface.hpp
 * @brief Putting a traced surface inside the scan it came from, so somebody can see whether it fits.
 *
 * @warning **This is the measurement the whole tool is for.** A traced sheet is judged today by
 * looking at flat slices with the trace drawn on top, one slice at a time; whether the surface
 * follows a real sheet along its whole length is a question those pictures answer badly. Drawn
 * inside the volume, at the scale a body walks, it is answered by looking.
 *
 * @warning **The surface is drawn as an INTRUSION, never as part of the scan.** It is somebody's
 * inference -- traced by this tool or loaded from a file -- and a reader has to be able to tell it
 * from the samples underneath at a glance. That is why it has its own colour, its own opacity, and
 * why the opacity is allowed to be low: a surface you can see through is a surface you can check
 * against what is behind it.
 *
 * @warning **Depth first, then the march.** The surface is rasterised into a per-pixel depth once,
 * and the marcher composites it when a ray reaches that depth. Intersecting every ray against
 * every triangle instead would put the triangle count on the hot path of every pixel, and a patch
 * is thousands of triangles.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_SURFACE_HPP
#    define LPL_VOXEL_SURFACE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/voxel/Raymarch.hpp>
#    include <lpl/voxel/Sheet.hpp>
#    include <lpl/voxel/Volume.hpp>

namespace lpl::voxel {

/**
 * @struct SurfaceMesh
 * @brief Triangles in level-0 sample space, not owned.
 *
 * Sample space rather than metres, because that is the space the trace produced them in and the
 * space a file stores them in. Converting once, here, keeps the staging scale in one place.
 */
struct SurfaceMesh final {
    const math::Vec3<core::f32> *points{nullptr}; ///< Vertices, (x, y, z) in level-0 samples.
    const core::u32 *indices{nullptr};            ///< Three per triangle.
    core::u32 pointCount{0};
    core::u32 indexCount{0};

    /**
     * Texture coordinates, and one index per corner rather than per vertex.
     *
     * @warning **Per CORNER, because a vertex on a seam has two of them.** A flattened sheet is
     * cut somewhere, and the vertices along that cut carry a different coordinate on each side; an
     * array indexed by vertex silently picks one, and everything painted from the texture is
     * smeared across the cut. Null when the mesh carries none, which is the normal case for a
     * surface this tool traced -- it has geometry and no flattening.
     */
    /**
     * One normal per vertex, or null.
     *
     * @warning **Without these the surface is FLAT-shaded, and that is what makes a traced sheet
     * look like a heap of shards.** A normal computed once per triangle gives every pixel of that
     * triangle the same brightness, so each facet reads as its own plate -- which is a fact about
     * the shading, not about the sheet. It is the first thing anybody notices and the easiest to
     * mistake for the geometry being wrong. Interpolated per-vertex normals cost one extra
     * interpolation per pixel and the facets disappear.
     */
    const math::Vec3<core::f32> *normals{nullptr};

    const core::f32 *texture{nullptr};        ///< Pairs (u, v).
    const core::u32 *textureIndices{nullptr}; ///< One per corner, so @ref indexCount of them.
    core::u32 textureCount{0};

    [[nodiscard]] constexpr bool textured() const noexcept
    {
        return texture != nullptr && textureIndices != nullptr && textureCount != 0u;
    }

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return points != nullptr && indices != nullptr && indexCount >= 3u && (indexCount % 3u) == 0u;
    }
};

/**
 * @struct SurfaceDepth
 * @brief Per-pixel depth of the nearest surface, and how squarely it faces the eye.
 */
struct SurfaceDepth final {
    core::f32 *metres{nullptr}; ///< Distance along the ray; a very large value means "no surface".
    core::f32 *facing{nullptr}; ///< |N . V| at that pixel, for shading. Zero where there is none.

    /**
     * Interpolated texture coordinate at each pixel, or null when nobody asked.
     *
     * @warning Perspective-correct, not affine. A patch seen edge-on -- which is most of a sheet,
     * most of the time -- has an enormous depth range across a single triangle, and an affine
     * interpolation there slides the texture along the surface by a visible amount. Painting a
     * prediction in the wrong place on the right sheet is worse than not painting it.
     */
    core::f32 *u{nullptr};
    core::f32 *v{nullptr};

    core::u32 width{0};
    core::u32 height{0};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return metres != nullptr && facing != nullptr && width != 0u && height != 0u;
    }
};

/// Depth written where no triangle covers a pixel. Larger than any ray this renderer takes.
inline constexpr core::f32 kNoSurface = 1.0e30f;

/**
 * @brief Turns a traced patch into triangles, in the caller's storage.
 *
 * @warning A row that ended short is padded with its last point, so the patch stays rectangular
 * and its ragged edge collapses into degenerate triangles rather than into a hole with the volume
 * showing through where the trace actually failed. Degenerate triangles are dropped here, which is
 * what makes the failure visible as a missing corner instead of an invented one.
 *
 * @param out       Storage for (rows - 1) * (columns - 1) * 6 indices.
 * @return Indices written.
 */
[[nodiscard]] core::u32 patchIndices(const SheetPatch &patch, core::u32 *out, core::u32 capacity) noexcept;

/**
 * @brief Area-weighted normal at each vertex of @p mesh, into @p out.
 *
 * Area-weighted rather than a plain average: a mesh whose triangles differ wildly in size -- which
 * a traced patch with a ragged edge always is -- would otherwise let a sliver pull the normal of
 * its corner as hard as the large triangle beside it, and the shading would ripple along the edge.
 *
 * @param out  One entry per vertex, at least @p mesh.pointCount of them.
 */
void computeVertexNormals(const SurfaceMesh &mesh, math::Vec3<core::f32> *out) noexcept;

/**
 * @brief Rasterises @p mesh into @p depth for the given eye.
 *
 * A depth-only software rasteriser: no colours, no texture, one comparison per covered pixel. The
 * marcher does the rest.
 *
 * @return Triangles that covered at least one pixel.
 */
core::u32 rasteriseSurface(const SurfaceMesh &mesh, const VolumeGeometry &geometry, const Eye &eye,
                           const SurfaceDepth &depth) noexcept;

/// @brief Fills a depth buffer with @ref kNoSurface and zero facing. Call before rasterising.
void clearSurfaceDepth(const SurfaceDepth &depth) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_SURFACE_HPP
