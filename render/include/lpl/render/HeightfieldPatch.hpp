/**
 * @file HeightfieldPatch.hpp
 * @brief Drawing one patch of a heightfield: stride, normals, light, skirts.
 *
 * The inner loop every terrain in this engine was going to write. It is not the
 * loop that is interesting — a quad per cell is obvious — it is everything around
 * it that is easy to get wrong and was got wrong at least once here:
 *
 *  - the SLOPE has to be divided by the stride, or a patch drawn at stride 4 is
 *    lit as if its cells were four times steeper than they are;
 *  - the quad goes through the near-plane clipper, because the patch under the
 *    walker's feet crosses that plane every single frame;
 *  - the shadow term multiplies the DIRECT light only, so a shadowed slope keeps
 *    the sky and comes out blue rather than black;
 *  - the edges need a skirt, because two patches sampled at different strides do
 *    not meet, and the gap goes straight through to the sky.
 *
 * The patch is accessed through a callable rather than a container, so this header
 * needs to know nothing about who stores the field or in what type — a Fixed32
 * grid, a float array, a procedural sampler.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_HEIGHTFIELD_PATCH_HPP
#    define LPL_RENDER_HEIGHTFIELD_PATCH_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/Sky.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::render {

/**
 * @struct HeightfieldPatchParams
 * @brief How one patch is sampled and lit.
 */
struct HeightfieldPatchParams {
    core::u32 size{24u};  ///< Cells along one edge of the patch.
    core::u32 stride{1u}; ///< Sampling stride: the level of detail.
    core::i32 originX{0}; ///< World cell of the patch's corner.
    core::i32 originZ{0};
    core::f32 ambient{0.28f};  ///< What a surface facing away still receives.
    core::f32 maxLight{1.25f}; ///< Ceiling on the lighting term.
};

/// The default hole rule: a heightfield is a continuous surface.
struct NoPatchHoles {
    [[nodiscard]] constexpr bool operator()(core::u32, core::u32) const noexcept { return false; }
};

/**
 * @brief Draws one patch, one quad per (stride x stride) block.
 *
 * @param heightAt   (x, z) -> f32, in patch-local cells.
 * @param shadeAt    (x, z) -> f32 occlusion in [0, 1]; 0 for a patch with no mask.
 * @param colourAt   (x, z) -> packed base colour.
 * @param shade      (worldX, worldZ, base, litTerm, nx, nz, occlusion) -> packed
 *                   colour, called PER PIXEL when @p perPixel is set, and once per
 *                   quad otherwise. One functor for both paths, so the flat path
 *                   cannot drift from the shaded one.
 * @param skipAt     (x, z) -> bool, patch-local. True leaves the quad UNDRAWN.
 *
 * The one way a heightfield can have a hole in it, and it needs one: a cave mouth is
 * an opening in a hillside, and the surface passes straight across it. Nothing else
 * about a heightfield changes — the hole is a decision the caller makes about a few
 * named cells, not a property of the field. Defaults to no holes, so a surface world
 * is exactly what it was.
 *
 * @return Triangles submitted.
 */
template <typename HeightAt, typename ShadeAt, typename ColourAt, typename Shader, typename SkipAt = NoPatchHoles>
core::u32 drawHeightfieldPatch(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                               const HeightfieldPatchParams &params, const SunState &sun, HeightAt &&heightAt,
                               ShadeAt &&shadeAt, ColourAt &&colourAt, Shader &&shade, bool perPixel,
                               SkipAt &&skipAt = SkipAt{})
{
    const core::u32 stride = params.stride == 0u ? 1u : params.stride;
    const core::f32 strideF = static_cast<core::f32>(stride);
    core::u32 triangles = 0u;

    // <=, not <. With size 24 the strict test stopped at 22, so the quad between
    // cell 23 and cell 24 was never drawn — a gap one cell wide around EVERY chunk,
    // on all four sides. From above it is invisible; standing in it, it is a bright
    // slit running away to the horizon, because what shows through is the skirt.
    //
    // The consequence is that @p heightAt is now asked for index == size, one cell
    // PAST the patch. That is deliberate and it is the only way two patches can
    // share an edge: the caller answers from the neighbouring chunk (or from the
    // world height function, which is defined everywhere), and because both chunks
    // sample the same absolute world coordinate they agree on it by construction.
    for (core::u32 z = 0u; z + stride <= params.size; z += stride)
    {
        for (core::u32 x = 0u; x + stride <= params.size; x += stride)
        {
            if (skipAt(x, z))
                continue;
            const core::f32 y00 = heightAt(x, z);
            const core::f32 y10 = heightAt(x + stride, z);
            const core::f32 y11 = heightAt(x + stride, z + stride);
            const core::f32 y01 = heightAt(x, z + stride);

            const core::f32 x0 = static_cast<core::f32>(params.originX + static_cast<core::i32>(x));
            const core::f32 x1 = x0 + strideF;
            const core::f32 z0 = static_cast<core::f32>(params.originZ + static_cast<core::i32>(z));
            const core::f32 z1 = z0 + strideF;

            // Divided by the stride: the height difference is over that many cells,
            // and forgetting it lights a coarse patch as a cliff.
            const core::f32 nx = (y00 - y10) / strideF;
            const core::f32 nz = (y00 - y01) / strideF;
            const core::f32 normalScale = 1.0f / (1.0f + nx * nx + nz * nz);
            core::f32 ndl = (nx * sun.x + sun.y + nz * sun.z) * normalScale;
            ndl = ndl < 0.0f ? 0.0f : ndl;

            const core::f32 occlusion = shadeAt(x, z);
            // The shadow multiplies the DIRECT term only.
            const core::f32 lit =
                OrbitCamera::clamp(params.ambient + (1.0f - params.ambient) * ndl * sun.intensity * (1.0f - occlusion),
                                   0.0f, params.maxLight);
            const core::u32 base = colourAt(x, z);

            const core::f32 quad[12] = {x0, y00, z0, x1, y10, z0, x1, y11, z1, x0, y01, z1};
            if (perPixel)
                triangles += fillPolygonShadedClipped(rt, mvp, quad, 4u, [&](core::f32 wx, core::f32 wy, core::f32 wz) {
                    (void) wy;
                    return shade(wx, wz, base, lit, nx, nz, occlusion);
                });
            else
                triangles += fillPolygonClipped(rt, mvp, quad, 4u, shade(x0, z0, base, lit, nx, nz, occlusion));
        }
    }
    return triangles;
}

/**
 * @brief Hangs a skirt around a patch's four edges.
 *
 * Adjacent patches sampled at different strides disagree along their shared edge,
 * and the disagreement is a crack through to the sky. A curtain does not make them
 * agree; it puts opaque geometry behind the gap, which is all the eye needs. Its
 * colour comes from the GROUND rather than a fixed dark tone: a black skirt
 * announces the grid it was meant to hide.
 *
 * @param cliffAt (x, z) -> packed colour for the skirt at that edge cell.
 */
template <typename HeightAt, typename CliffAt>
core::u32 drawPatchSkirts(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                          const HeightfieldPatchParams &params, core::f32 drop, HeightAt &&heightAt, CliffAt &&cliffAt)
{
    const core::u32 last = params.size - 1u;
    core::u32 triangles = 0u;

    for (core::u32 i = 0u; i + 1u < params.size; ++i)
    {
        const core::f32 wx = static_cast<core::f32>(params.originX + static_cast<core::i32>(i));
        const core::f32 wz = static_cast<core::f32>(params.originZ + static_cast<core::i32>(i));
        const core::f32 x0 = static_cast<core::f32>(params.originX);
        const core::f32 x1 = static_cast<core::f32>(params.originX + static_cast<core::i32>(last));
        const core::f32 z0 = static_cast<core::f32>(params.originZ);
        const core::f32 z1 = static_cast<core::f32>(params.originZ + static_cast<core::i32>(last));

        triangles +=
            drawSkirtQuad(rt, mvp, wx, heightAt(i, 0u), z0, wx + 1.0f, heightAt(i + 1u, 0u), z0, drop, cliffAt(i, 0u));
        triangles += drawSkirtQuad(rt, mvp, wx, heightAt(i, last), z1, wx + 1.0f, heightAt(i + 1u, last), z1, drop,
                                   cliffAt(i, last));
        triangles +=
            drawSkirtQuad(rt, mvp, x0, heightAt(0u, i), wz, x0, heightAt(0u, i + 1u), wz + 1.0f, drop, cliffAt(0u, i));
        triangles += drawSkirtQuad(rt, mvp, x1, heightAt(last, i), wz, x1, heightAt(last, i + 1u), wz + 1.0f, drop,
                                   cliffAt(last, i));
    }
    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_HEIGHTFIELD_PATCH_HPP
