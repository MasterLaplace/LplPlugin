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
    core::u32 size{24u};  /**< Cells along one edge of the patch. */
    core::u32 stride{1u}; /**< Sampling stride: the level of detail. */
    core::i32 originX{0}; /**< World cell of the patch's corner. */
    core::i32 originZ{0};
    core::f32 ambient{0.28f};  /**< What a surface facing away still receives. */
    core::f32 maxLight{1.25f}; /**< Ceiling on the lighting term. */

    /**
     * World-space camera position on the horizontal plane. Only read when
     * curvatureFactor is non-zero.
     */
    core::f32 cameraX{0.0f};
    core::f32 cameraZ{0.0f};
    /**
     * Per-metre-squared drop applied to a vertex as it recedes from the
     * camera on the XZ plane, so the ground falls away toward a false
     * horizon instead of staying a flat plate to infinity. 0 disables it
     * (every existing call site, and the default, is unaffected). This is a
     * display-only bend of already-computed heights: no trigonometry, one
     * multiply-add per corner, so it costs nothing to leave wired in and
     * runs anywhere plain arithmetic does, kernel included.
     */
    core::f32 curvatureFactor{0.0f};
};

/** @brief The default hole rule: a heightfield is a continuous surface. */
struct NoPatchHoles {
    [[nodiscard]] constexpr bool operator()(core::u32, core::u32) const noexcept { return false; }
};

/**
 * @brief Draws one patch, one quad per (stride x stride) block.
 *
 * The loops run while x + stride <= size, not <. With size 24 the strict test
 * stopped at 22, so the quad between cell 23 and cell 24 was never drawn — a gap
 * one cell wide around EVERY chunk, on all four sides. From above it is invisible;
 * standing in it, it is a bright slit running away to the horizon, because what
 * shows through is the skirt.
 *
 * The consequence is that @p heightAt is asked for index == size, one cell
 * PAST the patch. That is deliberate and it is the only way two patches can
 * share an edge: the caller answers from the neighbouring chunk (or from the
 * world height function, which is defined everywhere), and because both chunks
 * sample the same absolute world coordinate they agree on it by construction.
 *
 * Curvature bends the DRAWN position only, after the normal is taken from the
 * true local slope: a corner near the camera drops by nothing, one at the edge
 * of view drops by distanceXZ^2 * factor, so the ground recedes below a false
 * horizon the farther out it goes. Skipped entirely at the default (0), so every
 * caller that never asked for it pays nothing.
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
                               SkipAt &&skipAt = SkipAt{});

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
                          const HeightfieldPatchParams &params, core::f32 drop, HeightAt &&heightAt, CliffAt &&cliffAt);

} // namespace lpl::render

#    include "HeightfieldPatch.inl"

#endif // LPL_RENDER_HEIGHTFIELD_PATCH_HPP
