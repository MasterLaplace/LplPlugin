/**
 * @file Reflection.hpp
 * @brief Planar reflection by rendering the world again from a mirrored camera.
 *
 * The per-pixel mirror in @ref Water.hpp reflects the SKY, which is correct and
 * incomplete: a lake at the foot of a mountain shows the mountain. That cannot
 * come from a sky function, because the mountain is geometry — the only ways to
 * get it are to trace rays against the world or to render the world again from
 * the other side of the water. This is the second, which on a software rasterizer
 * is far cheaper: one extra pass at a quarter of the resolution against a mirrored
 * matrix, versus one ray march per water pixel.
 *
 * The mirrored camera is exact, not an approximation: reflecting the eye through
 * the plane and flipping the vertical axis of the frame produces the view that a
 * viewer standing under the water would have, which is precisely what the surface
 * shows. Sampling it back is a projection of the water point through the SAME
 * matrix the probe was rendered with — so the reflection lands where the geometry
 * is, and cannot drift when the camera moves.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_REFLECTION_HPP
#    define LPL_RENDER_REFLECTION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>
#    include <lpl/render/Texture.hpp>

namespace lpl::render {

/**
 * @struct ReflectionProbe
 * @brief A mirrored view of the world, and the matrix it was rendered with.
 *
 * The matrix is kept alongside the pixels because sampling requires it: a
 * reflection is only correct if the point being shaded is projected through the
 * same transform that produced the image. Storing the two apart is how a
 * reflection ends up lagging a frame behind the camera.
 */
struct ReflectionProbe {
    core::f32 planeY{0.0f};            /**< Height of the mirror plane. */
    math::Mat4<core::f32> mirrorMvp{}; /**< Matrix the probe was rendered with. */
    core::u32 width{0u};
    core::u32 height{0u};
    bool valid{false}; /**< False until something has been rendered into it. */
};

/**
 * @brief Camera frame reflected about a horizontal plane.
 *
 * Only the vertical components change sign: the eye moves to the far side of the
 * plane, the forward and up vectors flip in y, and right is untouched because the
 * plane is horizontal. Deriving it by negating the whole frame instead produces a
 * mirror that is also left-right flipped, which reads as a reflection until
 * something asymmetric appears in it.
 */
[[nodiscard]] inline CameraBasis mirrorBasisAboutPlane(const CameraBasis &basis, core::f32 planeY) noexcept;

/**
 * @brief View-projection for the mirrored camera.
 *
 * Up is flipped, so the handedness of the mirrored view is inverted — which is
 * exactly why the rasterizer's double-sided fills are needed for this pass.
 *
 * @param aspect Aspect ratio of the PROBE target, which need not match the frame:
 *               a reflection carries far less detail than the view that shows it,
 *               so the probe is rendered small on purpose.
 */
[[nodiscard]] inline math::Mat4<core::f32> mirrorViewProjection(const CameraBasis &mirrored, core::f32 aspect,
                                                                const CameraLens &lens) noexcept;

/**
 * @brief Samples a probe at a world point, or reports that it cannot.
 *
 * Returns false when the point projects outside the probe or behind its near
 * plane. A caller must then fall back to something else — the sky — because a
 * clamped edge texel would smear the shoreline across the whole lake, which is
 * the classic planar-reflection artefact and looks worse than no reflection.
 */
[[nodiscard]] inline bool sampleProbe(const ReflectionProbe &probe, const Texture &pixels, core::f32 worldX,
                                      core::f32 worldY, core::f32 worldZ, core::u32 &outColour) noexcept;

} // namespace lpl::render

#    include "Reflection.inl"

#endif // LPL_RENDER_REFLECTION_HPP
