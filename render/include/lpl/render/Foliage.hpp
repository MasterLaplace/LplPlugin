/**
 * @file Foliage.hpp
 * @brief Draws a grown plant: tapered branch ribbons and camera-facing leaves.
 *
 * Takes a MESH and a TRANSFORM, separately and on purpose. One plant is grown
 * once — by @c procgen::growTree, which this header deliberately does not know
 * about — and then drawn at as many positions, scales and headings as the world
 * has plants. That is instancing in the only form a software rasterizer can have
 * it: the vertex work per instance is unavoidable, the GROWING is not, and
 * growing is the expensive half by orders of magnitude.
 *
 * Kept in floats: foliage is scenery. Nothing authoritative depends on where a
 * leaf ended up, so nothing here has to be Fixed32 — the shape it draws was
 * decided in Fixed32 upstream, which is where determinism is owed.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_FOLIAGE_HPP
#    define LPL_RENDER_FOLIAGE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/Sky.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::render {

/** @brief One tapered segment of a plant, in model space (foot of stem at origin). */
struct FoliageSegment {
    core::f32 x0{0.0f}, y0{0.0f}, z0{0.0f};
    core::f32 x1{0.0f}, y1{0.0f}, z1{0.0f};
    core::f32 radius0{0.0f};
    core::f32 radius1{0.0f};
    core::u8 depth{0u};
};

/** @brief One foliage cluster, in model space. */
struct FoliageSprite {
    core::f32 x{0.0f}, y{0.0f}, z{0.0f};
    core::f32 size{0.0f};
    core::u8 depth{0u};
};

/**
 * @struct FoliageMesh
 * @brief A grown plant, ready to be drawn anywhere. Non-owning.
 */
struct FoliageMesh {
    const FoliageSegment *segments{nullptr};
    core::u32 segmentCount{0u};
    const FoliageSprite *sprites{nullptr};
    core::u32 spriteCount{0u};
    core::f32 height{0.0f}; /**< Model-space height, for culling and shadows. */
    core::f32 spread{0.0f}; /**< Model-space horizontal reach. */
};

/** @brief Where and how big one instance of a mesh is. */
struct FoliageInstance {
    core::f32 x{0.0f}, y{0.0f}, z{0.0f}; /**< World position of the foot. */
    core::f32 scale{1.0f};
    core::f32 yaw{0.0f}; /**< Rotation about Y, so a stand does not look stamped. */
};

/** @brief Colours and lighting for one draw. */
struct FoliageStyle {
    core::u32 bark{0x00553B24u};
    core::u32 leaf{0x00274F1Fu};
    core::f32 light{1.0f};      /**< Lighting term, already including any shadow. */
    core::u32 hazeTint{0u};     /**< Sky colour to fade into with distance. */
    core::f32 fogDensity{0.0f}; /**< 0 disables the fade. */
    core::u32 maxDepth{255u};   /**< Branches deeper than this are skipped: a LOD knob. */
    /**
     * Nearer than this, the leaves are dropped and only the wood is drawn.
     *
     * Standing next to a trunk is a legal position and looks like one: bark, and
     * the world past it. Standing INSIDE a canopy fills the view with leaf quads,
     * which is geometrically correct and useless — you cannot see, and neither
     * can you tell that what you are looking at is a tree. Dropping the sprites
     * and keeping the wood says "there is a tree here" in two triangles.
     */
    core::f32 spriteMinDistance{0.0f};
};

namespace detail {

/**
 * @brief Cross product, for the ribbon that stands in for a cylinder.
 */
inline void cross3(core::f32 ax, core::f32 ay, core::f32 az, core::f32 bx, core::f32 by, core::f32 bz, core::f32 &ox,
                   core::f32 &oy, core::f32 &oz) noexcept;

} // namespace detail

/**
 * @brief Draws one instance of a plant.
 *
 * A branch is a RIBBON, not a cylinder: a quad whose width axis is perpendicular
 * to both the branch and the view direction, which from any angle covers exactly
 * the silhouette a cylinder would while costing two triangles instead of sixteen.
 * At the scale a tree occupies on a 480x300 target the difference is invisible,
 * and the cost is not.
 *
 * Model to world is a rotation about Y and a uniform scale, written out rather
 * than composed into the MVP because the leaves need the world position on its
 * own, to face the camera from it. A branch seen exactly end-on covers nothing
 * and is skipped. Deeper branches are darker: a canopy shades what is under it,
 * and this is the cheapest honest way to say so without a second light pass.
 * Leaves take two tones, alternating by index: a canopy in one flat colour reads
 * as a solid object, and a crown is not solid.
 *
 * @return Triangles submitted, so a caller can report its own budget honestly.
 */
inline core::u32 drawFoliage(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, const CameraBasis &basis,
                             const FoliageMesh &mesh, const FoliageInstance &instance,
                             const FoliageStyle &style) noexcept;

} // namespace lpl::render

#    include "Foliage.inl"

#endif // LPL_RENDER_FOLIAGE_HPP
