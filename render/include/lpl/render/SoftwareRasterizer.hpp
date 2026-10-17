/**
 * @file SoftwareRasterizer.hpp
 * @brief Portable, depth-buffered software 3D triangle rasterizer.
 *
 * Renders an indexed triangle mesh into a caller-provided linear color buffer
 * with a parallel float depth buffer (z-test). Geometry/model rotation is
 * authored in Fixed32 (CORDIC) = authoritative; projection, the perspective
 * divide and the per-pixel barycentric fill run in float (SSE,
 * -ffp-contract=off) which is bit-identical host vs kernel. The same code path
 * draws the engine test `render` on both targets and what ring 0 presents.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_SOFTWARERASTERIZER_HPP
#    define LPL_RENDER_SOFTWARERASTERIZER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/render/Texture.hpp>

namespace lpl::render {

/** @brief Caller-owned render target: packed 0x00RRGGBB color + float depth. */
struct RenderTarget {
    core::u32 *color{nullptr}; /**< width*height packed pixels (row-major, no padding). */
    core::f32 *depth{nullptr}; /**< width*height depth, smaller = nearer. */
    core::u32 width{0u};
    core::u32 height{0u};
};

namespace detail {

/** @brief One projected vertex: screen x/y (float), NDC depth, in-front flag. */
struct ScreenVertex {
    core::f32 x{0.0f};
    core::f32 y{0.0f};
    core::f32 z{0.0f};
    bool valid{false};
};

[[nodiscard]] inline ScreenVertex projectVertex(const math::Mat4<core::f32> &mvp, core::f32 wx, core::f32 wy,
                                                core::f32 wz, core::u32 width, core::u32 height) noexcept;

[[nodiscard]] inline core::f32 edge(const ScreenVertex &a, const ScreenVertex &b, core::f32 px, core::f32 py) noexcept
{
    return (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x);
}

[[nodiscard]] inline core::i32 clampInt(core::i32 v, core::i32 lo, core::i32 hi) noexcept
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * @brief Depth-tested barycentric fill of one front-facing triangle.
 *
 * A back-facing or degenerate triangle is culled.
 */
inline void fillTriangle(const RenderTarget &rt, const ScreenVertex &v0, const ScreenVertex &v1, const ScreenVertex &v2,
                         core::u32 color) noexcept;

/**
 * @brief Depth-tested, affine-UV textured fill of one front-facing triangle.
 *
 * The interpolated UVs are clamped to [0,1) and converted to Q16 for the sampler.
 */
inline void fillTriangleTextured(const RenderTarget &rt, const ScreenVertex &v0, const ScreenVertex &v1,
                                 const ScreenVertex &v2, const core::f32 *uv0, const core::f32 *uv1,
                                 const core::f32 *uv2, const Texture &tex) noexcept;

/**
 * @brief Fills a triangle whichever way it faces.
 *
 * @c fillTriangle culls anything whose screen-space area is negative, which is
 * exactly right for a closed solid: a cube's far faces are hidden by its near
 * ones, so drawing them is wasted work. A heightfield is not a solid. It is a
 * single sheet with no inside, and half of it faces away from any given camera.
 * Culled, the ground vanishes entirely and only what stands on it survives,
 * which looks like a broken projection and is in fact a correct rasterizer
 * being asked the wrong question.
 *
 * Swapping two vertices when the area comes out negative costs one edge
 * function and draws the sheet from both sides.
 */
inline void fillTriangleDoubleSided(const RenderTarget &rt, const ScreenVertex &a, const ScreenVertex &b,
                                    const ScreenVertex &c, core::u32 colour) noexcept;

/**
 * @brief Two double-sided triangles over four already-projected corners.
 */
inline void fillQuadDoubleSided(const RenderTarget &rt, const ScreenVertex &a, const ScreenVertex &b,
                                const ScreenVertex &c, const ScreenVertex &d, core::u32 colour) noexcept;

/** @brief One vertex in clip space, before the divide by w. */
struct ClipVertex {
    core::f32 x{0.0f};
    core::f32 y{0.0f};
    core::f32 z{0.0f};
    core::f32 w{0.0f};
};

/**
 * @brief Transforms a world position into clip space.
 */
[[nodiscard]] inline ClipVertex toClip(const math::Mat4<core::f32> &mvp, core::f32 wx, core::f32 wy,
                                       core::f32 wz) noexcept;

/**
 * @brief Divides by w and maps to pixels.
 */
[[nodiscard]] inline ScreenVertex toScreen(const ClipVertex &v, core::u32 width, core::u32 height) noexcept;

} // namespace detail

/** Smallest @c w a vertex may have and still be projected. */
inline constexpr core::f32 kNearPlaneEpsilon = 0.05f;

/**
 * @struct ShadedVertex
 * @brief A clip-space vertex that also carries the world point it came from.
 *
 * The world position is what a per-pixel shader wants: sample a procedural
 * texture at it, reflect a view ray about it, decide a material from its
 * altitude. Carrying it alongside the clip position means the clipper can
 * interpolate both with one parameter, so a clipped edge gets the right world
 * point and not one guessed from the screen.
 */
struct ShadedVertex {
    detail::ClipVertex clip{};
    core::f32 wx{0.0f};
    core::f32 wy{0.0f};
    core::f32 wz{0.0f};
};

/**
 * @brief Fills a triangle by CALLING a shader for every pixel it covers.
 *
 * The rasterizer had one fill: a flat colour decided per primitive. Everything
 * that varies within a surface — a texture, a reflection, a gradient along a
 * slope — had to be approximated by cutting the surface into more primitives, and
 * a lake ends up as a thousand flat quads that still cannot reflect anything,
 * because a reflection depends on the direction to the pixel, not to the quad.
 *
 * @c shader receives the interpolated world position and returns a packed colour.
 * Interpolation is PERSPECTIVE-CORRECT: the barycentric weights are divided
 * through by w, without which a texture on a surface seen at a grazing angle
 * swims as the camera turns — the classic artefact of affine interpolation, and
 * the reason the existing textured fill is only used on cube faces seen head-on.
 *
 * The fill is double-sided, like the rest of the terrain path: a heightfield is a
 * sheet, and half of it faces away from any camera.
 */
template <typename Shader>
inline void fillTriangleShaded(const RenderTarget &rt, const ShadedVertex &a, const ShadedVertex &b,
                               const ShadedVertex &c, Shader &&shader) noexcept;

/**
 * @brief Near-plane-clipped, per-pixel-shaded convex polygon.
 *
 * The shaded twin of @ref fillPolygonClipped, and it clips for the same reason:
 * the polygon under the walker's feet crosses the near plane every frame.
 */
template <typename Shader>
inline core::u32 fillPolygonShadedClipped(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                          const core::f32 *worldPoints, core::u32 count, Shader &&shader) noexcept;

/**
 * @brief Fills a convex polygon, clipping it against the near plane first.
 *
 * The reason this exists, and why every large primitive should go through it:
 * @c projectVertex only rejects vertices with @c w <= 0. A vertex a centimetre in
 * FRONT of the eye has a tiny positive w, survives that test, and comes out at
 * screen coordinates in the hundreds of thousands. The triangle it belongs to is
 * then stretched across the entire frame — on screen, a beam radiating from a
 * point, or a black wedge where ground should be. Both were blamed on the
 * geometry that produced them; the geometry was fine, and the projection was
 * being asked to divide by nearly zero.
 *
 * Dropping such a primitive instead is not a fix either: a terrain quad under the
 * walker's feet always straddles the near plane, so dropping it punches a hole in
 * the world exactly where the walker is looking. The only correct answer is to cut
 * the polygon along the plane and draw the part that is in front — Sutherland and
 * Hodgman in clip space, which is where the plane is a straight comparison rather
 * than a projection.
 *
 * An edge that crosses the plane is cut where it reaches w = epsilon, linearly in
 * clip space. When nothing crosses the plane, the common case, there is nothing to
 * clip, and the fill stays as cheap as it was.
 *
 * @param worldPoints Vertices, x/y/z triples, convex and in order.
 * @param count       Number of vertices (3 or 4 in practice).
 * @return Triangles filled; 0 when the polygon is entirely behind the eye.
 */
inline core::u32 fillPolygonClipped(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                    const core::f32 *worldPoints, core::u32 count, core::u32 colour) noexcept;

/**
 * @brief A vertical quad hanging from an edge segment: a LOD skirt.
 *
 * Adjacent chunks sampled at different strides do not agree along their shared
 * edge, and the disagreement shows as a crack straight through to the sky.
 * A curtain dropped from the edge does not make the two agree; it puts opaque
 * geometry behind the gap, which is all the eye needs.
 */
inline core::u32 drawSkirtQuad(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, core::f32 ax, core::f32 ay,
                               core::f32 az, core::f32 bx, core::f32 by, core::f32 bz, core::f32 drop,
                               core::u32 colour) noexcept;

/**
 * @brief Clears the render target to a background color and far depth.
 */
inline void clearTarget(const RenderTarget &rt, core::u32 background) noexcept;

/**
 * @brief Renders the canonical parity cube (12 depth-tested triangles, per-face
 *        flat colors) through a rotate+perspective camera into the target.
 *
 * The 12 triangles are the CCW front faces, two per face, in the order -Z, +Z,
 * -X, +X, -Y, +Y.
 *
 * @param rt            Render target (color + depth, width*height each).
 * @param rotationAngle Model Y-axis rotation, Fixed32 radians (authoritative).
 */
inline void renderCube(const RenderTarget &rt, math::Fixed32 rotationAngle) noexcept;

/**
 * @brief Renders the canonical cube with a texture sampled per face (each face
 *        mapped to the full [0,1] UV square, affine-interpolated, bilinear).
 */
inline void renderTexturedCube(const RenderTarget &rt, math::Fixed32 rotationAngle, const Texture &tex) noexcept;

/**
 * @brief Renders the canonical cube with classical per-face flat lighting
 *        (Lambert/Phong/Blinn-Phong) from one directional + one point light.
 */
inline void renderLitCube(const RenderTarget &rt, math::Fixed32 rotationAngle, ShadingModel model) noexcept;

/**
 * @brief Copies a render target into an EXISTING texture, allocating nothing.
 *
 * @ref targetToTexture returns a fresh texture, which is right for a one-shot fold
 * and wrong once per frame: at 240x150 that is 144 KiB of allocation and a copy
 * every time, inside the render path. Measured, it was enough to stall the world's
 * streaming — the reflection looked correct and the world stopped arriving.
 *
 * @return false if the texture's size does not match, so a caller cannot silently
 *         sample a stale image of the wrong shape.
 */
inline bool copyTargetToTexture(const RenderTarget &rt, Texture &out) noexcept;

/**
 * @brief Copies a render target's color buffer into a same-size Texture.
 */
[[nodiscard]] inline Texture targetToTexture(const RenderTarget &rt);

/**
 * @brief Renders a 2x2 multi-viewport composite: four lit cubes at distinct
 *        rotation angles, each into its own quadrant of the target.
 */
inline void renderMultiViewport(const RenderTarget &composite) noexcept;

/**
 * @brief Render-to-texture: renders a lit cube into an offscreen texture, then
 *        maps that texture onto a second cube drawn into the target.
 */
inline void renderToTextureCube(const RenderTarget &rt, math::Fixed32 angle) noexcept;

/**
 * @brief FNV-1a fold of the whole color buffer (cross-target signature).
 */
[[nodiscard]] inline core::u32 foldTarget(const RenderTarget &rt) noexcept;

} // namespace lpl::render

#    include "SoftwareRasterizer.inl"

#endif // LPL_RENDER_SOFTWARERASTERIZER_HPP
