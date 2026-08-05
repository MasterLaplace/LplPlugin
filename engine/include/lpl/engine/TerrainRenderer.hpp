/**
 * @file TerrainRenderer.hpp
 * @brief One frame of a heightfield world: sky, ground, water, props, bodies.
 *
 * The pass itself, as an object a world OWNS rather than a base class it inherits.
 * It reads the pieces it draws — the streamer's chunks, the surface's light, the
 * prop library's meshes, the herd's bodies — and it decides nothing about them: the
 * palette, the ground height and the water's bed arrive as callbacks, because those
 * are the three things that differ between two games on the same terrain.
 *
 * Two orders in here are load-bearing:
 *
 *  - the reflection probe is refreshed AFTER the visible set is chosen and BEFORE
 *    the ground is drawn. It reflects what the frame will draw; refreshing it twice
 *    (which this code did for a while) renders the mirrored world for nothing.
 *  - the props are drawn AFTER the ground of their own chunk, so a boulder tests
 *    against a depth buffer that already holds the hill it stands on.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_TERRAIN_RENDERER_HPP
#    define LPL_ENGINE_TERRAIN_RENDERER_HPP

#    include <lpl/ai/Personality.hpp>
#    include <lpl/ecology/Herd.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/engine/PropLibrary.hpp>
#    include <lpl/engine/TerrainStreamer.hpp>
#    include <lpl/engine/TerrainSurface.hpp>
#    include <lpl/physics/Octree.hpp>
#    include <lpl/platform/IClockBackend.hpp>
#    include <lpl/render/ChunkedTerrainView.hpp>
#    include <lpl/render/HeightfieldPatch.hpp>
#    include <lpl/render/OrbitCamera.hpp>

namespace lpl::engine {

/**
 * @struct TerrainDrawParams
 * @brief What a frame needs to know about the world it draws.
 */
struct TerrainDrawParams {
    core::u32 chunkSize{24u};
    core::u32 lodRings{3u};
    core::f32 seaLevel{-1.0f};
    core::f32 ambient{0.28f};
    core::f32 skirtDrop{6.0f};
    core::f32 chunkCentreY{8.0f};
    core::f32 chunkHalfHeight{72.0f};
    core::f32 nearPlane{0.4f};
    core::f32 farPlane{600.0f};
    core::f32 fovRadians{1.04719755f};
    core::u32 grazerTint{0x00D0A852u};
    core::u32 hunterTint{0x00C03028u};
    core::f32 bodyScale{0.35f}; ///< World units per unit of genome size.

    /**
     * @brief Cull the resident set through a spatial hierarchy rather than linearly.
     *
     * Off is not a fallback, it is the right answer for a small set: below a few
     * dozen chunks the tree costs a rebuild to save a handful of box projections.
     * The threshold below is where the hierarchy starts paying, so the flag says
     * "use the tree when it is worth it", not "use the tree".
     */
    /**
     * @brief Anchor the view at a stated height instead of at the ground.
     *
     * The camera normally sits a fixed height above the terrain under its focus,
     * which is right for an orbit and wrong for a BODY: a jumping player is not on
     * the ground, and a view that queried the terrain would stay planted while the
     * character rose. When this is set the caller supplies the anchor, because only
     * the caller knows it is simulating something that leaves the floor.
     */
    bool useFocusHeight{false};
    core::f32 focusHeight{0.0f};

    bool useSpatialCull{true};
    core::u32 spatialCullThreshold{48u}; ///< Resident chunks below which the linear
                                         ///< pass wins outright.
};

/**
 * @class TerrainRenderer
 * @brief Draws a streamed or a bounded heightfield world.
 */
class TerrainRenderer {
public:
    [[nodiscard]] const render::ChunkedTerrainView &view() const noexcept { return _view; }
    [[nodiscard]] core::u32 lastTriangles() const noexcept { return _triangles; }

    /**
     * @brief Gives the pass a clock, so it can say WHERE a frame goes.
     *
     * Optional: with no clock every counter stays at zero and the pass is unchanged.
     * It exists because "the scene costs seventy-five percent of the frame" is not
     * an actionable statement — sky, ground, water, props and bodies are five
     * different fixes, and guessing which one dominates is how an afternoon goes
     * into the wrong one. Measured twice already on this project, wrong twice.
     */
    void setClock(platform::IClockBackend *clock) noexcept { _clock = clock; }

    [[nodiscard]] core::u64 skyCycles() const noexcept { return _skyCycles; }
    [[nodiscard]] core::u64 groundCycles() const noexcept { return _groundCycles; }
    [[nodiscard]] core::u64 waterCycles() const noexcept { return _waterCycles; }
    [[nodiscard]] core::u64 propCycles() const noexcept { return _propCycles; }
    [[nodiscard]] core::u64 herdCycles() const noexcept { return _herdCycles; }

    void resetPhaseCounters() noexcept
    {
        _skyCycles = 0u;
        _groundCycles = 0u;
        _waterCycles = 0u;
        _propCycles = 0u;
        _herdCycles = 0u;
    }

    /**
     * @brief Draws the streamed world around the camera.
     *
     * @param palette (BiomeId) -> packed colour.
     * @param groundAt (worldX, worldZ) -> f32, used for the water's depth and for
     *                 standing the bodies on the ground that is actually DRAWN.
     */
    template <typename Palette, typename GroundAt>
    core::u32 drawStreamed(const render::RenderTarget &rt, const render::OrbitCamera &camera, TerrainStreamer &streamer,
                           TerrainSurface &surface, const PropLibrary &props, const ecs::Registry &registry,
                           const TerrainDrawParams &params, core::u32 frame, Palette &&palette, GroundAt &&groundAt);

    /**
     * @brief Draws the bounded world: one patch, one sea quad, its plants and herd.
     *
     * @param plants Standing vegetation, in grid cells; the caller owns the list.
     */
    template <typename Palette, typename HeightAt, typename ColourAt, typename GroundAt>
    core::u32 drawBounded(const render::RenderTarget &rt, const render::OrbitCamera &camera, TerrainSurface &surface,
                          const PropLibrary &props, const ecs::Registry &registry, core::u32 gridWidth, core::u32 gridDepth,
                          const ecology::PlantCell *plants, core::u32 plantCount, const TerrainDrawParams &params,
                          Palette &&palette, HeightAt &&heightAt, ColourAt &&colourAt, GroundAt &&groundAt);

private:
    /** @brief Refreshes the reflection probe from the visible set. */
    template <typename Palette>
    void refreshProbe(TerrainStreamer &streamer, TerrainSurface &surface, const render::CameraBasis &basis,
                      core::u32 frame, const TerrainDrawParams &params, Palette &&palette);

    /** @brief The herd, as billboards standing on the ground the eye is looking at. */
    template <typename GroundAt>
    core::u32 drawHerd(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp, const ecs::Registry &registry,
                       const TerrainDrawParams &params, GroundAt &&groundAt) const;

    /**
     * @brief Chooses the visible chunks, through the hierarchy or linearly.
     *
     * The two paths drive the SAME @c consider / @c endSelect, so the frustum test,
     * the ring rule and the ordering exist once. What the tree changes is only which
     * chunks get tested at all.
     */
    void selectChunks(const math::Mat4<core::f32> &mvp, const render::CameraBasis &basis, core::u32 targetWidth,
                      core::u32 targetHeight, const render::ChunkedViewParams &view, const TerrainDrawParams &params,
                      core::i32 focusChunkX, core::i32 focusChunkZ, TerrainStreamer &streamer);

    /// The cycle counter, or zero when no clock was given.
    [[nodiscard]] core::u64 now() const noexcept { return _clock != nullptr ? _clock->timestampCounter() : 0u; }

    platform::IClockBackend *_clock{nullptr};
    core::u64 _skyCycles{0u};
    core::u64 _groundCycles{0u};
    core::u64 _waterCycles{0u};
    core::u64 _propCycles{0u};
    core::u64 _herdCycles{0u};

    render::ChunkedTerrainView _view;
    /**
     * @brief The resident chunks, indexed for culling.
     *
     * Built with a leaf capacity of FOUR, not the broad-phase default of thirty-two:
     * a culler tests every object in a surviving leaf individually, so a large leaf
     * means the hierarchy prunes nothing. At thirty-two, nineteen chunks made one
     * node and the traversal was the linear scan it was meant to replace.
     */
    physics::Octree _chunkIndex{math::AABB<math::Fixed32>{}, 4u};
    core::u32 _triangles{0u};
};

} // namespace lpl::engine

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/engine/TerrainRenderer.inl>

#endif // LPL_ENGINE_TERRAIN_RENDERER_HPP
