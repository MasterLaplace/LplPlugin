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
#    include <lpl/engine/PropLibrary.hpp>
#    include <lpl/engine/TerrainStreamer.hpp>
#    include <lpl/engine/TerrainSurface.hpp>
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
     * @brief Draws the streamed world around the camera.
     *
     * @param palette (BiomeId) -> packed colour.
     * @param groundAt (worldX, worldZ) -> f32, used for the water's depth and for
     *                 standing the bodies on the ground that is actually DRAWN.
     */
    template <typename Palette, typename GroundAt>
    core::u32 drawStreamed(const render::RenderTarget &rt, const render::OrbitCamera &camera,
                           TerrainStreamer &streamer, TerrainSurface &surface, const PropLibrary &props,
                           const ecology::Herd &herd, const TerrainDrawParams &params, core::u32 frame,
                           Palette &&palette, GroundAt &&groundAt);

    /**
     * @brief Draws the bounded world: one patch, one sea quad, its plants and herd.
     *
     * @param plants Standing vegetation, in grid cells; the caller owns the list.
     */
    template <typename Palette, typename HeightAt, typename ColourAt, typename GroundAt>
    core::u32 drawBounded(const render::RenderTarget &rt, const render::OrbitCamera &camera, TerrainSurface &surface,
                          const PropLibrary &props, const ecology::Herd &herd, core::u32 gridWidth,
                          core::u32 gridDepth, const ecology::PlantCell *plants, core::u32 plantCount,
                          const TerrainDrawParams &params, Palette &&palette, HeightAt &&heightAt,
                          ColourAt &&colourAt, GroundAt &&groundAt);

private:
    /** @brief Refreshes the reflection probe from the visible set. */
    template <typename Palette>
    void refreshProbe(TerrainStreamer &streamer, TerrainSurface &surface, const render::CameraBasis &basis,
                      core::u32 frame, const TerrainDrawParams &params, Palette &&palette);

    /** @brief The herd, as billboards standing on the ground the eye is looking at. */
    template <typename GroundAt>
    core::u32 drawHerd(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp, const ecology::Herd &herd,
                       const TerrainDrawParams &params, GroundAt &&groundAt) const;

    render::ChunkedTerrainView _view;
    core::u32 _triangles{0u};
};

} // namespace lpl::engine

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/engine/TerrainRenderer.inl>

#endif // LPL_ENGINE_TERRAIN_RENDERER_HPP
