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
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/ecology/Herd.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/engine/PropLibrary.hpp>
#    include <lpl/engine/TerrainStreamer.hpp>
#    include <lpl/engine/TerrainSurface.hpp>
#    include <lpl/physics/Octree.hpp>
#    include <lpl/platform/IClockBackend.hpp>
#    include <lpl/render/Box.hpp>
#    include <lpl/render/ChunkedTerrainView.hpp>
#    include <lpl/render/HeightfieldPatch.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/std/cmath.hpp>

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
     * @brief How far the water stands above a river's carved bed, in world units.
     *
     * Zero — the default — draws no river water at all, which is what this renderer did:
     * a river cell was a Lake-coloured cell of TERRAIN, so the blue on screen was ground
     * that happened to be blue. It had no reflection, no Fresnel, no depth and no
     * current, and no amount of shading fixes that because there was no water there.
     *
     * Necessarily the same number the generator carved with — `riverDepth * riverFill`
     * of procgen::ChunkTerrainRule — or the surface floats over its bed or sinks under
     * it. procgen::endlessPlanFromRecipe derives both from one description, and this
     * comes from the same plan.
     */
    core::f32 riverSurfaceRise{0.0f};

    /**
     * @brief Whether a BOUNDED field dips below the sea level at all.
     *
     * The streamed path answers this per chunk from what it generated; a bounded field
     * is one patch, so the answer is one bit and its owner is whoever built the field.
     * Defaults to true because the two mistakes are not symmetric: drawing a sea nobody
     * needed costs a pass, and skipping one that was needed loses an ocean.
     */
    bool boundedHasSea{true};

    /**
     * @brief What a cave mouth and a village are made of.
     *
     * Content, like the creature tints beside them: a world of whitewashed houses and one
     * of black timber are two worlds, and neither is a host budget. The mouth is very dark
     * but NOT black — a black opening in a heightfield reads as a hole in the render, and
     * a hint of warmth in it reads as depth.
     */
    core::u32 caveMouthTint{0x00120E0Cu};
    core::u32 buildingTint{0x00B4A183u};
    core::u32 roofTint{0x00714634u};

    /**
     * @brief How far a cave mouth's shelf was cut, so the opening can stand in it.
     *
     * Necessarily the same number the generator carved with —
     * procgen::ChunkTerrainRule::caveMouthDrop — or the opening floats above its own floor
     * or is buried under it. Same hazard as @ref riverSurfaceRise, and derived from the
     * same plan for the same reason.
     */
    core::f32 caveMouthDrop{2.0f};

    /**
     * @brief How tall the dark opening stands above the shelf floor, in metres.
     *
     * Human scale and its own number, rather than a multiple of the shelf depth: tying the
     * two meant that calibrating the shelf against the world's relief silently resized the
     * door with it.
     */
    core::f32 mouthHeight{2.8f};

    // ── Underground ─────────────────────────────────────────────────────────

    // ⚠ There is no `underground` flag here, and there was: it was fed from
    // engine::CharacterController::isEnclosed on the argument that the body already
    // resolves it, so the renderer should read rather than re-derive. That argument is
    // wrong, and wrong in a way that switched the whole feature off. The body is not
    // the EYE. Detach the camera and the eye is sixty cells behind the walker, often
    // inside a hill while the body stands in daylight — and the flag said "surface",
    // so the sky was drawn over an eye buried in rock and no cave was lit. What a
    // renderer needs is a fact about the eye, so @ref drawStreamed asks the streamer
    // where the eye is. Non-authoritative: it picks a shading path, nothing else.

    /**
     * @brief What a cave is made of, and what it fades into.
     *
     * Very dark but NOT black, for the reason the mouth tint gives: black reads as a
     * hole in the render, and a hint of warmth in it reads as depth. The fog density
     * is what makes a lamp a lamp — its reciprocal is roughly how far you see.
     */
    core::u32 caveRockTint{0x00595049u};
    core::u32 caveDarkTint{0x00080706u};
    core::f32 caveFogDensity{0.11f};

    /**
     * @brief Cells around the eye whose cave geometry is drawn. Zero draws none.
     *
     * A host budget, in the shape engine::Config::waterTessellation already has: a
     * warren is thirteen thousand voxels and a body underground can see a few hundred
     * of them, so the cost is set by how far the light reaches rather than by how big
     * the cave is. The first knob to turn down.
     */
    core::u32 caveDrawRadius{0u};

    /**
     * @brief The lamp you carry, as a cone rather than a glow.
     *
     * A cave has no sun in it, so something has to light it or the geometry is drawn and
     * invisible. What was here first was closer to a bare bulb at the eye — brightness
     * from the angle between a face and the view — which lights everything around you
     * equally and gives a cave no direction to look IN.
     *
     * A spot has an axis, and the axis is where you are looking: turn your head and the
     * beam goes with it, which is the thing that makes a passage readable. It belongs to
     * the EYE and not to an entity — a @c ComponentId::Light with exactly one instance
     * and one consumer would be the orphan this repository keeps deleting — but the
     * numbers are content, so they live here beside the rock colour.
     *
     * @c coneInner is the cosine of the half-angle of the full-strength core and
     * @c coneOuter of the edge where it has fallen to nothing; cosines rather than
     * angles because comparing a dot product against one is a multiply, and getting an
     * angle out of it is an arccosine nothing in a kernel path may call.
     */
    core::f32 lampConeInner{0.70f};
    core::f32 lampConeOuter{0.10f};
    /// How far the beam carries, in world units. Beyond it, only the ambient floor.
    core::f32 lampReach{20.0f};
    /**
     * @brief What a surface outside the beam still receives.
     *
     * Not zero, deliberately: a cave lit by the cone alone is a bright disc in absolute
     * blackness, and you cannot tell a wall a step to your left from an open passage.
     * A little bounce is what makes the shape of a gallery legible.
     */
    core::f32 lampAmbient{0.22f};

    /**
     * @brief What the lamp is bright enough to make of a rock face.
     *
     * Separate from @ref caveRockTint, which is what unlit rock is. The first pass used
     * one colour for both and the result was a cave you could not play in: a torch on
     * stone a couple of metres away reads as light grey-brown, and modulating a dark
     * rock colour by a beam can only ever make it darker. Measured by looking — the lit
     * patches came out around thirty of two hundred and fifty-five, which is a shape you
     * can just about infer rather than a surface you can walk along.
     */
    core::u32 lampLitTint{0x009A8C7Au};

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
                          const PropLibrary &props, const ecs::Registry &registry, core::u32 gridWidth,
                          core::u32 gridDepth, const ecology::PlantCell *plants, core::u32 plantCount,
                          const TerrainDrawParams &params, Palette &&palette, HeightAt &&heightAt, ColourAt &&colourAt,
                          GroundAt &&groundAt);

private:
    /** @brief Refreshes the reflection probe from the visible set. */
    template <typename Palette>
    void refreshProbe(TerrainStreamer &streamer, TerrainSurface &surface, const render::CameraBasis &basis,
                      core::u32 frame, const TerrainDrawParams &params, Palette &&palette);

    /**
     * @brief The cave around the eye: exposed rock faces, within the lamp's reach.
     *
     * Walked from the volume rather than meshed once and cached, because the window is
     * the part the eye can see and it moves every frame. procgen::forEachVoxelFace is
     * the same walk procgen::appendVoxelFaces uses — one enumeration, two sinks, and
     * neither of them is a sixth copy of the six faces of a cube.
     */
    core::u32 drawWarrens(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                          const TerrainStreamer &streamer, const render::CameraBasis &basis,
                          const TerrainDrawParams &params, const render::SunState &sun) const;

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
