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
    core::f32 bodyScale{0.35f}; /**< World units per unit of genome size. */

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

    /**
     * @name Underground
     *
     * @warning There is no `underground` flag here, and there was: it was fed from
     * engine::CharacterController::isEnclosed on the argument that the body already
     * resolves it, so the renderer should read rather than re-derive. That argument is
     * wrong, and wrong in a way that switched the whole feature off. The body is not
     * the EYE. Detach the camera and the eye is sixty cells behind the walker, often
     * inside a hill while the body stands in daylight — and the flag said "surface",
     * so the sky was drawn over an eye buried in rock and no cave was lit. What a
     * renderer needs is a fact about the eye, so @ref TerrainRenderer::drawStreamed
     * asks the streamer where the eye is. Non-authoritative: it picks a shading path,
     * nothing else.
     * @{
     */

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
    core::f32 lampReach{20.0f}; /**< How far the beam carries, in world units. Beyond it, only the ambient floor. */
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
    /** @} */

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

    /**
     * @brief Cull the resident set through a spatial hierarchy rather than linearly.
     *
     * Off is not a fallback, it is the right answer for a small set: below a few
     * dozen chunks the tree costs a rebuild to save a handful of box projections.
     * The threshold below is where the hierarchy starts paying, so the flag says
     * "use the tree when it is worth it", not "use the tree".
     */
    bool useSpatialCull{true};
    core::u32 spatialCullThreshold{48u}; /**< Resident chunks below which the linear pass wins outright. */
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

    void resetPhaseCounters() noexcept;

    /**
     * @brief Draws the streamed world around the camera.
     *
     * The eye never goes below the water: the terrain goes on under a lake, and an eye
     * that followed it there looks up at the underside of the ground sheet.
     *
     * Whether the eye is under rock is asked of the EYE and of nothing else (see
     * TerrainDrawParams for the flag this replaced and why a body's answer was the wrong
     * one), and it is NOT gated on the draw budget: it is a fact about the world, how much
     * cave geometry a host can afford is a fact about the host, and tying them made a
     * budget of zero mean "the sky is visible from inside a hill".
     *
     * The lamp you carry: darkness is two things and one lamp, being under rock and being
     * out at night. render::SunState::intensity is already zero at night, so "dark" is
     * read off the sky rather than defined a second time here — a second definition is a
     * thing to keep in step with the first, and it would drift at dusk. Its heading is in
     * the GROUND plane, renormalised: pitching the view down shortens the horizontal part
     * of forward, and a cone measured against an unnormalised axis narrows as you look at
     * your feet.
     *
     * The reflection probe is refreshed ONCE per frame, after the visible set is known and
     * before the ground is drawn. It was called twice for a while, which rendered the
     * mirrored world for nothing every fourth frame. And not at all underground: a
     * reflection probe renders the world upside down to put a sky in the water, and there
     * is no sky.
     *
     * Ground seams: index == size is legitimate and expected. It is the shared edge with
     * the next chunk, and drawing it is what closes the one-cell gap that used to run
     * around every chunk. The neighbour answers, through the same world height function
     * both chunks were generated from — so the two agree on that column exactly, which is
     * the whole reason a world-absolute sampler was chosen in the first place. Shade and
     * colour are CLAMPED rather than fetched from the neighbour. A seam column lit or
     * tinted by its own chunk is invisible; a seam column that is not drawn at all is a
     * slit to the horizon. The two are not the same kind of wrong.
     *
     * Doorways: the ones that fall in a chunk are gathered per chunk rather than tested
     * against every resident warren per cell. A warren spills across chunk borders, so the
     * aperture of a cave owned next door can land here, and the rectangle test that finds
     * it costs a handful of comparisons per chunk against nine hundred per cell. They are
     * the holes cave mouths are. Without them the surface passes straight across the
     * opening — a continuous hillside quad in front of a gallery you can walk into but
     * never see, which is the worst of the two failures because it looks like the
     * collision is broken. A quad is indexed by its LOW corner and spans a stride past it,
     * so the one that has to go is every quad TOUCHING a doorway cell — not the one
     * indexed at it. Matching the index alone is off by one in the adit's direction: the
     * cliff quad between the trench and the mouth is indexed at the trench cell when the
     * adit runs +Z, so the face the hole is meant to be in stayed drawn and the hole
     * opened a cell further in, where it removed nothing anybody could see. Measured from
     * the trench: 1175 cave triangles submitted, ZERO pixels of them visible.
     *
     * Skirts: none underground. A skirt is a curtain hung at a chunk border to put opaque
     * geometry behind an LOD crack — it exists so a seam does not show the SKY through it.
     * Under a hill there is no sky behind the crack, and what the curtain does instead is
     * stand a flat panel of surface-coloured ground across the gallery. Measured by
     * looking: it was the most conspicuous thing in the first frame rendered from inside
     * a cave.
     *
     * The sea, per chunk: one flat quad rather than a world-sized plane, because an
     * endless world has no width to span — and only where there IS water, or a chunk whose
     * bed never drops below sea level pays for a per-pixel pass to draw nothing. And only
     * over the cells that are actually under it. A chunk with one drowned corner used to
     * pay a per-pixel water pass over its whole extent, with the depth buffer throwing
     * almost all of it away — the most expensive way there is to draw nothing. One cell of
     * margin on each side, deliberately: clipped exactly to the submerged cells the
     * waterline is a hard edge on a cell boundary, and it stair-steps. The margin gives
     * the bed-depth fade — and the surf — a cell to happen in, and the ground there is
     * above the sea so the depth buffer rejects the pixels anyway. The lattice pitch is a
     * WORLD constant — the chunk span divided by the host's division count — not the
     * quad's extent divided by it. Quads differ in size because each is tightened to its
     * own drowned cells, so a per-quad division puts neighbouring vertices at different
     * world positions and the sheet tears along every chunk border.
     *
     * The river, per cell, and this is what the blue tiles were missing: they were
     * Lake-coloured cells of GROUND. Water is a surface standing in a carved bed at its own
     * height, with the reflection and the Fresnel every other water surface gets — and
     * with its swell running along its OWN current rather than with the wind, because a
     * river does not care which way the air goes. Near rings only: a river cell is a metre
     * across; past the second ring it is sub-pixel, and a per-pixel shader over sub-pixel
     * quads is pure cost. A river does not heave, so it keeps the shading and drops the
     * swell — which also drops the tessellated path for it, by the same condition. Flat
     * water still moves, at the wind's own rate. This was the other half of the "no
     * animation between two flows": with the speed folded into the direction, a slow
     * surface had its WAVELENGTH stretched instead.
     *
     * @warning ONE river surface for the whole chunk, not one per cell. `WaterParams` is
     * consumed per PIXEL but supplied per QUAD, so anything that varies from cell to cell
     * makes the ripple pattern restart at every cell border — which is precisely the "you
     * can see the tile boundaries" the screenshots showed, and it survived making the
     * GEOMETRY continuous because it was never geometry. The cost is stated rather than
     * hidden: a river that turns inside one chunk ripples the same way along its whole
     * length there. A drift that varied per pixel would need the wave to take a direction
     * FIELD rather than a constant, which is a change to render::sampleWater and not to
     * this loop.
     *
     * @warning THE CORNERS ARE SHARED, and that is the whole difference between a river
     * and a stack of plates. Each cell used to get a flat quad at its OWN bed height, so
     * two neighbours a metre apart in bed height drew two surfaces a metre apart — steps,
     * never a sheet, and nothing a current could run along. A corner's level is a function
     * of the CORNER's position and nothing else: the lowest of the four cells that touch
     * it, plus the fill. Two adjacent cells share two corners, so they agree there by
     * construction — the same argument the terrain uses for its own seams. The lowest
     * rather than the average because water sits in the channel, not on its banks.
     *
     * Below the sea a river cell IS the sea, already drawn: two surfaces at the same place
     * fight over the depth buffer and flicker. And A WATERFALL IS NOT A SHEET EITHER.
     * Where the four corners disagree by more than the water is deep, the channel is
     * falling rather than flowing, and a quad spanning it is a ramp of water hanging in
     * the air. Measured on the corners rather than on the neighbouring beds: it is the
     * same question asked of the surface that is actually drawn. The bed depth comes from
     * the CARVED bed of the very cell: the channel is what makes the water look like water
     * rather than a wet stripe, and asking the world height function would answer with the
     * bed of whichever cell the point rounded into.
     *
     * @warning The bed depth is bilinear from the four SHARED corners, not `top - bed` —
     * which is one number for the whole cell and therefore paints the body colour in
     * cell-sized blocks. It is consumed per pixel and it drives both the shallow-to-deep
     * mix and the shore foam, so a constant here is a visible tile edge however continuous
     * the geometry is.
     *
     * Landmarks come before the props and after the ground of their own chunk, for the
     * same reason the boulders do: they stand ON that ground, so the depth buffer has to
     * hold the hill first. Only the chunk that OWNS a site draws it — the ground of it was
     * carved by every chunk in reach, and drawing it there too would put one village on
     * screen once per chunk that can see it. A building is two boxes rather than one, and
     * it is the cheapest thing that stops a village reading as a crate yard: walls, then a
     * roof block oversized by a quarter cell so it overhangs. A single box has no eaves and
     * no roof colour, because drawBox lights a face from its normal and cannot know that
     * this face is thatch.
     *
     * A cave mouth's dark quad is drawn only where no real warren stands behind it. It was
     * a STAND-IN from when a mouth led nowhere: a flat panel stood at the back of the shelf
     * so the shelf read as an opening. With a warren behind it, it is a fake in front of
     * the real thing — a black rectangle stuck on a hillside, which is exactly what it
     * looks like. The opening stands at the UPHILL edge of the shelf and faces downhill,
     * which is the only orientation that reads as going INTO the hill: the shelf was cut
     * out of the slope, so the rock is behind it. It is perpendicular to the facing, so the
     * mouth is a wall across the slope rather than a sliver seen edge-on, and it is a
     * DOORWAY in metres, not a fraction of the hill: both numbers were multiples of the
     * shelf depth, so when that depth started tracking the world's relief the opening
     * became a seventeen-metre black wall. A grazed plant is not drawn: the herd ate it,
     * and the map says so.
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
     * The bounded grid is ONE patch, centred on the origin: the same module call the
     * streamed world makes per chunk, which is the point of having it. A bounded world
     * with no sea used to pay a full-grid per-pixel water pass for a surface entirely
     * behind its own terrain, hence @ref TerrainDrawParams::boundedHasSea.
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
    /**
     * @brief Refreshes the reflection probe from the visible set.
     *
     * Only when there is water on screen to reflect into: the probe is a whole second
     * render of the world, and over an inland world every frame of it was thrown away.
     * That is asked of the VISIBLE set rather than of the resident one: a lake four
     * chunks behind the camera is resident, is not drawn, and has nothing to reflect.
     *
     * The probe draws terrain only, flat-shaded, at a coarse stride: at probe
     * resolution a tree is two pixels, and the pass exists to put the MOUNTAIN in the
     * water. Index == size is legitimate and expected — it is the shared edge with the
     * next chunk, which is why the ground pass answers it from the world height
     * function. This pass did NOT: it indexed the grid directly, so every chunk's far
     * edge read one row past the end of its own heightfield. On the target there are no
     * bounds checks, so it quietly mirrored whatever followed the block in the heap; on
     * the host it aborts, which is how it was found. The colour is clamped rather than
     * fetched, exactly as the ground pass clamps its own: at probe resolution a seam
     * column tinted by its own chunk is invisible, and one that is not drawn is a slit
     * to the horizon.
     */
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
     *
     * Two regions are worth walking, not one, and the second is the whole reason a
     * doorway reads as a doorway. Around the EYE, because that is what you see when you
     * are inside; and around the DOORWAY, because that is what you see through when you
     * are outside. With only the first, a warren twelve cells past the eye emitted
     * nothing and the hole in the hillside showed the background — the cave was there,
     * collidable, and invisible from the one place a player first meets it.
     *
     * ONE of the two, never their union, and that is a cost bound rather than a
     * simplification: merging them into a bounding box is fine while you stand in the
     * doorway and spans the whole warren the moment you are sixty cells off it, which is
     * eleven thousand voxels per cave with three caves resident. Inside, the eye's own
     * window is what you can see; outside, you can see nothing of the interior EXCEPT
     * through the door, so the door's window is the whole of it. Either way the walk is
     * one box of a bounded side. Only a doorway the eye could actually be looking at
     * counts. The bound is generous — a mouth is a landmark and you spot it before you
     * reach it — and it is a distance rather than a frustum test because a frustum here
     * would have to be recomputed per warren for a handful of boxes.
     *
     * Outside the ARRAY is rock. A warren is a hole in a hill, not a building standing
     * in air: answering "empty" there would wrap the whole volume in a shell of faces
     * nobody can ever be on the outside of. And a column with no rock over it is
     * TERRAIN, whatever its voxels say, so a face pointing at one is a face pointing at
     * open air — which is exactly what the doorway is made of. Reading the volume alone
     * would leave the mouth walled in from the outside.
     *
     * The faces are lit from the EYE, not from the sun: underground the sun is behind a
     * hill, so a lambert against it leaves every face equally black and the cave reads
     * as a flat silhouette. And as a SPOT rather than a glow — the beam's axis is the
     * view direction, so turning your head lights what you turned towards, which is what
     * makes a passage readable. A bulb at the eye lights everything around you the same
     * and gives a gallery no direction to go in. How far off the beam's axis a face sits
     * comes from the vector FROM the eye, the negation of the one to it, so the dot
     * against forward is a cosine straight away — no angle is ever formed, which is what
     * keeps this off any transcendental. The cone is squared, so the edge of the beam
     * softens instead of ending on a line. The colour lies between unlit rock and rock
     * under a torch, rather than being a single colour scaled down: scaling can only
     * darken, so one tint means the brightest thing in the cave is still as dark as its
     * darkest. The distance term is the surface's aerial perspective, already pointed at
     * the cave's own darkness by TerrainSurface::beginCaveFrame — one fade, not a second
     * one competing with it.
     */
    core::u32 drawWarrens(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                          const TerrainStreamer &streamer, const render::CameraBasis &basis,
                          const TerrainDrawParams &params, const render::SunState &sun) const;

    /**
     * @brief The herd, as billboards standing on the ground the eye is looking at.
     *
     * Draws what is IN THE WORLD, not what a container remembers. Everything a body
     * needs to be drawn — where it is, what kind it is, how big it is — is a component
     * now, so this pass no longer takes an ecology::Herd at all: the renderer and the
     * simulation stopped having to agree about a second list.
     *
     * It reads the side the WRITER wrote. A creature's components are written on the
     * back buffer, by the spawn and then by the locomotion system, and the buffer swap
     * is a phase callback the Engine installs after Physics — so a world whose systems
     * are all PrePhysics never publishes a front buffer at all. Reading the front drew
     * every animal at the origin with a size of zero, which looks exactly like no
     * animals.
     *
     * A body stands on the ground that is DRAWN, not the noise: the two differ by exactly
     * what erosion moved, and the herd hung in the air above the ridges it lowered. It
     * is a billboard rather than a box: at this scale a creature is a handful of pixels,
     * and six lit faces cost six times as much to say the same.
     */
    template <typename GroundAt>
    core::u32 drawHerd(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp, const ecs::Registry &registry,
                       const TerrainDrawParams &params, GroundAt &&groundAt) const;

    /**
     * @brief Chooses the visible chunks, through the hierarchy or linearly.
     *
     * The two paths drive the SAME @c consider / @c endSelect, so the frustum test,
     * the ring rule and the ordering exist once. What the tree changes is only which
     * chunks get tested at all.
     *
     * The tree's bounds are the resident set's own extent, recomputed each frame: a
     * streamed world slides, and bounds pinned at the origin would spend the whole key
     * range on space that holds nothing. The node test is the SAME projection the chunks
     * use, applied to a node's box. A view volume is not an AABB, so querying its
     * bounding box would hand back most of the world whenever the camera is pitched;
     * rejecting a node instead prunes a whole subtree, which is the only thing a
     * hierarchy is worth having for.
     */
    void selectChunks(const math::Mat4<core::f32> &mvp, const render::CameraBasis &basis, core::u32 targetWidth,
                      core::u32 targetHeight, const render::ChunkedViewParams &view, const TerrainDrawParams &params,
                      core::i32 focusChunkX, core::i32 focusChunkZ, TerrainStreamer &streamer);

    /**
     * @brief The cycle counter, or zero when no clock was given.
     */
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

#    include "TerrainRenderer.inl"

#endif // LPL_ENGINE_TERRAIN_RENDERER_HPP
