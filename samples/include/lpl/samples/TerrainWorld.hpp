/**
 * @file TerrainWorld.hpp
 * @brief A living heightfield world: bounded or streamed, drawn, walked and grazed.
 *
 * The engine's answer to "I want a world you can stand in". Everything a game of
 * that shape needs is here, and every piece of it was written inside a sample
 * first: build a world from a seed, stream it in chunks around a walker, light and
 * shade its surface, grow trees and scatter boulders on it, put water in its
 * valleys, and run a population of animals over it.
 *
 * The class is final and has no hook to override: a game takes it as it is,
 * and what it chooses is settings. The settings have owners: the terrain and its
 * scatter are procgen::WorldRecipe's (a .lplscene serialises it), the streaming
 * radii are procgen::StreamingParams', the presentation and the memory ceiling are
 * engine::Config's through engine::HostProfile.
 *
 * @warning Everything drawn is float and none of it flows back: the authoritative
 *          state is the Fixed32 grids and the creature positions; the projection
 *          and the lighting read them and produce pixels.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_TERRAINWORLD_HPP
#    define LPL_SAMPLES_TERRAINWORLD_HPP

#    include <lpl/ai/Personality.hpp>
#    include <lpl/ai/ScentWindow.hpp>
#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/ai/Swarm.hpp>
#    include <lpl/core/Log.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/ecology/Herd.hpp>
#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/ecology/Populations.hpp>
#    include <lpl/ecology/Vegetation.hpp>
#    include <lpl/engine/CharacterController.hpp>
#    include <lpl/engine/Engine.hpp>
#    include <lpl/engine/ITerrainQuery.hpp>
#    include <lpl/engine/LivingLayer.hpp>
#    include <lpl/engine/PropLibrary.hpp>
#    include <lpl/engine/TerrainRenderer.hpp>
#    include <lpl/engine/TerrainStreamer.hpp>
#    include <lpl/engine/TerrainSurface.hpp>
#    include <lpl/engine/ViewProfile.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/engine/systems/CreatureSystems.hpp>
#    include <lpl/image/Font8x16.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedMath.hpp>
#    include <lpl/math/Random.hpp>
#    include <lpl/platform/IPlatform.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/Botany.hpp>
#    include <lpl/procgen/ChunkResidency.hpp>
#    include <lpl/procgen/ChunkTerrain.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Climate.hpp>
#    include <lpl/procgen/EndlessPlan.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>
#    include <lpl/procgen/Streaming.hpp>
#    include <lpl/procgen/ValueNoise.hpp>
#    include <lpl/procgen/WorldBuilder.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>
#    include <lpl/procgen/WorldSnapshot.hpp>
#    include <lpl/render/ChunkedTerrainView.hpp>
#    include <lpl/render/CommandBuffer.hpp>
#    include <lpl/render/Foliage.hpp>
#    include <lpl/render/HeightfieldPatch.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/MipTexture.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/Overlay.hpp>
#    include <lpl/render/Pbr.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/Reflection.hpp>
#    include <lpl/render/Revolve.hpp>
#    include <lpl/render/Scatter.hpp>
#    include <lpl/render/Sky.hpp>
#    include <lpl/render/SkyDome.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>
#    include <lpl/render/Topology.hpp>
#    include <lpl/render/Water.hpp>
#    include <lpl/samples/ReliefBlob.hpp>

namespace lpl::samples {

/**
 * @brief Builds a field over the checked-in window of real ground.
 *
 * @warning **A free function of the recipe, not a method, because that is what it IS.** Which ground
 * the blob describes and where it lands depends on the recipe and on nothing about a world instance
 * -- and a world cannot be driven from a test without an `engine::WorldContext`, so a method here
 * would have been logic reachable only by booting. Extracting it is what let the join below be
 * measured rather than asserted from the outside.
 *
 * @warning **The projection is built here rather than carried in the blob**: a blob is an array of
 * metres, a projection is a decision about where those metres go. Its sea level comes from the
 * recipe, so elevation zero lands exactly where the biome classifier thinks the coast is. Those two
 * disagreeing is the pattern this repository has already paid for three times.
 *
 * The blob names the tile's SOUTH-west corner; a projection wants the NORTH-west one, and a
 * tile is one degree tall. One addition, done once, rather than a mirror in the sampler.
 *
 * The vertical scale is derived from the recipe rather than picked: the measured range is
 * several kilometres of rock, and at true scale it would tower over a world whose noise is tens
 * of units tall.
 *
 * @warning **At its TRUE projection cells, starting at the origin, and NOT centred on it.** A
 * centred window spans negative columns, and a closed world folds those to the far side of the
 * planet -- so the survey's western half would be cut off and reappear an ocean away, which
 * reads as a coastline in the wrong place rather than as an error. The blob is the north-west
 * corner of its tile, and the projection's origin is that same corner, so cells 0..side-1 are
 * exactly where this ground really is.
 *
 * Every edge of the field is exposed: a lone survey faces invented country on every side.
 *
 * @param recipe        The world being built.
 * @param noiseAmplitude The recipe's terrain amplitude, already scaled for walking.
 * @return The field, at its true projection cells from the world origin, so a body starting
 *         there stands on measured ground.
 */
[[nodiscard]] inline math::ReliefField makeRealReliefField(const procgen::WorldRecipe &recipe,
                                                           core::f32 noiseAmplitude) noexcept;

/**
 * @class TerrainWorld
 * @brief A world on a heightfield: generated, streamed, lit, walked and grazed.
 *
 * Also the ground the creature systems stand on: it implements
 * engine::ITerrainQuery, because the two questions an animal asks the terrain —
 * may I stand here, is there anything to eat here — are the two only a world can
 * answer, and answering them differs completely between the bounded grid and the
 * streamed one.
 */
class TerrainWorld final : public engine::World, public engine::ITerrainQuery {
public:
    /**
     * @brief The parameters this world generates its ground from.
     *
     * @warning Exposed so a test can ask whether the survey is actually READ, which no assertion
     * about rendering can answer: a world that carried a relief field and ignored it would draw
     * exactly the same picture as one that consulted it, and both would look like terrain.
     *
     * @return The chunk parameters, survey included.
     */
    [[nodiscard]] const procgen::ChunkParams &chunkParamsForTest() const noexcept { return _chunkParams; }
    /**
     * @brief Builds the viewer around a recipe.
     *
     * The recipe is the game. It arrives decoded from a `.lplpak` cartridge — a
     * GRUB module sitting next to the kernel — or from the reference pack compiled
     * into the image when no cartridge is present. Either way the viewer runs
     * @ref procgen::applyRecipe, the same one function the parity gate and the
     * editor run, so what is on screen is what the `.lplscene` document says and
     * not a second pipeline that happens to look similar.
     */
    explicit TerrainWorld(const procgen::WorldRecipe &recipe) noexcept : _recipe(recipe), _seed(recipe.seed) {}

    /**
     * @brief Builds the viewer around a world AND the ecosystem declared on it.
     *
     * Both come from the same cartridge. Before this the `.lplscene` could
     * describe a valley down to its erosion iteration count and had no way to say
     * what grazed in it — the food web was compiled into this header.
     */
    TerrainWorld(const procgen::WorldRecipe &recipe, const ecology::LivingRecipe &living) noexcept
        : _recipe(recipe), _livingRecipe(living), _seed(recipe.seed)
    {
    }

    /**
     * @brief Builds the viewer around a world, its ecosystem, AND its look.
     *
     * The third thing a cartridge can now say. Before this the `.lplscene` could
     * describe a valley and what grazed in it, and every one of them came out under
     * the same blue midday sky, because the palette and the sun were constants in
     * this header.
     */
    TerrainWorld(const procgen::WorldRecipe &recipe, const ecology::LivingRecipe &living,
                 const engine::ViewProfile &view) noexcept
        : _recipe(recipe), _livingRecipe(living), _view(view), _seed(recipe.seed)
    {
    }

    TerrainWorld() = default;
    /**
     * @brief Reads the host's settings, generates the world, and registers the creature systems.
     *
     * The presentation knobs come from the Config, not from constants in this
     * file: how many LOD rings, how far props are drawn, whether the terrain
     * casts shadows and how many chunk masks a tick may refresh, which surface
     * shading path runs, and how big a sky block is. A host says all of it
     * through engine::HostProfile — a desktop and a 4 MiB kernel disagree about
     * every one of them.
     *
     * One call hands the whole surface behaviour to the engine: the host's
     * profile decided it, and this world only says what is ITS own — where the
     * sea is, how dense the haze, how much light a shaded face keeps.
     * The cartridge's look wins over the constants in this file whenever it said
     * anything; the CONFIG still owns the budgets. Those are two different
     * questions and they are answered by two different sources on purpose — see
     * engine/ViewProfile.hpp for why a cartridge must not carry a budget.
     *
     * Headless is legitimate — the server profile has no display. The
     * generation still runs and still logs, only the pixels are skipped.
     *
     * A body is an entity now, so the herd needs its registry BEFORE the first
     * spawn — which happens inside generate().
     *
     * Boot INTO the world, not above a map of it.
     *
     * The viewer opened in the bounded, orbiting map view, and the walking body
     * only exists in the streamed one — so the demo's first screen showed a
     * landscape from outside with no walker in it, and getting to the thing this
     * world is for took two undocumented keystrokes. The map view is still one
     * press of O away, which is the right way round: the interesting mode is the
     * default and the diagnostic one is opt-in.
     *
     * WHICH SHAPE the world has is now declared, by Config::enableStreaming,
     * and not inferred from whether a display turned up. Those are two different
     * questions that happened to have the same answer here, and while they were
     * one condition the parity smokes folded a bounded world only because the
     * headless profile has no surface — a coincidence, not a contract. First
     * person needs a screen to look at, which is a presentation question.
     *
     * An animal's whole tick, as systems, registered after generate() because
     * that is when the living layer knows its recipe.
     *
     * ORDER MATTERS and is enforced, not hoped for — and it is stated in
     * engine::systems::CreatureStage rather than by the shape of six calls here.
     * Registration order breaks ties, but every consecutive pair also shares a
     * declared dependency, so the scheduler builds a real edge: deposit and
     * evaporation both write ResourceId::ScentField, steering reads it, flocking
     * and steering both write Velocity, and grazing reads the Position that
     * locomotion writes. Get flocking before steering and the boid pass
     * overwrites the scent impulse — the pack stops flanking, and nothing else
     * says why. This order used to be written out twice, here and in the map
     * viewer, and neither copy could tell you it was the same order.
     */
    [[nodiscard]] inline core::Expected<void> onInit(engine::WorldContext &context) override;

    /**
     * @name engine::ITerrainQuery
     *
     * The three answers the creature systems need and cannot derive. All three dispatch
     * on the world's mode, which is exactly why they belong to the world: a
     * bounded grid has a blocked mask and a plant array, a streamed one has noise
     * and per-chunk plants, and no system should have to know which it is in.
     * @{
     */

    /**
     * @copydoc engine::ITerrainQuery::standable
     *
     * The endless world has no obstacle mask — there is no grid to precompute one
     * over — so the question is answered from the height alone, which
     * @c sampleWorldHeight can answer anywhere, including in chunks nobody has
     * generated. That is exactly what a creature walking toward the horizon needs.
     */
    [[nodiscard]] inline bool standable(math::Fixed32 x, math::Fixed32 z) const override;

    /**
     * @copydoc engine::ITerrainQuery::recoveryDirection
     *
     * The bounded grid is centred on the origin, so its ground is toward (0,0).
     * The streamed world has no centre at all and answers with the focus it
     * streams around — the one place its chunks are guaranteed to be resident.
     *
     * A unit step per axis, not a normalised vector: this is a nudge out of
     * somewhere illegal, and the pace belongs to the genome.
     */
    [[nodiscard]] inline bool recoveryDirection(math::Fixed32 x, math::Fixed32 z, math::Fixed32 &outX,
                                                math::Fixed32 &outZ) const override;

    /**
     * @copydoc engine::ITerrainQuery::consumePlantAt
     */
    inline bool consumePlantAt(core::i32 worldX, core::i32 worldZ) override;
    /** @} */

    /**
     * @brief Authoritative: the herd walks, the vegetation regrows, the web steps.
     *
     * Almost none of that is written here any more. The animals are six systems on
     * the scheduler, so this tick is what remains of a world's own job: move the
     * walker, follow it with the window and the resident chunks, and integrate the
     * population every so often. The body moves FIRST, before the streaming, the
     * window and the systems: following last tick's position means the chunk under
     * your feet was requested a tick late, which at a run is a visible seam opening
     * ahead of you.
     *
     * One tick of the world's clock: the sun moves, the ripples travel — both scaled by
     * dt, so both are rates per SECOND rather than per tick. This loop runs about
     * seventeen fixed steps per rendered frame (the HUD's f/100t said 6), so a per-tick
     * constant made the sea's speed a property of the tick counter. That is the defect
     * the creatures had before they were put on the scheduler.
     *
     * Shadow masks, amortised at the Config's budget: rebuilding every
     * resident chunk's mask on the tick the sun moves is a spike, and the
     * sun moves slowly enough that a mask one second stale is invisible.
     * A budget of zero turns terrain shadows off entirely, which is what a
     * headless profile asks for.
     *
     * The creature view is refreshed before the scheduler, because the systems read it.
     * The animals: mark, evaporate, steer, flock, graze, walk. Evaporation is
     * NOT a line in this function any more — it used to be one, inside the
     * infinite branch and before its `return`, so on a BOUNDED map the field
     * never evaporated at all. A system registered once cannot be forgotten in
     * one branch of an if.
     */
    inline void onFixedStep(core::f32 dt) override;

    /**
     * @brief The cycle counter, or zero when the host has no clock to ask.
     */
    [[nodiscard]] core::u64 timestamp() const noexcept
    {
        return _platform != nullptr ? _platform->clock().timestampCounter() : 0u;
    }

    /**
     * @brief Non-authoritative: input, camera, rasterize, scale, HUD, present.
     *
     * Four phases timed separately, because "the frame is slow" is not an
     * actionable statement and each of these is a different fix: the scene is
     * triangles and shading, the blit is memory bandwidth to a bigger buffer,
     * the HUD is text, and present is whatever the display costs.
     */
    inline void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override;

    /**
     * @brief Turns the accumulated cycles into shares, once per window.
     *
     * SHARES, not milliseconds. A share needs no clock frequency — which the kernel
     * would have to calibrate and which differs between a machine and an emulator —
     * and it answers the only question that decides what to optimise: which phase
     * owns the frame. Milliseconds would look more precise and say less.
     */
    void refreshProfile() noexcept;

    /**
     * @brief Starts a fresh measurement window: the rate is only about what follows.
     */
    void resetRateWindow() noexcept;

    void onShutdown() override { core::Log::info("TerrainWorld: exited"); }

    [[nodiscard]] const char *name() const noexcept override { return "TerrainWorld"; }

private:
    /**
     * @name Budgets
     *
     * Every one of these is a kernel budget, not a taste: the heap is 4 MiB and
     * the world arena is half a megabyte. A 48x48 field is ~2300 cells, which is
     * ~4400 triangles a frame — the same order as the 1024 cubes the other sample
     * already rasterizes at a usable rate in QEMU.
     * @{
     */
    /** Upper bound on the grid a cartridge may ask for, so the budgets hold. */
    static constexpr core::u32 kMaxSize = 96u;
    static constexpr core::u32 kRenderWidth = 480u;
    static constexpr core::u32 kRenderHeight = 300u;
    static constexpr core::u32 kMaxCreatures = 48u;
    static constexpr core::u32 kWebPeriod = 60u;
    core::u32 _caveDrawRadius{0u};
    static constexpr core::u32 kChunkSize = 24u;   /**< Cells per chunk edge in endless mode. */
    static constexpr core::u32 kStreamRadius = 2u; /**< Chunks the streamer keeps around the focus. */
    /**
     * @brief Ceiling on residency when the Config asks for none: the chunk table and
     * its height fields have to fit a 4 MiB heap whatever a host requests.
     *
     * It must cover what the RELEASE radius keeps, not what the generate radius
     * wants — otherwise the cap silently becomes the eviction policy and the
     * hysteresis it was paired with never gets to act. Measured at 40: the
     * streamer sat at 34 residents and climbing with `released 0`, which reads
     * like a working policy and is a queue about to be truncated by a constant.
     *
     * Release radius is 1.5x the generate radius of 2, so the kept set is a 7x7
     * ring: 49 chunks, and 56 leaves the margin. At 24 cells square that is about
     * 165 KiB of heightfield and biome map — affordable on a 4 MiB heap, which is
     * the number that decides this rather than a round figure.
     */
    static constexpr core::u32 kMaxResidentCeiling = 56u;
    static constexpr core::f32 kWalkStep = 3.0f; /**< World cells covered by one key press. */
    /**
     * @brief Radians of view per unit of pointer motion.
     *
     * A PS/2 mouse at its default resolution reports four counts per millimetre,
     * so a comfortable sweep across a desk is a few hundred counts. At 0.004 that
     * is most of a full turn, which is where a first-person view wants to be.
     */
    static constexpr core::f32 kLookSensitivity = 0.004f;
    /** Eye height of the detached camera, which floats rather than stands. */
    static constexpr core::f32 kDetachedEyeHeight = 2.0f;

    /**
     * @brief How far the cave warp searches, and how much it may spend looking.
     *
     * Derived from the measured DENSITY rather than from one seed's distance, which is
     * the mistake the first attempt made: eight rings is about two hundred cells, and the
     * shipped seed's nearest real cave is a little past that, so the key reported "no
     * cave" on the one world anybody runs.
     *
     * The density is one valid cave per roughly thirty-four cells of the mouth lattice,
     * so a disc of r rings holds about (2r + 1)^2 / 34 of them. Fourteen rings is some
     * two dozen expected caves — a margin that makes "usually finds one" into "finds one
     * unless the world genuinely has none", which is the only honest way to bound a
     * search whose subject is scattered.
     *
     * The COST is bounded separately and that is the number that matters: confirming a
     * site carries a cave is a full buildCaveWarren, about 1.4 ms, so ninety-six of them
     * is a fifth of a second spent inside one keystroke. Fine for a debug key pressed by
     * hand; never acceptable on a tick.
     */
    static constexpr core::u32 kCaveWarpRings = 14u;
    static constexpr core::u32 kCaveWarpBuilds = 96u;

    /**
     * @brief How far the spawn search may wander looking for dry, walkable ground.
     *
     * DERIVED from the terrain, not chosen. Forty cells was chosen, and it was nine
     * short: for the shipped viewer seed the nearest standable dry cell is at radius
     * 49, so the search failed, the fallback put the walker at the origin, and the
     * origin is twelve metres under the sea. The demo opened underwater and nothing
     * said why — the fallback's comment calls an all-ocean seed "legitimate", which is
     * true and was not what had happened.
     *
     * A coastline's size is set by the noise, so the reach has to be too: one full
     * wavelength of the LOWEST octave is the largest feature the terrain can make, and
     * therefore the widest bay it can put between the origin and dry land. Anything
     * smaller is a number that works until the frequency changes.
     */
    [[nodiscard]] core::i32 spawnSearchCells() const noexcept;
    static constexpr core::f32 kFirstPersonPitch = 0.05f; /**< Standing in the world, looking very slightly up. */
    static constexpr core::f32 kOrbitPitch = 0.6f;        /**< Orbiting it, looking down at the map. */
    /**
     * @brief A day every four minutes, as a rate per second.
     *
     * Long enough that the light reads as moving rather than flickering, short enough to see a
     * sunset without waiting for one.
     *
     * Was 1/14400 per TICK, which is the same thing only while the loop runs at exactly
     * sixty ticks a second. Sixty of them per second is 1/240.
     */
    static constexpr core::f32 kDayFractionPerSecond = 1.0f / 240.0f;

    /**
     * @brief How fast the swell travels, as wave phase per second.
     *
     * The wave folds with a period of two, and its spatial frequency is the water's
     * rippleScale — so a crest moves `ripplePhase / rippleScale` world cells per second.
     * At 0.42 and a scale of 0.85 that is half a cell a second: a swell that breathes.
     * 1.2 was tried first and still read as agitated — the eye judges a sea by how long a
     * crest takes to cross its own wavelength, and at 1.2 that was under two seconds.
     */
    static constexpr core::f32 kRipplePhasePerSecond = 0.42f;
    static constexpr core::u32 kFieldSpan = 64u; /**< Cells across the pheromone window that follows the walker. */
    /** Fog density. The reciprocal is roughly the distance at which haze wins. */
    static constexpr core::f32 kFogDensity = 0.010f;
    static constexpr core::f32 kSkirtDrop = 6.0f;     /**< How far a skirt hangs below the edge it is dropped from. */
    static constexpr core::f32 kPropDistance = 45.0f; /**< Furthest a boulder is worth its triangles, in world cells. */
    static constexpr core::u32 kRockAlbedo = 0x00877F76u;
    /**
     * Nearest and furthest a plant is worth drawing, in world cells.
     *
     * The near cutoff is a VISIBILITY choice, not a fix for oversized geometry:
     * the trees measure 6.8 m tall with segments of 0.85 m and a trunk 22 cm
     * across, which at two metres from the eye covers the frame — correctly, and
     * uselessly. A walker standing in a wood should see the wood, so the plant
     * whose cell they are standing on is dropped and its neighbours keep their
     * wood but lose their leaves.
     */
    static constexpr core::f32 kPlantNearDistance = 5.0f;
    /**
     * @brief Furthest a plant is worth drawing, when the host states no budget.
     *
     * A FALLBACK only: Config::viewDistance overrides it on every real profile, so
     * editing this constant alone changes nothing — measured the hard way, with a
     * frame that came back byte-identical.
     */
    static constexpr core::f32 kPlantFarDistance = 70.0f;
    /** Nearer than this the walker is under the canopy: wood only, no leaves. */
    static constexpr core::f32 kPlantCanopyDistance = 11.0f;
    /**
     * A chunk's vertical extent for culling: generous, since a box that is too
     * big only costs a chunk that is drawn and turns out to be invisible, while
     * one that is too small costs terrain that vanishes.
     */
    static constexpr core::f32 kChunkCentreY = 8.0f;
    static constexpr core::f32 kChunkHalfHeight = 72.0f;
    /** @} */

    /** How the surface is coloured. */
    enum class Shading : core::u32 {
        Biome = 0,
        Height,
        Moisture,
        Count
    };

    /** A plant is ecology::PlantCell: a signed world cell, standing or regrowing. */
    using Plant = ecology::PlantCell;

    /** A resident chunk is engine::TerrainChunk: terrain, shadow mask and plants. */
    using ResidentChunk = engine::TerrainChunk;

    /** One line of readout, reused across the frame. */
    using HudLine = render::TextLine<80>;

    /**
     * @brief Builds the bounded world from the recipe and the current seed.
     *
     * The previous world's props are retired before the next one is scattered: the
     * registry has no bulk clear by design, and the ids came back from
     * materializeProps for exactly this.
     *
     * The cartridge's own seed on the first build; N derives the next ones
     * from it. Overwriting it unconditionally — which this did once — meant
     * the document could describe a world and the viewer would draw a
     * different one, silently, from a default the .lplscene never mentions.
     *
     * A heightfield viewer does not need one entity per ground cell, and on
     * a 4 MiB heap it cannot afford them. The scatter's props DO become real
     * entities in the World's registry — the same thing a game would do.
     *
     * Building a world from a seed is procgen's business, and freeing the
     * passes' intermediate grids the moment the few this game reads have
     * been copied out is procgen::buildSnapshot's.
     *
     * @warning **The bounded map stands on the same survey the streamed world does.** Without
     * this, pressing O to look at the map showed invented ground while the world underfoot
     * was measured -- a map of somewhere else, and the one place a player would compare the
     * two. The blend is the shared one; a second implementation here would be two answers to
     * how a survey meets invented ground.
     *
     * Whether the world has a sea is read from its lowest cell, so the renderer never has to walk
     * the grid to find out whether the sea it is about to draw is behind the terrain.
     * toFloat() rather than a Fixed32 comparison: the sea level is a float here,
     * and it belongs to the RENDER — the only thing this decides is whether a water
     * pass is worth submitting, so there is nothing authoritative to preserve.
     *
     * The bounded world's field IS a window that happens to cover the whole map:
     * one type, so the herd's code cannot care which world it is walking.
     *
     * Vegetation: one plant per forested cell, thinned so the map is not a
     * solid canopy. These ARE the producer level — the population is counted,
     * never integrated, so grazing a valley bare moves the number because the
     * plants are gone.
     *
     * The food web is the cartridge's; the producer's capacity is the one value a
     * document cannot state, because it is how much vegetation THIS seed grew.
     *
     * The props are grown once per world: an L-system expansion and a swept
     * profile per variant, not per instance.
     */
    void generate();

    /**
     * @brief Builds the survey this world stands on, for whichever path is generating.
     *
     * @warning Both the streamed world and the bounded map call it, because a map that showed
     * invented ground while the world underfoot was measured would be a map of somewhere else --
     * and the map is the one place a player compares the two.
     *
     * @param amplitude The terrain amplitude the survey is scaled against.
     */
    void attachRealRelief(core::f32 amplitude);

    /**
     * @brief Switches between the bounded world and the endless one.
     *
     * The cartridge describes a bounded world: a heightfield of a stated size,
     * eroded, drained, with a town on it. An endless world cannot have those —
     * erosion and drainage are RELAXATIONS over a whole grid, and a relaxation
     * has no meaning on a piece of a world that continues past its edge. What
     * chunks seamlessly is the noise: sampled at absolute world coordinates, the
     * shared edge of two chunks is the same position and gives the same answer.
     *
     * So the two modes are not one world at two scales. Bounded is the authored
     * world; endless is the raw substrate, with everything the relaxation passes
     * would have added still missing. Saying so is the honest version of this
     * feature, and it is also the roadmap: routing drainage at HiGen's coarse
     * level and constraining the fine chunks from it is exactly what the cascade
     * rule exists to make mechanical.
     *
     * ONE description of this world, scaled up to walk through. The chunk
     * parameters and the content rule used to be written out by hand right here,
     * beside the recipe they were supposed to agree with — two descriptions of one
     * world, in one function, with nothing to say when they drifted. See
     * procgen::endlessPlanFromRecipe for what the scaling means and why it is not
     * the identity.
     *
     * Real ground under the generated ground. The measured window displaces the lowest
     * frequency where it reaches, the recipe's noise keeps the rest, and the blend band makes
     * the join instead of a cliff -- so a body can walk out of the survey and off into invented
     * country without meeting an edge. On top of it go the high frequencies 1200-metre samples
     * cannot carry. Without them the measured window is visibly smoother than the country around
     * it, which reads as a flaw in the terrain rather than as the resolution of a survey.
     *
     * @warning **The world closes on itself, and it is DERIVED rather than typed.** How many cells
     * go round is a fact about the projection -- 360 degrees at its own scale over its cell size
     * -- so asking the projection is the only way the answer cannot disagree with the ground it
     * measures. Walking east far enough now returns from the west, and the seam does not exist
     * rather than being hidden: the column past the last IS the first, same cell, same survey
     * lookup, same noise lattice.
     *
     * Whether a survey is attached is said out loud, because "the world carries a survey" is
     * otherwise invisible from a boot: a world that silently failed to attach one draws invented
     * ground that looks exactly like terrain, and nothing in the log would differ. The numbers
     * live in the generated header rather than being formatted here -- Log takes a string, and
     * hand-rolled integer formatting has already been removed from this project once as
     * speculative.
     *
     * The release ratio is 1.5x: the hysteresis that stops a camera sitting on a boundary from
     * rebuilding the same chunk every tick forever. One chunk is generated per tick. Generation is
     * the expensive thing in this loop, and a budget is what keeps a fixed step fixed.
     *
     * The body stands on the ground rather than at altitude zero: dropping it
     * from y=0 onto a mountain would spend the first seconds of the demo
     * underground, and onto a valley, falling.
     * The body is NOT placed here. Where the ground is depends on erosion and
     * on the rivers, and neither has run yet — the streamer is empty, so a query
     * would answer with raw noise. Deferred to @ref ensureBodySpawned, which
     * waits for terrain that actually exists.
     *
     * The scent is a pheromone window, not a world-sized grid.
     *
     * The ecosystem comes with: a world you can walk across for an hour and never
     * meet anything is a landscape, not a world. The producer capacity is the
     * recipe's here — a streamed world has no total to count, only what is
     * resident.
     *
     * One call says what the endless world is: the terrain parameters, the
     * streaming policy and the memory ceiling. The rule is the CONTENT —
     * where the sea and the snow line are, how much erosion, how dense the woods.
     * The river parameters come from the PLAN, not from a default-constructed member.
     * They used to be the latter, and the two then disagreed about where the sea was —
     * -1.0 against the rule's -4.0 — while the threshold stayed the one calibrated for a
     * map read from above. One derivation, one answer. The river surface rise is one product,
     * derived by the plan. See EndlessPlan::riverSurfaceRise.
     *
     * The swell runs with the PREVAILING WIND, which this world already models — the same
     * `climate.windDirection` its rain shadow is built from. Before this the two crest
     * directions were constants in the ripple function, so a lake and a river rippled
     * identically and both ignored the weather the recipe declares. Open water takes the
     * wind; a river's own current is the next step and needs the per-cell water surface.
     * Read from `seeded`, the very description the plan was derived from, rather than
     * from the member: the wind has to be the one this world was built with.
     *
     * And the RENDER's sea level too, which was a fourth independent answer to "where is
     * the water" — the view profile's default of -1.0, against the classifier's -4.0 and
     * the river pass's own. The renderer floods a PLANE at this height, so a plane three
     * metres above the sea the world was classified against drowns the coast it drew as
     * land. §33 caught two answers to "how high is the ground"; this is the same shape.
     */
    void setInfinite(bool infinite);

    /**
     * @brief Keeps the pheromone window under the walker; ai::ScentWindow owns the policy,
     * including the slack that stops a pacing walker from clearing it every step.
     */
    void recentreField();

    /**
     * @brief One authoritative tick of the player's body.
     *
     * Here rather than in @ref drainInput because a body is SIMULATION: it is
     * Fixed32, it folds, and it has to advance on the fixed step whatever the frame
     * rate is. Input is sampled on the render side and consumed here — the turn and
     * the jump are edges, so they are drained rather than read, or a slow frame
     * would apply the same jump to several ticks.
     *
     * The body steps through the SPACE, not the height. Above ground this is exactly the height
     * field with open sky over it; inside a warren it is the gallery the body is in,
     * and the body has no way to tell — which is the whole point of the seam.
     *
     * The camera is told where the body IS; it is not what moves. Reading the
     * authoritative yaw into the float camera is the allowed direction of that
     * dependency — simulation feeds presentation, never the reverse.
     */
    void stepBody(core::f32 dt);

    /**
     * @brief World cell to a cell of the pheromone window, when it is inside it.
     */
    [[nodiscard]] bool fieldCell(core::i32 worldX, core::i32 worldZ, core::u32 &outX, core::u32 &outZ) const noexcept
    {
        return _living.scent().toWindow(worldX, worldZ, outX, outZ);
    }

    /**
     * @brief Fills a chunk's shadow mask through the engine's surface layer.
     *
     * The ray march over the height field and the prop-shadow arithmetic are
     * TerrainSurface's. What this world contributes is the two things only it
     * knows: how to sample the WORLD's height (not the chunk's — a ridge one chunk
     * over still casts here, and a shadow that stopped at a border would be the most
     * visible seam in the scene) and which props stand on this chunk, how tall.
     */
    void refreshShadows(ResidentChunk &chunk) const;

    /**
     * @brief The living budgets this world asks for; the host caps them through Config.
     */
    [[nodiscard]] engine::LivingLayerParams livingParams() const noexcept;

    /**
     * @brief Fills the herd to what the freshly seeded web says should exist.
     */
    void seedHerd();

    /**
     * @brief Proposes a place a body may stand, which only this world can answer.
     *
     * In the endless world, around the FOCUS: a world with no origin would otherwise
     * put its herd at (0,0), where nobody ever goes. In the bounded one, any unblocked
     * cell of the grid.
     */
    [[nodiscard]] bool proposeSpawn(math::Random &random, core::u32 attempt, math::Fixed32 &outX,
                                    math::Fixed32 &outZ) const;

    /**
     * @brief Generates and releases chunks around the walker, within the tick budget.
     */
    void streamChunks();

    /**
     * @brief Tells the creature systems where the pheromone window sits now.
     *
     * The one thing the systems cannot derive: the window FOLLOWS the walker in the
     * endless world, so the mapping from a world cell to a field cell moves every
     * tick. It is passed as data — an origin and a size — rather than as a callback,
     * because data can be written down, folded and replayed.
     */
    void refreshCreatureView() { _creatures.setView(_living.fieldView()); }

    /**
     * @brief Eats one plant in the chunk that holds a world cell.
     *
     * The search stops at the chunk that holds the cell: when it has no plants, no other chunk
     * holds the cell either.
     *
     * @return true when it ate.
     */
    [[nodiscard]] bool grazeEndless(core::i32 worldX, core::i32 worldZ);

    /**
     * @brief Eats one plant near a world position, on the bounded grid.
     *
     * @return true when it ate.
     */
    [[nodiscard]] bool grazeBounded(math::Fixed32 x, math::Fixed32 z);

    /**
     * @brief Regrowth, and the producer population it implies.
     *
     * The producer level is what is actually STANDING: a streamed world has no total
     * to count, only a neighbourhood, and the herd only eats what it can reach.
     */
    void tickVegetation();

    /**
     * @brief Brings the bodies in line with the census; engine::LivingLayer keeps the ratio.
     */
    void reconcile();

    /**
     * @name Vegetation and props
     *
     * engine::PropLibrary grows one mesh per species and per boulder variant, decides
     * from a cell hash which one stands where, and batches the draws. Nothing about
     * that is this game's, so none of it is here.
     * @{
     */

    /**
     * @brief The one ground height, from the streamer: the field that is DRAWN.
     */
    [[nodiscard]] core::f32 groundAt(core::i32 worldX, core::i32 worldZ) const noexcept
    {
        return _streamer.groundAt(worldX, worldZ);
    }
    /** @} */

    /**
     * @name Surfaces
     *
     * The grain and its mip chain, the Lambert and physically based paths, the fog,
     * the water with its Fresnel and its probe, and the shadow masks are all
     * TerrainSurface's — configured by the host's HostProfile rather than by
     * constants in a game. What is left here is what only this world knows: which
     * colour a biome is, and where its bed lies.
     * @{
     */

    /**
     * @brief Bed height under a world point, for the water's depth tint.
     */
    [[nodiscard]] core::f32 bedHeightAt(core::f32 worldX, core::f32 worldZ) const noexcept;

    /**
     * @brief One frame, handed to engine::TerrainRenderer.
     *
     * What this world supplies is the three things a renderer cannot know: its
     * palette, how high its ground is, and — in the bounded case — what a cell looks
     * like in the data view the player selected.
     *
     * The creatures' colours are content too: a world of white hares and one of
     * black wolves are two worlds, and neither is a host budget.
     * A jumping body is not on the ground, so the eye may not be derived from
     * the terrain under it.
     *
     * The river surface rise is the same number the generator carved with, from the same plan —
     * see TerrainDrawParams::riverSurfaceRise for why it may not be a second answer. The cave
     * mouth drop is the same number the generator cut with. See TerrainDrawParams::caveMouthDrop.
     *
     * No `underground` here any more, and it is worth saying why rather than just
     * deleting it: it used to be `_embodied && _body.isEnclosed()`, and detaching
     * the camera made `_embodied` false — so in the orbit view the cave path was
     * switched off entirely, sky and all, with the eye sixty cells back and often
     * inside a hill. The renderer asks the streamer about the EYE now, which is the
     * thing it actually needs to know.
     */
    void renderScene(const render::RenderTarget &rt) const noexcept;

    /**
     * @brief The palette, which is very nearly the whole identity of a world.
     *
     * Chosen to read at 480x300 through a nearest-neighbour upscale, and that rules
     * out subtlety: adjacent biomes have to differ in VALUE, not only in hue, or a
     * coastline turns to mush at that resolution.
     *
     * An unmapped biome is magenta on purpose: it must be impossible to mistake for a
     * design choice.
     */
    [[nodiscard]] static core::u32 builtInBiomeColour(procgen::BiomeId biome) noexcept;

    /**
     * @brief The colour a biome is drawn in: the cartridge's, or this file's.
     *
     * A palette the document did not state is NOT a palette of zeroes — that
     * distinction is the whole reason the wire struct carries a count and a flag
     * rather than just sixteen words. Unstated falls through to @ref
     * builtInBiomeColour, which is what every world looked like before a cartridge
     * could say otherwise.
     */
    [[nodiscard]] core::u32 biomeColour(procgen::BiomeId biome) const noexcept
    {
        return _view.colourFor(static_cast<core::u32>(biome), builtInBiomeColour(biome));
    }

    /**
     * @brief A cell's colour in the current view: the biome, or a scalar overlay.
     *
     * The overlays are why this world keeps its grids after generating: a height or
     * moisture view is the only way to SEE what a pass produced, and a screenshot of
     * a data view has settled more arguments here than any log line.
     */
    [[nodiscard]] core::u32 cellColour(core::u32 x, core::u32 z) const noexcept;

    void drawShadowedText(core::u32 pitchPixels, core::u32 x, core::u32 y, const HudLine &line,
                          core::u32 colour) const noexcept
    {
        render::drawShadowedText8x16(_surface.buffer, pitchPixels, x, y, line.c_str(), colour);
    }

    /**
     * @brief Writes the readout over the frame.
     *
     * The position is a signed coordinate, printed signed: the previous readout took the
     * absolute value and appended "(west/north)", which is ambiguous the
     * moment one axis is negative and the other is not.
     *
     * The body is written out because it is the only way to tell a jump from a
     * fall from a slide without guessing at the picture.
     *
     * The frame rate is windowed, not cumulative: a rate counted from boot is dominated
     * by the seconds before the world finished streaming, which is
     * exactly the period that has nothing to do with what is being
     * measured. The window resets whenever the mode changes.
     *
     * @warning The landmark readout used to live BELOW the endless branch, and that branch
     * returns — so in endless mode, the only mode that has landmarks in it, it
     * was never drawn at all. Not overdrawn: unreachable. It is the line that
     * answers "I cannot see a cave entrance" against "there are no cave
     * entrances", which is the distinction TerrainStreamer's own docstring says
     * no amount of looking can make, and it had never once been on screen.
     *
     * Where the body IS, vertically, is the only readout that separates "the cave
     * is unlit" from "I never got inside it" — two completely different faults
     * that look identical from a dark screen.
     */
    void drawHud() const noexcept;

    [[nodiscard]] const char *shadingName() const noexcept;
    /** @} */

    /**
     * @name Input
     * @{
     */

    /**
     * @brief The movement bindings this class owns, then the game's own keys.
     *
     * WASD, the turn and tilt pair, the zoom, the view toggle and the shading
     * switches belong here because they act on state this class owns.
     *
     * In the endless world WASD WALKS, because there is somewhere to walk
     * to; in the bounded one it orbits, because there is not. Forward is
     * where the camera looks, which is the only reading of "forward" a
     * player ever means.
     * When the body is walking from HELD keys, a typed character must not
     * also nudge it — the two would stack and every keystroke would add a
     * lurch on top of the steady walk. The typed form stays as the fallback
     * for a backend that cannot report key states at all, and keeps driving
     * the bounded world, which has no body to move.
     * No AZERTY aliases in the TYPED path: 'q' is already zoom here, and a
     * key cannot mean two things. The held-key sampler accepts both layouts
     * and is what actually walks the body; this switch is only the fallback
     * for a backend with no key states, where a QWERTY binding is the one
     * that was always there.
     *
     * Jump is an EDGE, buffered by the controller, so pressing it a few ticks
     * early still fires on landing instead of being dropped.
     *
     * The cave warp is 'm' on QWERTY, ';' on AZERTY — and this is the one alias the typed path
     * CAN carry. The paragraph above explains why there are no others: 'q' and 'z'
     * already mean zoom and walk, so a key cannot mean two things. ';' means
     * nothing on either layout, so it is free.
     *
     * The hazard is worth spelling out because it cost a round trip: the physical
     * key labelled M on an AZERTY keyboard is scancode 0x27, which the US table
     * decodes as ';'. Pressing M on a QWERTY-built kernel therefore produced a
     * character nothing was listening for, and "nothing happens" is exactly what
     * an unbound key looks like from the outside.
     *
     * First person and orbit are one camera at two distances: collapse the
     * orbit onto the eye and the same code stands in the world. Nothing
     * switches, which is why there is no second camera to keep in step.
     *
     * N builds a whole new world, generated in ring 0 on a keystroke. This is
     * the demo's one real claim, and it is the only key that proves it.
     *
     * An unbound key SAYS so. "I press a key and nothing happens" has exactly
     * one cause — the character is not in this switch — and naming the character
     * turns that from a guess into a fact. It is the readout that was missing
     * when the M key of a French keyboard turned out to be arriving as ';': the
     * physical key is scancode 0x27, which the US table decodes that way, so a
     * QWERTY-built kernel was handed a character nothing listened for.
     *
     * @note A game cannot bind its own keys here: there is no hook, no virtual and
     * no call site.
     */
    void drainInput(engine::WorldContext &context);

    /**
     * @brief Free look: the pointer turns and tilts the camera.
     *
     * Until there was a mouse driver, looking around meant tapping J/L/I/K and the
     * view moved in fixed steps — which is enough to prove a camera works and not
     * enough to look at anything. A device that reports motion turns the same two
     * calls into a continuous one.
     *
     * Two details are the difference between this feeling right and feeling broken:
     *
     *  - The device reports UP as positive and a pitch control usually reads the
     *    other way, so the vertical delta is negated once, here, rather than
     *    argued about at four call sites.
     *  - Every packet waiting is consumed in one go. A frame that pops a single
     *    report leaves the rest in the ring, and a fast flick then arrives spread
     *    over the next second as a slow drift.
     *
     * Reported on the serial console rather than the HUD: the body line is
     * already at the width the overlay can hold, and a counter that is silently
     * truncated reads as a missing field rather than as a zero.
     *
     * Three DISTINCT sentences rather than two counters, because they are three
     * distinct diagnoses and the logger takes a string, not a format. No
     * interrupt at all is a wiring problem; interrupts without packets is a
     * decoding one; packets arriving while the view stays still is a consumer
     * one. Told apart here, they each point at one file.
     *
     * Turning goes to the BODY when there is one, because the heading picks the
     * walk direction and that makes it authoritative. Tilt stays on the camera:
     * looking up does not move you, so it is presentation and may be float.
     */
    void drainPointer(engine::WorldContext &context);

    /**
     * @brief Samples the keys that are HELD, which is what walking is made of.
     *
     * The character ring says what was typed; a direction being held is a state, and
     * rebuilding it from key repeat inherits the repeat delay as a stutter at the
     * start of every step. A backend without key states falls back to the character
     * stream — see @ref drainInput — so a host that cannot report them still walks,
     * just less smoothly.
     *
     * Asked by CHARACTER so it follows the layout: on AZERTY forward is the key that
     * types 'z', on QWERTY the one that types 'w', and neither call site has to know
     * which keyboard is plugged in.
     *
     * Two characters per direction: the same PHYSICAL keys type 'w'/'a' on
     * QWERTY and 'z'/'q' on AZERTY, and a walker should not care which keyboard
     * is plugged in. Accepting both costs one test and strands nobody.
     *
     * Opposite keys held together cancel. Letting the later one win would make a
     * player who rolls their fingers across both drift in whichever order the
     * scancodes happened to arrive.
     */
    void sampleHeldKeys(engine::WorldContext &context);
    /** @} */

    /**
     * @name Small helpers
     * @{
     */

    /**
     * @brief Turns the view, and sends the turn wherever it belongs.
     *
     * ONE statement of that routing, because there are two ways to ask for it — a
     * pointer and a pair of keys — and they were giving two different answers. A turn
     * goes to the BODY when there is one: the heading picks the walk direction, which
     * makes it authoritative, and @ref stepBody writes it into the camera every fixed
     * step. The key path turned the CAMERA instead, so the body overwrote it a few
     * milliseconds later and J/L did nothing whatsoever in first person — which reads
     * as the keys being unbound rather than as a turn going to the wrong place.
     *
     * The sign is inverted for the body deliberately: its yaw and the camera's turn
     * run opposite ways, and this is where that is reconciled rather than at each of
     * the two call sites.
     *
     * @param yawRadians Radians to turn by, in the camera's sense.
     */
    void look(core::f32 yawRadians) noexcept;

    /**
     * @brief Walks along the camera's heading, in world cells.
     */
    void walk(core::f32 sign) noexcept { _camera.walk(sign * kWalkStep); }

    /**
     * @brief Sidesteps: the heading rotated a quarter turn.
     */
    void strafe(core::f32 sign) noexcept { _camera.strafe(sign * kWalkStep); }

    /**
     * @brief Places the body once there is real terrain to place it on.
     *
     * The first attempt did this at world creation and it read the wrong surface:
     * with no chunk resident, a height query falls through to the raw noise — the
     * shape BEFORE erosion lowered the ridges and before the rivers were carved. The
     * noise said the origin was dry ground; the world that gets drawn has a lake
     * there, so the walker sank to the bed and the demo opened underwater. Exactly
     * the mistake that once left the herd hanging in the air over lowered ridges.
     *
     * Waiting for a three-by-three neighbourhood is what makes the search look at
     * the terrain the renderer will draw.
     */
    void ensureBodySpawned();

    /**
     * @brief Puts the body down somewhere it can stand — on land, not in a lake.
     *
     * Dropping it at the origin was the obvious thing and it was wrong on this seed:
     * the origin is a lake, so the walker sank to the bed and the demo opened with
     * the camera underwater looking up through the surface. The physics was right;
     * the spawn was not.
     *
     * The search is a widening ring rather than a random scatter, so the body lands
     * as close to the requested spot as the terrain allows and the same seed always
     * spawns in the same place — a random probe would have made the opening view
     * depend on how many times the allocator had been called.
     *
     * Sea level lives here rather than in the controller on purpose: an engine brick
     * that walks has no business knowing a world has water in it, while THIS world
     * knows exactly where its shoreline is.
     *
     * Only the RING is tested at each radius, not the filled square: the inner cells were
     * already rejected at a smaller radius, and re-testing them makes the search quadratic in
     * the radius for nothing. Not on a face it would immediately slide off, either.
     *
     * Spawned at the SURFACE and settled from there: the search
     * looked at the terrain, so the gap the body means is the one over
     * the terrain and not whatever gallery happens to run under it.
     *
     * Nothing dry within the search: an all-ocean seed is a legitimate world, and
     * starting at the origin is a better answer than refusing to start. SAID, though
     * — silently drowning the walker and letting the player work out why is how this
     * went unnoticed while the reach was too short to clear a bay.
     */
    void spawnBody();

    /**
     * @brief Puts the walker in the trench of the nearest cave, facing the way in.
     *
     * A debug affordance, and it exists because of a measurement: on the shipped viewer
     * seed the nearest cave to the origin is 224 cells away, and caves are sited about
     * one per twenty-four chunks. Walking to one to look at it is minutes of holding a
     * key, so the cave rendering could only ever be checked on the host — which is a
     * poor substitute for standing in the thing.
     *
     * Three details separate this from a bare coordinate assignment, and each is the
     * difference between arriving somewhere useful and arriving confused:
     *
     *  - it lands OUTSIDE the doorway, in the trench, so the approach and the mouth are
     *    both in front of you rather than behind;
     *  - it sets the HEADING, so forward walks in. Arriving on the threshold pointing at
     *    a hillside looks exactly like arriving nowhere;
     *  - it takes the body and puts the eye in it, because an orbit camera sixty cells
     *    back is usually inside the hill — which was the original report.
     *
     * Says what it did either way. A teleport that silently fails is indistinguishable
     * from a key that is not bound, which is how J/L went unnoticed for a whole session.
     *
     * The walker stands on the SITE CENTRE, which is the middle of the shelf — not a fixed number
     * of cells back down the adit. Six back was measured landing 7.4 m BELOW the trench floor,
     * out on the raw hillside: the shelf is only about four cells across and the
     * trench a couple more, so a constant standoff overshoots the flat ground and
     * drops you on a slope too steep to climb. The centre of the shelf is flat by
     * construction — it is the one cell the carve guarantees is at the trench floor —
     * and the mouth is a few cells ahead of it, in view.
     *
     * Facing the mouth, derived the way the parity walk derives it: the body's own
     * convention is wish = (-forward * sin(yaw), -forward * cos(yaw)), so the heading
     * that walks along the adit is atan2 of its negated step.
     */
    void teleportToNearestCaveMouth();

    /**
     * @brief Whether the body is what movement keys act on this frame.
     */
    [[nodiscard]] bool bodyDrivesMovement(const engine::WorldContext &context) const noexcept
    {
        return _embodied && context.platform.input().hasKeyStates();
    }

    /**
     * @brief The fallback walk: one typed keystroke becomes one tick of intent.
     *
     * For a backend with no key states, and for the detached camera. It writes the
     * INTENT rather than the position even in the embodied case, so the body still
     * decides — a keystroke that teleported the walker would put it inside a hill
     * the controller was refusing to enter.
     */
    void nudge(core::f32 forward, core::f32 strafeAmount) noexcept;

    [[nodiscard]] math::Fixed32 cellToWorldX(core::u32 cell) const noexcept;

    [[nodiscard]] math::Fixed32 cellToWorldZ(core::u32 cell) const noexcept;

    [[nodiscard]] bool worldToCell(math::Fixed32 x, math::Fixed32 z, core::u32 &outX, core::u32 &outZ) const noexcept;

    [[nodiscard]] core::u32 standingPlants() const noexcept
    {
        return _plants.empty() ? 0u : ecology::countStanding(&_plants[0], static_cast<core::u32>(_plants.size()));
    }

    [[nodiscard]] core::u32 countSpecies(core::u32 species) const noexcept
    {
        return _living.herd().countSpecies(species);
    }

    [[nodiscard]] core::u32 presentBiomes() const noexcept;

    static constexpr core::f32 kFloor = -6.0f;
    /**
     * @brief Where the sea is: the RECIPE's, never a constant here.
     *
     * There was a `kSeaLevel = -1.0f` in this block, and with it the shipped
     * cartridge had three answers to one question: the classifier called everything
     * at or below -4 (the engine default, since the document said nothing) Ocean, the
     * view profile drew water at -1, and this constant blocked walking below -1. The
     * band between them was land you could not walk on, under water, that the scatter
     * was free to plant trees in.
     */
    [[nodiscard]] core::f32 seaLevel() const noexcept { return _recipe.biomes.seaLevel; }
    /** @} */

    /**
     * @name The frame buffers
     *
     * They live in BSS, like the other sample's: a kernel stack
     * cannot hold 480x300 pixels plus their depth, and the heap should not have to.
     * @{
     */
    static core::u32 _color[kRenderWidth * kRenderHeight];
    static core::f32 _depth[kRenderWidth * kRenderHeight];
    /** @} */

    platform::SurfaceDescriptor _surface{};
    bool _hasSurface{false};

    procgen::Heightfield _height;
    procgen::Heightfield _moisture;
    procgen::BiomeMap _biomes;
    procgen::Grid<core::u8> _rivers;
    procgen::Grid<core::u8> _blocked;
    procgen::SettlementMap _settlement;
    procgen::Grid<core::u8> _roads;
    procgen::WorldRecipe _recipe{procgen::parityWorldRecipe()};

    /**
     * @name The endless world
     * @{
     */
    procgen::ChunkParams _chunkParams{};

    /**
     * The measured window, and the mosaic that offers it.
     *
     * @warning The mosaic holds a POINTER to the field, so the two must live and die together and
     * neither may move. Members of the world, so their lifetime is the world's -- which outlives
     * every chunk that reads through them.
     */
    math::ReliefField _reliefField{};
    math::ReliefMosaic _reliefMosaic{};
    procgen::StreamingParams _streamParams{};
    procgen::EndlessRiverParams _riverParams{};
    /** @} */

    bool _infinite{false};

    mutable core::u32 _lastTriangles{0u};
    ecology::LivingRecipe _livingRecipe{ecology::parityLivingRecipe()};
    /**
     * @brief What the world looks like, as the cartridge stated it.
     *
     * Default-constructed it reproduces the constants this file used to hold, field
     * for field — which is what makes a pack with no view section indistinguishable
     * from the version before the section existed.
     */
    engine::ViewProfile _view{};
    lpl::pmr::vector<ecs::EntityId> _propIds;
    core::u32 _gridWidth{0u};
    core::u32 _gridDepth{0u};
    core::u32 _propCount{0u};
    core::f32 _riverSurfaceRise{0.0f};
    core::f32 _caveMouthDrop{0.0f};
    bool _boundedHasSea{true};
    core::u32 _plotCount{0u};
    core::u32 _roadCells{0u};
    core::u32 _riverCells{0u};
    core::u32 _caveFloor{0u};
    bool _gatePassed{false};
    math::Fixed32 _low{};
    math::Fixed32 _high{};
    core::u32 _biomeCounts[static_cast<core::u32>(procgen::BiomeId::Count)] = {};

    lpl::pmr::vector<Plant> _plants;

    /** One grown plant per species, and the transforms are the instances. */
    static constexpr core::u32 kTreeSpecies = 3u;

    engine::LivingLayer _living; /**< Population, bodies and the scent they read: engine::LivingLayer. */
    /**
     * The six systems of an animal's tick, ordered by engine::systems::CreatureStage. Owned by the
     * scheduler; held to keep their window in step with the scent's.
     */
    engine::systems::CreaturePipeline _creatures;

    mutable engine::TerrainStreamer _streamer; /**< The endless world's chunks: residency, generation, ground height. */

    mutable engine::PropLibrary _props; /**< The props: grown once, placed by a cell hash, drawn batched. */

    mutable engine::TerrainSurface _terrainSurface; /**< Surface layer: sky, sun, grain, shadows, water, reflection. */

    render::OrbitCamera _camera{};
    /**
     * @name The probe's buffers
     *
     * The sun, the sky, the water, the grain and the probe live in the facility;
     * only the probe's BUFFERS stay here, because they must sit in BSS: a 240x150 colour+depth
     * pair is 288 KiB, and the kernel stack is nowhere near that. The scene learned this lesson
     * once already.
     * @{
     */
    static constexpr core::u32 kProbeWidth = 240u;
    static constexpr core::u32 kProbeHeight = 150u;
    static core::u32 _probeColor[kProbeWidth * kProbeHeight];
    static core::f32 _probeDepth[kProbeWidth * kProbeHeight];
    /** @} */
    mutable render::ReflectionProbe _probe{};
    mutable render::Texture _probePixels{};
    bool _reflectionProbeOn{true};
    /** Boulder meshes, swept once from a tessellated profile and instanced. */
    static constexpr core::u32 kRockVariants = 3u;
    /**
     * Per-pixel surface shading, switchable at runtime.
     *
     * Not a debug leftover: it is the only way to MEASURE what the per-pixel path
     * costs on this hardware. Two builds cannot be compared frame for frame — the
     * world drifts, the herd moves — so the comparison has to happen inside one
     * boot, with the frame counter on screen.
     */
    bool _perPixelSurface{true};
    mutable engine::TerrainRenderer _renderer; /**< The frame: engine::TerrainRenderer owns the passes. */

    /**
     * @name Salts that keep the two scatter layers from agreeing with each other
     * @{
     */
    static constexpr core::u32 kPlantSalt = 0x11u;
    static constexpr core::u32 kRockSalt = 0x5Bu;
    /** @} */

    core::u32 _skyBlock{1u};
    /**
     * @name Read from the Config at init; see @ref onInit
     * @{
     */
    core::u32 _lodRings{3u};
    core::f32 _viewDistance{70.0f};
    bool _terrainShadows{true};
    core::u32 _shadowChunksPerTick{1u};
    core::u32 _maxResident{56u};
    bool _pbrSurface{false};
    /** @} */
    mutable core::u32 _windowFrames{0u};
    core::u32 _windowTicks{0u};
    Shading _shading{Shading::Biome};
    core::u32 _seed{1337u};
    core::u32 _ticks{0u};
    core::u32 _frames{0u};
    core::u32 _pointerButtons{0u}; /**< Buttons held at the last pointer report; bit 0 left, 1 right, 2 middle. */
    /**
     * @brief The platform, kept so the AUTHORITATIVE step can read the clock.
     *
     * onFixedStep receives no context — deliberately, it is simulation and must not
     * reach for a display. A timestamp is not a display, and measuring where a frame
     * goes is the only way to know which half to make faster.
     */
    platform::IPlatform *_platform{nullptr};
    core::u64 _simCycles{0u};
    core::u64 _sceneCycles{0u};
    core::u64 _blitCycles{0u};
    core::u64 _hudCycles{0u};
    core::u64 _presentCycles{0u};
    core::u32 _profileWindow{0u};
    core::u32 _simShare{0u};
    core::u32 _sceneShare{0u};
    core::u32 _blitShare{0u};
    core::u32 _presentShare{0u};
    core::u32 _skyShare{0u};
    core::u32 _groundShare{0u};
    core::u32 _waterShare{0u};
    core::u32 _propShare{0u};
    bool _hasPointer{false};
    core::u32 _pointerPackets{0u};
    core::u32 _pointerReported{0u};

    /**
     * @brief The walker's body: authoritative, and subject to the world.
     *
     * Not a camera with a position. The camera used to BE the player — it moved by
     * teleporting its focus a fixed step per keypress, passed through hills, and
     * never fell. This is an entity like the herd's: pulled down by gravity, standing
     * on the terrain that is actually drawn, stopped by cliffs, able to jump.
     */
    engine::CharacterController _body;
    engine::CharacterParams _bodyParams{};
    engine::CharacterIntent _intent{}; /**< Sampled on the render side, consumed by the authoritative step. */
    bool _bodySpawned{false};          /**< False until terrain exists to stand on; see @ref ensureBodySpawned. */
    math::Fixed32 _pendingTurn{};      /**< Turn accumulated by the pointer since the last authoritative step. */
    bool _jumpPressed{false};          /**< Jump pressed since the last authoritative step; consumed by it. */
    /**
     * @brief Whether the player is a body or a free camera.
     *
     * Both are worth having and they answer different questions: embodied is what
     * the world FEELS like, detached is how you look at what you built. Toggled with
     * V, and the body keeps simulating while detached so coming back is seamless.
     */
    bool _embodied{true};
};

inline core::u32 TerrainWorld::_color[TerrainWorld::kRenderWidth * TerrainWorld::kRenderHeight];
inline core::f32 TerrainWorld::_depth[TerrainWorld::kRenderWidth * TerrainWorld::kRenderHeight];
inline core::u32 TerrainWorld::_probeColor[TerrainWorld::kProbeWidth * TerrainWorld::kProbeHeight];
inline core::f32 TerrainWorld::_probeDepth[TerrainWorld::kProbeWidth * TerrainWorld::kProbeHeight];

} // namespace lpl::samples

#    include "TerrainWorld.inl"

#endif // LPL_SAMPLES_TERRAINWORLD_HPP
