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
 * A game derives from this and answers only what is ITS own — which is, in
 * practice, a colour palette and a readout. The hooks are:
 *
 *  - @ref biomeColourOf : what a biome looks like.
 *  - @ref cellColourOf : a scalar view of a cell, for data overlays.
 *  - @ref onHud : what to write on the frame.
 *  - @ref onKey : keys beyond the movement set this class already binds.
 *
 * Everything else is settings, and the settings have owners: the terrain and its
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

namespace lpl::samples {

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
    [[nodiscard]] core::Expected<void> onInit(engine::WorldContext &context) override
    {
        // The presentation knobs come from the Config, not from constants in this
        // file: how many LOD rings, how far props are drawn, whether the terrain
        // casts shadows and how many chunk masks a tick may refresh, which surface
        // shading path runs, and how big a sky block is. A host says all of it
        // through engine::HostProfile — a desktop and a 4 MiB kernel disagree about
        // every one of them.
        _lodRings = context.config.lodRings();
        _viewDistance = context.config.viewDistance() > 0.0f ? context.config.viewDistance() : kPlantFarDistance;
        _terrainShadows = context.config.enableTerrainShadows();
        _shadowChunksPerTick = context.config.shadowChunksPerTick();
        _perPixelSurface = context.config.enablePerPixelSurface();
        _pbrSurface = context.config.enablePbrSurface();
        _reflectionProbeOn = context.config.enableWaterReflection();
        _skyBlock = context.config.skyBlockSize() == 0u ? 1u : context.config.skyBlockSize();

        // One call hands the whole surface behaviour to the engine: the host's
        // profile decided it, and this world only says what is ITS own — where the
        // sea is, how dense the haze, how much light a shaded face keeps.
        // The cartridge's look wins over the constants in this file whenever it said
        // anything; the CONFIG still owns the budgets. Those are two different
        // questions and they are answered by two different sources on purpose — see
        // engine/ViewProfile.hpp for why a cartridge must not carry a budget.
        _terrainSurface.configure(context.config, _view.surface, _seed);
        _terrainSurface.applyLook(_view.sky, _view.water, _view.dayFraction);
        _terrainSurface.attachProbe(_probeColor, _probeDepth, kProbeWidth, kProbeHeight);
        _maxResident =
            context.config.maxResidentChunks() == 0u ? kMaxResidentCeiling : context.config.maxResidentChunks();
        _caveDrawRadius = context.config.caveDrawRadius();

        _platform = &context.platform;
        _renderer.setClock(&context.platform.clock());
        _hasPointer = context.platform.input().hasPointer();
        core::Log::info(_hasPointer ? "TerrainWorld: pointing device present — mouse look enabled" :
                                      "TerrainWorld: no pointing device — look with I/K and the pointer keys");

        if (!context.platform.display().querySurface(_surface) || _surface.buffer == nullptr ||
            _surface.bitsPerPixel != 32u || _surface.width < 64u || _surface.height < 64u)
        {
            // Headless is legitimate — the server profile has no display. The
            // generation still runs and still logs, only the pixels are skipped.
            _hasSurface = false;
            core::Log::info("TerrainWorld: no usable 32bpp surface, running headless");
        }
        else
        {
            _hasSurface = true;
            core::Log::info("TerrainWorld: WASD=walk SPACE=jump mouse=look O=map V=detach X=exit");
        }

        // A body is an entity now, so the herd needs its registry BEFORE the first
        // spawn — which happens inside generate().
        _living.bind(registry());

        generate();

        // Boot INTO the world, not above a map of it.
        //
        // The viewer opened in the bounded, orbiting map view, and the walking body
        // only exists in the streamed one — so the demo's first screen showed a
        // landscape from outside with no walker in it, and getting to the thing this
        // world is for took two undocumented keystrokes. The map view is still one
        // press of O away, which is the right way round: the interesting mode is the
        // default and the diagnostic one is opt-in.
        //
        // WHICH SHAPE the world has is now declared, by Config::enableStreaming,
        // and not inferred from whether a display turned up. Those are two different
        // questions that happened to have the same answer here, and while they were
        // one condition the parity smokes folded a bounded world only because the
        // headless profile has no surface — a coincidence, not a contract.
        if (context.config.enableStreaming())
            setInfinite(true);
        // First person needs a screen to look at, which is a presentation question.
        if (_hasSurface && _infinite)
        {
            _camera.setFirstPerson(true);
            _camera.setPitch(kFirstPersonPitch);
        }

        // An animal's whole tick, as systems, registered after generate() because
        // that is when the living layer knows its recipe.
        //
        // ORDER MATTERS and is enforced, not hoped for — and it is stated in
        // engine::systems::CreatureStage rather than by the shape of six calls here.
        // Registration order breaks ties, but every consecutive pair also shares a
        // declared dependency, so the scheduler builds a real edge: deposit and
        // evaporation both write ResourceId::ScentField, steering reads it, flocking
        // and steering both write Velocity, and grazing reads the Position that
        // locomotion writes. Get flocking before steering and the boid pass
        // overwrites the scent impulse — the pack stops flanking, and nothing else
        // says why. This order used to be written out twice, here and in the map
        // viewer, and neither copy could tell you it was the same order.
        _creatures.build(registry(), _living, *this);
        if (auto registered = _creatures.registerOn(scheduler()); !registered)
            core::Log::warn("TerrainWorld: a creature system was refused; the living layer will be incomplete");
        return {};
    }

    // ── engine::ITerrainQuery ────────────────────────────────────────────────
    //
    // The two answers the creature systems need and cannot derive. Both dispatch
    // on the world's mode, which is exactly why they belong to the world: a
    // bounded grid has a blocked mask and a plant array, a streamed one has noise
    // and per-chunk plants, and no system should have to know which it is in.

    /// @copydoc engine::ITerrainQuery::standable
    [[nodiscard]] bool standable(math::Fixed32 x, math::Fixed32 z) const override
    {
        if (_infinite)
        {
            const core::f32 height =
                procgen::sampleWorldHeight(_streamer.chunkParams(), x.toInt(), z.toInt()).toFloat();
            return height > seaLevel() + 0.2f;
        }
        core::u32 cx = 0u;
        core::u32 cz = 0u;
        if (!worldToCell(x, z, cx, cz))
            return false;
        return _blocked.at(cx, cz) == 0u;
    }

    /// @copydoc engine::ITerrainQuery::recoveryDirection
    ///
    /// The bounded grid is centred on the origin, so its ground is toward (0,0).
    /// The streamed world has no centre at all and answers with the focus it
    /// streams around — the one place its chunks are guaranteed to be resident.
    [[nodiscard]] bool recoveryDirection(math::Fixed32 x, math::Fixed32 z, math::Fixed32 &outX,
                                         math::Fixed32 &outZ) const override
    {
        const math::Fixed32 targetX = _infinite ? math::Fixed32::fromFloat(_camera.focusX()) : math::Fixed32{};
        const math::Fixed32 targetZ = _infinite ? math::Fixed32::fromFloat(_camera.focusZ()) : math::Fixed32{};
        // A unit step per axis, not a normalised vector: this is a nudge out of
        // somewhere illegal, and the pace belongs to the genome.
        outX = x.raw() > targetX.raw() ? -math::Fixed32::one() : math::Fixed32::one();
        outZ = z.raw() > targetZ.raw() ? -math::Fixed32::one() : math::Fixed32::one();
        return true;
    }

    /// @copydoc engine::ITerrainQuery::consumePlantAt
    bool consumePlantAt(core::i32 worldX, core::i32 worldZ) override
    {
        const bool ate = _infinite ? grazeEndless(worldX, worldZ) :
                                     grazeBounded(math::Fixed32::fromInt(worldX), math::Fixed32::fromInt(worldZ));
        if (ate)
            _living.countGrazed();
        return ate;
    }

    /// Authoritative: the herd walks, the vegetation regrows, the web steps.
    ///
    /// Almost none of that is written here any more. The animals are six systems on
    /// the scheduler, so this tick is what remains of a world's own job: move the
    /// walker, follow it with the window and the resident chunks, and integrate the
    /// population every so often. The body moves FIRST, before the streaming, the
    /// window and the systems: following last tick's position means the chunk under
    /// your feet was requested a tick late, which at a run is a visible seam opening
    /// ahead of you.
    void onFixedStep(core::f32 dt) override
    {
        const core::u64 stepBegan = timestamp();
        // A day every four minutes at 60 Hz: long enough that the light reads as
        // moving rather than flickering, short enough to see a sunset without
        // waiting for one.
        ++_windowTicks;
        // One tick of the world's clock: the sun moves, the ripples travel — both scaled by
        // dt, so both are rates per SECOND rather than per tick. This loop runs about
        // seventeen fixed steps per rendered frame (the HUD's f/100t said 6), so a per-tick
        // constant made the sea's speed a property of the tick counter. That is the defect
        // the creatures had before they were put on the scheduler.
        _terrainSurface.advance(kDayFractionPerSecond * dt, kRipplePhasePerSecond * dt);

        if (_infinite)
        {
            stepBody(dt);
            recentreField();
            streamChunks();
            // Shadow masks, amortised at the Config's budget: rebuilding every
            // resident chunk's mask on the tick the sun moves is a spike, and the
            // sun moves slowly enough that a mask one second stale is invisible.
            // A budget of zero turns terrain shadows off entirely, which is what a
            // headless profile asks for.
            if (_terrainShadows && !_streamer.empty())
                for (core::u32 refreshed = 0u; refreshed < _shadowChunksPerTick; ++refreshed)
                {
                    if (ResidentChunk *chunk = _streamer.nextShadowChunk(); chunk != nullptr)
                        refreshShadows(*chunk);
                }
        }

        // Before the scheduler, because the systems read it.
        refreshCreatureView();
        // The animals: mark, evaporate, steer, flock, graze, walk. Evaporation is
        // NOT a line in this function any more — it used to be one, inside the
        // infinite branch and before its `return`, so on a BOUNDED map the field
        // never evaporated at all. A system registered once cannot be forgotten in
        // one branch of an if.
        engine::World::onFixedStep(dt);
        ++_ticks;

        if (_ticks % kWebPeriod == 0u)
        {
            _living.web().step(1u);
            reconcile();
        }
        _simCycles += timestamp() - stepBegan;
    }

    /// The cycle counter, or zero when the host has no clock to ask.
    [[nodiscard]] core::u64 timestamp() const noexcept
    {
        return _platform != nullptr ? _platform->clock().timestampCounter() : 0u;
    }

    /// Non-authoritative: input, camera, rasterize, scale, HUD, present.
    void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override
    {
        drainInput(context);
        if (!_hasSurface)
            return;

        render::RenderTarget target{_color, _depth, kRenderWidth, kRenderHeight};

        // Four phases timed separately, because "the frame is slow" is not an
        // actionable statement and each of these is a different fix: the scene is
        // triangles and shading, the blit is memory bandwidth to a bigger buffer,
        // the HUD is text, and present is whatever the display costs.
        const core::u64 sceneBegan = timestamp();
        renderScene(target);
        const core::u64 blitBegan = timestamp();
        render::blitScaled(_surface.buffer, _surface.pitch / 4u, _surface.width, _surface.height, _color, kRenderWidth,
                           kRenderHeight);
        const core::u64 hudBegan = timestamp();
        drawHud();
        const core::u64 presentBegan = timestamp();
        context.platform.display().present();
        const core::u64 frameEnded = timestamp();

        _sceneCycles += blitBegan - sceneBegan;
        _blitCycles += hudBegan - blitBegan;
        _hudCycles += presentBegan - hudBegan;
        _presentCycles += frameEnded - presentBegan;
        ++_frames;
        ++_windowFrames;
        refreshProfile();
    }

    /**
     * @brief Turns the accumulated cycles into shares, once per window.
     *
     * SHARES, not milliseconds. A share needs no clock frequency — which the kernel
     * would have to calibrate and which differs between a machine and an emulator —
     * and it answers the only question that decides what to optimise: which phase
     * owns the frame. Milliseconds would look more precise and say less.
     */
    void refreshProfile() noexcept
    {
        if (++_profileWindow < 24u)
            return;
        _profileWindow = 0u;

        const core::u64 total = _simCycles + _sceneCycles + _blitCycles + _hudCycles + _presentCycles;
        if (total != 0u)
        {
            _simShare = static_cast<core::u32>((_simCycles * 100u) / total);
            _sceneShare = static_cast<core::u32>((_sceneCycles * 100u) / total);
            _skyShare = static_cast<core::u32>((_renderer.skyCycles() * 100u) / total);
            _groundShare = static_cast<core::u32>((_renderer.groundCycles() * 100u) / total);
            _waterShare = static_cast<core::u32>((_renderer.waterCycles() * 100u) / total);
            _propShare = static_cast<core::u32>((_renderer.propCycles() * 100u) / total);
            _blitShare = static_cast<core::u32>((_blitCycles * 100u) / total);
            _presentShare = static_cast<core::u32>((_presentCycles * 100u) / total);
        }
        _simCycles = 0u;
        _sceneCycles = 0u;
        _blitCycles = 0u;
        _hudCycles = 0u;
        _presentCycles = 0u;
        _renderer.resetPhaseCounters();
    }

    /// Starts a fresh measurement window: the rate is only about what follows.
    void resetRateWindow() noexcept
    {
        _windowFrames = 0u;
        _windowTicks = 0u;
    }

    void onShutdown() override { core::Log::info("TerrainWorld: exited"); }

    [[nodiscard]] const char *name() const noexcept override { return "TerrainWorld"; }

private:
    // ── Budgets ──────────────────────────────────────────────────────────────
    //
    // Every one of these is a kernel budget, not a taste: the heap is 4 MiB and
    // the world arena is half a megabyte. A 48x48 field is ~2300 cells, which is
    // ~4400 triangles a frame — the same order as the 1024 cubes the other sample
    // already rasterizes at a usable rate in QEMU.
    /// Upper bound on the grid a cartridge may ask for, so the budgets hold.
    static constexpr core::u32 kMaxSize = 96u;
    static constexpr core::u32 kRenderWidth = 480u;
    static constexpr core::u32 kRenderHeight = 300u;
    static constexpr core::u32 kMaxCreatures = 48u;
    static constexpr core::u32 kWebPeriod = 60u;
    /// Cells per chunk edge in endless mode.
    core::u32 _caveDrawRadius{0u};
    static constexpr core::u32 kChunkSize = 24u;
    /// Chunks the streamer keeps around the focus.
    static constexpr core::u32 kStreamRadius = 2u;
    /**
     * @brief Hard ceiling on residency.
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
    /// Ceiling on residency when the Config asks for none: the chunk table and
    /// its height fields have to fit a 4 MiB heap whatever a host requests.
    static constexpr core::u32 kMaxResidentCeiling = 56u;
    /// World cells covered by one key press.
    static constexpr core::f32 kWalkStep = 3.0f;
    /**
     * @brief Radians of view per unit of pointer motion.
     *
     * A PS/2 mouse at its default resolution reports four counts per millimetre,
     * so a comfortable sweep across a desk is a few hundred counts. At 0.004 that
     * is most of a full turn, which is where a first-person view wants to be.
     */
    static constexpr core::f32 kLookSensitivity = 0.004f;
    /// Eye height of the detached camera, which floats rather than stands.
    static constexpr core::f32 kDetachedEyeHeight = 2.0f;
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
    [[nodiscard]] core::i32 spawnSearchCells() const noexcept
    {
        const procgen::NoiseParams &noise = _streamer.chunkParams().noise;
        if (noise.frequency <= 0.0f)
            return 64;
        core::f32 wavelength = 1.0f / noise.frequency;
        for (core::u32 octave = 1u; octave < noise.octaves; ++octave)
            wavelength *= noise.lacunarity > 1.0f ? noise.lacunarity : 2.0f;
        const core::i32 cells = static_cast<core::i32>(wavelength);
        return cells < 64 ? 64 : (cells > 512 ? 512 : cells);
    }
    /// Standing in the world, looking very slightly up.
    static constexpr core::f32 kFirstPersonPitch = 0.05f;
    /// Orbiting it, looking down at the map.
    static constexpr core::f32 kOrbitPitch = 0.6f;
    /// One tick of the day: a full cycle takes about four minutes at 60 Hz.
    /**
     * @brief A day every four minutes, as a rate per second.
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
    /// Cells across the pheromone window that follows the walker.
    static constexpr core::u32 kFieldSpan = 64u;
    /// Ambient floor: what a surface facing away from the sun still receives.
    static constexpr core::f32 kAmbient = 0.28f;
    /// Fog density. The reciprocal is roughly the distance at which haze wins.
    static constexpr core::f32 kFogDensity = 0.010f;
    /// Cells a shadow ray marches. Also the longest shadow the terrain can cast.
    static constexpr core::u32 kShadowSteps = 24u;
    /// A chunk's vertical extent for culling: generous, since a box that is too
    /// big only costs a chunk that is drawn and turns out to be invisible, while
    /// one that is too small costs terrain that vanishes.
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
    /// How far a skirt hangs below the edge it is dropped from.
    static constexpr core::f32 kSkirtDrop = 6.0f;
    /// Furthest a boulder is worth its triangles, in world cells.
    static constexpr core::f32 kPropDistance = 45.0f;
    static constexpr core::u32 kRockAlbedo = 0x00877F76u;
    static constexpr core::u32 kGrazerTint = 0x00D0A852u;
    static constexpr core::u32 kHunterTint = 0x00C03028u;
    static constexpr core::f32 kPlantNearDistance = 5.0f;
    /**
     * @brief Furthest a plant is worth drawing, when the host states no budget.
     *
     * A FALLBACK only: Config::viewDistance overrides it on every real profile, so
     * editing this constant alone changes nothing — measured the hard way, with a
     * frame that came back byte-identical.
     */
    static constexpr core::f32 kPlantFarDistance = 70.0f;
    /// Nearer than this the walker is under the canopy: wood only, no leaves.
    static constexpr core::f32 kPlantCanopyDistance = 11.0f;
    static constexpr core::f32 kChunkCentreY = 8.0f;
    static constexpr core::f32 kChunkHalfHeight = 72.0f;

    /// How the surface is coloured.
    enum class Shading : core::u32 {
        Biome = 0,
        Height,
        Moisture,
        Count
    };

    /// A plant is ecology::PlantCell: a signed world cell, standing or regrowing.
    using Plant = ecology::PlantCell;

    /// A resident chunk is engine::TerrainChunk: terrain, shadow mask and plants.
    using ResidentChunk = engine::TerrainChunk;

    /// One line of readout, reused across the frame.
    using HudLine = render::TextLine<80>;

    void generate()
    {
        // Retire the previous world's props before the next one is scattered: the
        // registry has no bulk clear by design, and the ids came back from
        // materializeProps for exactly this.
        for (core::usize i = 0u; i < _propIds.size(); ++i)
            (void) registry().destroyEntity(_propIds[i]);
        _propIds.clear();

        {
            // The cartridge's own seed on the first build; N derives the next ones
            // from it. Overwriting it unconditionally — which this did once — meant
            // the document could describe a world and the viewer would draw a
            // different one, silently, from a default the .lplscene never mentions.
            procgen::WorldRecipe recipe = _recipe;
            recipe.seed = _seed;
            recipe.terrain.seed = _seed;
            recipe.caves.seed = _seed ^ 0xCA4Eu;
            // A heightfield viewer does not need one entity per ground cell, and on
            // a 4 MiB heap it cannot afford them. The scatter's props DO become real
            // entities in the World's registry — the same thing a game would do.
            recipe.materializeGround = false;

            // Building a world from a seed is procgen's business, and freeing the
            // passes' intermediate grids the moment the few this game reads have
            // been copied out is procgen::buildSnapshot's.
            const procgen::WorldSnapshot snapshot = procgen::buildSnapshot(
                recipe, &registry(), &_propIds, procgen::WalkabilityRule{recipe.biomes.seaLevel, 2.4f});

            _height = snapshot.height;
            _biomes = snapshot.biomes;
            _moisture = snapshot.moisture;
            _rivers = snapshot.rivers;
            _settlement = snapshot.settlement;
            _roads = snapshot.roads;
            _blocked = snapshot.blocked;
            _gridWidth = snapshot.width;
            _gridDepth = snapshot.depth;
            _low = snapshot.lowest;
            _high = snapshot.highest;
            for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
                _biomeCounts[i] = snapshot.biomeCounts[i];

            _propCount = snapshot.stats.propEntities;
            _plotCount = snapshot.stats.settlementPlots;
            _roadCells = snapshot.stats.roadCells;
            _riverCells = snapshot.stats.riverCells;
            _caveFloor = snapshot.stats.dungeonFloor;
            _gatePassed = snapshot.gatePassed;
            // The lowest cell already answers this, so the renderer never has to walk the
            // grid to find out whether the sea it is about to draw is behind the terrain.
            // toFloat() rather than a Fixed32 comparison: the sea level is a float here,
            // and it belongs to the RENDER — the only thing this decides is whether a water
            // pass is worth submitting, so there is nothing authoritative to preserve.
            _boundedHasSea = _low.toFloat() < seaLevel();

            // Vegetation: the producer level of the food web. Counted, never
            // integrated, so grazing a valley bare moves the number because the
            // plants are gone.
            _plants.clear();
            procgen::scatterVegetation(
                snapshot, _seed ^ 0x5EED11u, 2u,
                [](procgen::BiomeId biome) {
                    return biome == procgen::BiomeId::Forest || biome == procgen::BiomeId::Taiga ||
                           biome == procgen::BiomeId::Rainforest;
                },
                [this](core::i32 cellX, core::i32 cellZ) {
                    Plant plant;
                    plant.cellX = cellX;
                    plant.cellZ = cellZ;
                    _plants.push_back(plant);
                });
        }

        // The bounded world's field IS a window that happens to cover the whole map:
        // one type, so the herd's code cannot care which world it is walking.
        _living.scent().open(_gridWidth < _gridDepth ? _gridWidth : _gridDepth, 2u);
        _living.scent().centreOn(0, 0);
        _living.scent().field().setObstacles(_blocked);

        // Vegetation: one plant per forested cell, thinned so the map is not a
        // solid canopy. These ARE the producer level — the population is counted,
        // never integrated, so grazing a valley bare moves the number because the
        // plants are gone.
        _plants.clear();
        math::Random thin{_seed ^ 0x5EED11u};
        for (core::u32 z = 0u; z < _gridDepth; ++z)
            for (core::u32 x = 0u; x < _gridWidth; ++x)
            {
                const procgen::BiomeId biome = _biomes.at(x, z);
                const bool wooded = biome == procgen::BiomeId::Forest || biome == procgen::BiomeId::Taiga ||
                                    biome == procgen::BiomeId::Rainforest;
                if (!wooded || _blocked.at(x, z) != 0u || thin.below(2u) != 0u)
                    continue;
                Plant plant;
                plant.cellX = static_cast<core::i32>(x);
                plant.cellZ = static_cast<core::i32>(z);
                _plants.push_back(plant);
            }

        // The food web is the cartridge's; the producer's capacity is the one value a
        // document cannot state, because it is how much vegetation THIS seed grew.
        _living.configure(livingParams(), _livingRecipe, _seed);
        _living.seedWeb(static_cast<core::u32>(_plants.size()));
        _living.openScent(_gridWidth < _gridDepth ? _gridWidth : _gridDepth, _living.params().scentLayers);
        _living.scent().centreOn(0, 0);
        _living.scent().field().setObstacles(_blocked);
        seedHerd();

        _ticks = 0u;

        // Allocated once, reused every probe pass.
        // The props are grown once per world: an L-system expansion and a swept
        // profile per variant, not per instance.
        engine::PropLibraryParams propParams;
        propParams.treeSpecies = kTreeSpecies;
        propParams.rockVariants = kRockVariants;
        propParams.viewDistance = _viewDistance;
        propParams.nearDistance = kPlantNearDistance;
        propParams.canopyDistance = kPlantCanopyDistance;
        propParams.propDistance = kPropDistance;
        propParams.fogDensity = kFogDensity;
        propParams.rockAlbedo = kRockAlbedo;
        propParams.plantSalt = kPlantSalt;
        propParams.rockSalt = kRockSalt;
        _props.build(propParams, _seed);

        core::Log::info("TerrainWorld: world generated");
    }

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
     */
    void setInfinite(bool infinite)
    {
        if (_infinite == infinite)
            return;
        _infinite = infinite;
        _streamer.clear();

        if (!_infinite)
        {
            generate();
            return;
        }

        // ONE description of this world, scaled up to walk through. The chunk
        // parameters and the content rule used to be written out by hand right here,
        // beside the recipe they were supposed to agree with — two descriptions of one
        // world, in one function, with nothing to say when they drifted. See
        // procgen::endlessPlanFromRecipe for what the scaling means and why it is not
        // the identity.
        procgen::WorldRecipe seeded = _recipe;
        seeded.seed = _seed;
        seeded.terrain.seed = _seed;
        const procgen::EndlessPlan plan = procgen::endlessPlanFromRecipe(seeded, kChunkSize);
        _chunkParams = plan.chunk;

        _streamParams.generateRadius = kStreamRadius;
        // 1.5x: the hysteresis that stops a camera sitting on a boundary from
        // rebuilding the same chunk every tick forever.
        _streamParams.releaseRatio16 = 24u;
        _streamParams.directionWeight16 = 12u;
        // One chunk per tick. Generation is the expensive thing in this loop, and
        // a budget is what keeps a fixed step fixed.
        _streamParams.maxGeneratePerTick = 1u;
        _streamParams.maxReleasePerTick = 4u;

        _camera.setFocus(0.0f, 0.0f);
        // The body stands on the ground rather than at altitude zero: dropping it
        // from y=0 onto a mountain would spend the first seconds of the demo
        // underground, and onto a valley, falling.
        // The body is NOT placed here. Where the ground is depends on erosion and
        // on the rivers, and neither has run yet — the streamer is empty, so a query
        // would answer with raw noise. Deferred to @ref ensureBodySpawned, which
        // waits for terrain that actually exists.
        _bodySpawned = false;
        _camera.setEyeHeight(_bodyParams.eyeHeight.toFloat());

        // The pheromone window, not a world-sized grid.
        _living.scent().open(kFieldSpan, 2u);
        _living.scent().centreOn(0, 0);

        // The ecosystem comes with: a world you can walk across for an hour and never
        // meet anything is a landscape, not a world. The producer capacity is the
        // recipe's here — a streamed world has no total to count, only what is
        // resident, and tickVegetation corrects it every tick from what it finds.
        _living.configure(livingParams(), _livingRecipe, _seed);
        _living.seedWeb(0u);
        _living.openScent(kFieldSpan, _living.params().scentLayers);
        _living.scent().centreOn(0, 0);

        // One place says what the residency policy is: the chunk parameters, the
        // streaming policy (radii, release ratio, direction weight, per-tick budget
        // — procgen::StreamingParams' business, not a copy of it here) and the
        // memory ceiling the host asked for.
        // One call says what the endless world is: the terrain parameters, the
        // streaming policy and the memory ceiling. The rule below is the CONTENT —
        // where the sea and the snow line are, how much erosion, how dense the woods.
        // The river parameters come from the PLAN, not from a default-constructed member.
        // They used to be the latter, and the two then disagreed about where the sea was —
        // -1.0 against the rule's -4.0 — while the threshold stayed the one calibrated for a
        // map read from above. One derivation, one answer.
        _riverParams = plan.rivers;
        // One product, derived by the plan. See EndlessPlan::riverSurfaceRise.
        _riverSurfaceRise = plan.riverSurfaceRise;
        _caveMouthDrop = plan.rule.caveMouthDrop;

        // The swell runs with the PREVAILING WIND, which this world already models — the same
        // `climate.windDirection` its rain shadow is built from. Before this the two crest
        // directions were constants in the ripple function, so a lake and a river rippled
        // identically and both ignored the weather the recipe declares. Open water takes the
        // wind; a river's own current is the next step and needs the per-cell water surface.
        // Read from `seeded`, the very description the plan above was derived from, rather than
        // from the member: the wind has to be the one this world was built with.
        const core::u32 wind = seeded.climate.windDirection & 3u;
        _view.water.setDrift(static_cast<core::f32>(procgen::kNeighbor4X[wind]),
                             static_cast<core::f32>(procgen::kNeighbor4Z[wind]));

        // And the RENDER's sea level too, which was a fourth independent answer to "where is
        // the water" — the view profile's default of -1.0, against the classifier's -4.0 and
        // the river pass's own. The renderer floods a PLANE at this height, so a plane three
        // metres above the sea the world was classified against drowns the coast it drew as
        // land. §33 caught two answers to "how high is the ground"; this is the same shape.
        _view.surface.seaLevel = plan.rule.seaLevel;

        _streamer.configure(_chunkParams, _riverParams, _streamParams, _maxResident, plan.rule);

        seedHerd();

        core::Log::info("TerrainWorld: endless mode — WASD walks, the world streams around you");
    }

    /**
     * @brief Builds one chunk: its terrain, its biomes and what grows on it.
     *
     * Biomes are classified from a climate assembled at ABSOLUTE coordinates and
     * handed to the same @ref procgen::nearestBiomeProfile the bounded classifier
     * uses. Not a second classifier — the same one, with seamless inputs. A
     * per-chunk climate would have drawn a visible tint discontinuity on every
     * chunk border, which is the usual way a streamed world announces itself.
     */
    /// Keeps the pheromone window under the walker; ai::ScentWindow owns the policy,
    /// including the slack that stops a pacing walker from clearing it every step.
    void recentreField()
    {
        (void) _living.scent().follow(static_cast<core::i32>(_camera.focusX()),
                                      static_cast<core::i32>(_camera.focusZ()));
    }

    /**
     * @brief One authoritative tick of the player's body.
     *
     * Here rather than in @ref drainInput because a body is SIMULATION: it is
     * Fixed32, it folds, and it has to advance on the fixed step whatever the frame
     * rate is. Input is sampled on the render side and consumed here — the turn and
     * the jump are edges, so they are drained rather than read, or a slow frame
     * would apply the same jump to several ticks.
     */
    void stepBody(core::f32 dt)
    {
        if (!_embodied)
            return;
        ensureBodySpawned();

        engine::CharacterIntent intent = _intent;
        intent.turn = _pendingTurn;
        intent.jump = _jumpPressed;
        _pendingTurn = math::Fixed32{};
        _jumpPressed = false;

        // The SPACE, not the height. Above ground this is exactly the height field
        // with open sky over it; inside a warren it is the gallery the body is in,
        // and the body has no way to tell — which is the whole point of the seam.
        _body.step(_bodyParams, intent, math::Fixed32::fromFloat(dt),
                   [this](core::i32 x, core::i32 z, math::Fixed32 y) { return _streamer.spanAt(x, z, y); });

        // The camera is told where the body IS; it is not what moves. Reading the
        // authoritative yaw into the float camera is the allowed direction of that
        // dependency — simulation feeds presentation, never the reverse.
        _camera.setFocus(_body.x().toFloat(), _body.z().toFloat());
        _camera.setYaw(_body.yaw().toFloat());
    }

    /// World cell to a cell of the pheromone window, when it is inside it.
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
    void refreshShadows(ResidentChunk &chunk) const
    {
        render::HeightfieldPatchParams patch;
        patch.size = kChunkSize;
        patch.stride = 1u;
        patch.originX = chunk.coord.x * static_cast<core::i32>(kChunkSize);
        patch.originZ = chunk.coord.z * static_cast<core::i32>(kChunkSize);

        const procgen::ChunkParams &params = _streamer.chunkParams();
        _terrainSurface.fillShadowMask(
            patch, [&params](core::i32 x, core::i32 z) { return procgen::sampleWorldHeight(params, x, z).toFloat(); },
            [this, &chunk](auto &&emit) {
                for (core::u32 i = 0u; i < chunk.plants.size(); ++i)
                {
                    if (!chunk.plants[i].standing)
                        continue;
                    core::f32 height = 0.0f;
                    core::f32 spread = 0.0f;
                    _props.plantExtent(chunk.plants[i].cellX, chunk.plants[i].cellZ, height, spread);
                    emit(chunk.plants[i].cellX, chunk.plants[i].cellZ, height, spread);
                }
            },
            chunk.shade);
    }

    /// Generates and releases chunks around the focus, within the tick budget.
    /// The living budgets this world asks for; the host caps them through Config.
    [[nodiscard]] engine::LivingLayerParams livingParams() const noexcept
    {
        engine::LivingLayerParams params;
        params.maxBodies = kMaxCreatures;
        params.speciesCount = 2u;
        params.scentSpan = kFieldSpan;
        params.webPeriod = kWebPeriod;
        return params;
    }

    /// Fills the herd to what the freshly seeded web says should exist.
    void seedHerd()
    {
        math::Random stock{_seed ^ 0x57EA11u};
        for (core::u32 species = 0u; species + 1u < _living.web().species.size() && species < 2u; ++species)
        {
            const core::u32 wanted = _living.bodiesFor(species);
            for (core::u32 i = 0u; i < wanted; ++i)
                (void) _living.spawn(stock, species,
                                     [this](math::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
                                         return proposeSpawn(r, a, x, z);
                                     });
        }
    }

    /**
     * @brief Proposes a place a body may stand, which only this world can answer.
     *
     * In the endless world, around the FOCUS: a world with no origin would otherwise
     * put its herd at (0,0), where nobody ever goes. In the bounded one, any unblocked
     * cell of the grid.
     */
    [[nodiscard]] bool proposeSpawn(math::Random &random, core::u32 attempt, math::Fixed32 &outX,
                                    math::Fixed32 &outZ) const
    {
        (void) attempt;
        if (_infinite)
        {
            const core::f32 offsetX = static_cast<core::f32>(random.range(-20, 20));
            const core::f32 offsetZ = static_cast<core::f32>(random.range(-20, 20));
            outX = math::Fixed32::fromFloat(_camera.focusX() + offsetX);
            outZ = math::Fixed32::fromFloat(_camera.focusZ() + offsetZ);
            return standable(outX, outZ);
        }
        const core::u32 x = random.below(_gridWidth);
        const core::u32 z = random.below(_gridDepth);
        if (_blocked.at(x, z) != 0u)
            return false;
        outX = cellToWorldX(x);
        outZ = cellToWorldZ(z);
        return true;
    }

    /// Generates and releases chunks around the walker, within the tick budget.
    void streamChunks()
    {
        _streamer.update(_camera.focusX(), _camera.focusZ(), -render::OrbitCamera::sinOf(_camera.yaw()),
                         -render::OrbitCamera::cosOf(_camera.yaw()), [this](ResidentChunk &chunk) {
                             if (_terrainShadows)
                                 refreshShadows(chunk);
                         });
    }

    /**
     * @brief Tells the creature systems where the pheromone window sits now.
     *
     * The one thing the systems cannot derive: the window FOLLOWS the walker in the
     * endless world, so the mapping from a world cell to a field cell moves every
     * tick. It is passed as data — an origin and a size — rather than as a callback,
     * because data can be written down, folded and replayed.
     */
    void refreshCreatureView() { _creatures.setView(_living.fieldView()); }

    /// Eats one plant in the chunk that holds a world cell. @return true when it ate.
    [[nodiscard]] bool grazeEndless(core::i32 worldX, core::i32 worldZ)
    {
        for (core::u32 c = 0u; c < _streamer.size(); ++c)
        {
            ResidentChunk &chunk = _streamer.at(c);
            const core::i32 originX = chunk.coord.x * static_cast<core::i32>(kChunkSize);
            const core::i32 originZ = chunk.coord.z * static_cast<core::i32>(kChunkSize);
            if (worldX < originX || worldZ < originZ || worldX >= originX + static_cast<core::i32>(kChunkSize) ||
                worldZ >= originZ + static_cast<core::i32>(kChunkSize))
                continue;
            if (chunk.plants.empty())
                return false; // the cell's chunk was found; no other holds it
            return ecology::grazeAt(&chunk.plants[0], static_cast<core::u32>(chunk.plants.size()), worldX, worldZ, 1,
                                    _living.recipe().regrowthTicks);
        }
        return false;
    }

    /// Eats one plant near a world position, on the bounded grid. @return true when it ate.
    [[nodiscard]] bool grazeBounded(math::Fixed32 x, math::Fixed32 z)
    {
        if (_plants.empty())
            return false;
        core::u32 cx = 0u;
        core::u32 cz = 0u;
        if (!worldToCell(x, z, cx, cz))
            return false;
        return ecology::grazeAt(&_plants[0], static_cast<core::u32>(_plants.size()), static_cast<core::i32>(cx),
                                static_cast<core::i32>(cz), 1, _living.recipe().regrowthTicks);
    }

    /**
     * @brief Regrowth, and the producer population it implies.
     *
     * The producer level is what is actually STANDING: a streamed world has no total
     * to count, only a neighbourhood, and the herd only eats what it can reach.
     */
    /// Regrowth, then the producer population it implies.
    void tickVegetation()
    {
        core::u32 standing = 0u;
        if (_infinite)
        {
            for (core::u32 c = 0u; c < _streamer.size(); ++c)
            {
                ResidentChunk &chunk = _streamer.at(c);
                if (!chunk.plants.empty())
                    standing += ecology::tickPlants(&chunk.plants[0], static_cast<core::u32>(chunk.plants.size()));
            }
        }
        else if (!_plants.empty())
        {
            standing = ecology::tickPlants(&_plants[0], static_cast<core::u32>(_plants.size()));
        }
        _living.setProducerPopulation(standing);
    }

    /// Brings the bodies in line with the census; engine::LivingLayer keeps the ratio.
    void reconcile()
    {
        _living.reconcile(_ticks, [this](math::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
            return proposeSpawn(r, a, x, z);
        });
    }

    // ── Vegetation and props ─────────────────────────────────────────────────
    //
    // engine::PropLibrary grows one mesh per species and per boulder variant, decides
    // from a cell hash which one stands where, and batches the draws. Nothing about
    // that is this game's, so none of it is here.

    /// The one ground height, from the streamer: the field that is DRAWN.
    [[nodiscard]] core::f32 groundAt(core::i32 worldX, core::i32 worldZ) const noexcept
    {
        return _streamer.groundAt(worldX, worldZ);
    }

    // ── Surfaces ─────────────────────────────────────────────────────────────
    //
    // The grain and its mip chain, the Lambert and physically based paths, the fog,
    // the water with its Fresnel and its probe, and the shadow masks are all
    // TerrainSurface's — configured by the host's HostProfile rather than by
    // constants in a game. What is left here is what only this world knows: which
    // colour a biome is, and where its bed lies.

    /// Bed height under a world point, for the water's depth tint.
    [[nodiscard]] core::f32 bedHeightAt(core::f32 worldX, core::f32 worldZ) const noexcept
    {
        if (_infinite)
            return groundAt(static_cast<core::i32>(worldX), static_cast<core::i32>(worldZ));
        core::u32 cx = 0u;
        core::u32 cz = 0u;
        if (_height.empty() || !worldToCell(math::Fixed32::fromFloat(worldX), math::Fixed32::fromFloat(worldZ), cx, cz))
            return kFloor;
        return _height.at(cx, cz).toFloat();
    }

    /**
     * @brief One frame, handed to engine::TerrainRenderer.
     *
     * What this world supplies is the three things a renderer cannot know: its
     * palette, how high its ground is, and — in the bounded case — what a cell looks
     * like in the data view the player selected.
     */
    void renderScene(const render::RenderTarget &rt) const noexcept
    {
        render::clearTarget(rt, 0x000A0E18u);

        engine::TerrainDrawParams params;
        params.chunkSize = kChunkSize;
        params.lodRings = _lodRings;
        params.seaLevel = _view.surface.seaLevel;
        params.ambient = _view.surface.ambient;
        // The creatures' colours are content too: a world of white hares and one of
        // black wolves are two worlds, and neither is a host budget.
        // A jumping body is not on the ground, so the eye may not be derived from
        // the terrain under it.
        if (_infinite && _embodied)
        {
            params.useFocusHeight = true;
            params.focusHeight = _body.y().toFloat();
        }
        params.grazerTint = _view.grazerTint;
        params.hunterTint = _view.hunterTint;
        params.bodyScale = _view.bodyScale;
        params.skirtDrop = kSkirtDrop;
        params.chunkCentreY = kChunkCentreY;
        params.chunkHalfHeight = kChunkHalfHeight;
        // The same number the generator carved with, from the same plan — see
        // TerrainDrawParams::riverSurfaceRise for why it may not be a second answer.
        params.riverSurfaceRise = _riverSurfaceRise;
        params.boundedHasSea = _boundedHasSea;
        // Same number the generator cut with. See TerrainDrawParams::caveMouthDrop.
        params.caveMouthDrop = _caveMouthDrop;
        // No `underground` here any more, and it is worth saying why rather than just
        // deleting it: it used to be `_embodied && _body.isEnclosed()`, and detaching
        // the camera made `_embodied` false — so in the orbit view the cave path was
        // switched off entirely, sky and all, with the eye sixty cells back and often
        // inside a hill. The renderer asks the streamer about the EYE now, which is the
        // thing it actually needs to know.
        params.caveDrawRadius = _caveDrawRadius;

        const auto palette = [this](procgen::BiomeId biome) { return biomeColour(biome); };

        if (_infinite)
        {
            _lastTriangles =
                _renderer.drawStreamed(rt, _camera, _streamer, _terrainSurface, _props, registry(), params, _frames,
                                       palette, [this](core::i32 x, core::i32 z) { return groundAt(x, z); });
            return;
        }

        if (_height.empty())
            return;
        _lastTriangles = _renderer.drawBounded(
            rt, _camera, _terrainSurface, _props, registry(), _gridWidth, _gridDepth,
            _plants.empty() ? nullptr : &_plants[0], static_cast<core::u32>(_plants.size()), params, palette,
            [this](core::u32 x, core::u32 z) { return _height.at(x, z).toFloat(); },
            [this](core::u32 x, core::u32 z) { return cellColour(x, z); },
            [this](core::i32 x, core::i32 z) {
                return bedHeightAt(static_cast<core::f32>(x), static_cast<core::f32>(z));
            });
    }

    /**
     * @brief The palette, which is very nearly the whole identity of a world.
     *
     * Chosen to read at 480x300 through a nearest-neighbour upscale, and that rules
     * out subtlety: adjacent biomes have to differ in VALUE, not only in hue, or a
     * coastline turns to mush at that resolution.
     */
    [[nodiscard]] static core::u32 builtInBiomeColour(procgen::BiomeId biome) noexcept
    {
        switch (biome)
        {
        case procgen::BiomeId::Ocean: return 0x00123A6Au;
        case procgen::BiomeId::Lake: return 0x00295A8Cu;
        case procgen::BiomeId::Beach: return 0x00D4C48Cu;
        case procgen::BiomeId::Snow: return 0x00F0F2F6u;
        case procgen::BiomeId::Tundra: return 0x0099A08Fu;
        case procgen::BiomeId::Taiga: return 0x00335F50u;
        case procgen::BiomeId::Rock: return 0x00706A66u;
        case procgen::BiomeId::Desert: return 0x00D9B86Bu;
        case procgen::BiomeId::Savanna: return 0x00B3AD57u;
        case procgen::BiomeId::Grassland: return 0x006B994Du;
        case procgen::BiomeId::Forest: return 0x00337238u;
        case procgen::BiomeId::Rainforest: return 0x001C5C2Du;
        case procgen::BiomeId::Marsh: return 0x004F6B54u;
        case procgen::BiomeId::Count: break;
        }
        // Magenta on purpose: an unmapped biome must be impossible to mistake for a
        // design choice.
        return 0x00FF00FFu;
    }

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
    [[nodiscard]] core::u32 cellColour(core::u32 x, core::u32 z) const noexcept
    {
        if (_shading == Shading::Height)
        {
            const core::f32 span = (_high - _low).toFloat();
            const core::f32 t = span > 0.0f ? (_height.at(x, z) - _low).toFloat() / span : 0.5f;
            return render::heatRamp(t);
        }
        if (_shading == Shading::Moisture)
            return _moisture.empty() ? 0x00404040u : render::heatRamp(_moisture.at(x, z).toFloat());

        if (!_rivers.empty() && _rivers.at(x, z) != 0u)
            return 0x002A6BB8u;
        if (!_settlement.empty())
        {
            switch (_settlement.at(x, z))
            {
            case procgen::SettlementCell::Road: return 0x00595049u;
            case procgen::SettlementCell::Plaza: return 0x008C7F70u;
            case procgen::SettlementCell::Plot: return 0x00D18C38u;
            default: break;
            }
        }
        if (!_roads.empty() && _roads.at(x, z) != 0u)
            return 0x003D3833u;
        return biomeColour(_biomes.at(x, z));
    }

    void drawShadowedText(core::u32 pitchPixels, core::u32 x, core::u32 y, const HudLine &line,
                          core::u32 colour) const noexcept
    {
        render::drawShadowedText8x16(_surface.buffer, pitchPixels, x, y, line.c_str(), colour);
    }

    void drawHud() const noexcept
    {
        const core::u32 pitchPixels = _surface.pitch / 4u;
        HudLine line;

        if (_infinite)
            line.text("LPLKERNEL ENDLESS WORLD  seed ")
                .number(_seed)
                .text("  chunks ")
                .number(static_cast<core::u32>(_streamer.size()))
                .text(" of ")
                .number(_maxResident);
        else
            line.text("LPLKERNEL WORLD VIEWER  seed ")
                .number(_seed)
                .text("  ")
                .number(_gridWidth)
                .text("x")
                .number(_gridDepth);
        drawShadowedText(pitchPixels, 8u, 8u, line, 0x00FFAA22u);

        if (_infinite)
        {
            // A signed coordinate, printed signed: the previous readout took the
            // absolute value and appended "(west/north)", which is ambiguous the
            // moment one axis is negative and the other is not.
            line.clear()
                .text("at ")
                .integer(static_cast<core::i32>(_camera.focusX()))
                .text(",")
                .integer(static_cast<core::i32>(_camera.focusZ()))
                .text("  built ")
                .number(_streamer.generatedCount())
                .text("  released ")
                .number(_streamer.releasedCount());
            drawShadowedText(pitchPixels, 8u, 26u, line, 0x00C8C8C0u);

            line.clear()
                .text(_camera.isFirstPerson() ? "first person  eye " : "orbit  distance ")
                .decimal(_camera.isFirstPerson() ? _camera.eyeHeight() : _camera.distance())
                .text("  LOD rings ")
                .number(_lodRings)
                .text("  tris ")
                .number(_lastTriangles);
            drawShadowedText(pitchPixels, 8u, 44u, line, 0x00A0B4C8u);

            // The body, written out because it is the only way to tell a jump from a
            // fall from a slide without guessing at the picture.
            line.clear()
                .text(_embodied ? "body " : "detached ")
                .text(_body.isSliding() ? "sliding" : (_body.isGrounded() ? "grounded" : "airborne"))
                .text("  y ")
                .decimal(_body.y().toFloat())
                .text("  speed ")
                .decimal(_body.groundSpeed().toFloat())
                .text("  jumps ")
                .number(_body.jumpCount())
                .text("  blocked ")
                .number(_body.blockedCount());
            drawShadowedText(pitchPixels, 8u, 62u, line, _body.isGrounded() ? 0x00A0B4C8u : 0x00E0C070u);

            line.clear()
                .text("herd ")
                .number(countSpecies(0u))
                .text(" grazers  ")
                .number(countSpecies(1u))
                .text(" hunters  grazed ")
                .number(_living.grazedCount());
            drawShadowedText(pitchPixels, 8u, 80u, line, 0x0060FF80u);

            line.clear()
                .text("cost% sim ")
                .number(_simShare)
                .text("  scene ")
                .number(_sceneShare)
                .text("  blit ")
                .number(_blitShare)
                .text("  present ")
                .number(_presentShare)
                .text("  | sky ")
                .number(_skyShare)
                .text(" gnd ")
                .number(_groundShare)
                .text(" wat ")
                .number(_waterShare)
                .text(" prop ")
                .number(_propShare);
            drawShadowedText(pitchPixels, 8u, 98u, line, 0x00FFAA22u);

            line.clear()
                .text("scent window ")
                .number(kFieldSpan)
                .text(" cells, recentred ")
                .number(_living.scent().recentres())
                .text(" times");
            drawShadowedText(pitchPixels, 8u, 116u, line, 0x00A0B4C8u);

            line.clear()
                .text("drawn ")
                .number(_renderer.view().stats().drawn)
                .text(" of ")
                .number(static_cast<core::u32>(_streamer.size()))
                .text(" chunks, culled ")
                .number(_renderer.view().stats().culled)
                // Windowed, not cumulative: a rate counted from boot is dominated
                // by the seconds before the world finished streaming, which is
                // exactly the period that has nothing to do with what is being
                // measured. The window resets whenever the mode changes.
                .text("  boxes ")
                .number(_renderer.view().stats().considered)
                .text("-")
                .number(_renderer.view().stats().culled)
                .text("  f/100t ")
                .number(_windowTicks == 0u ? 0u : (_windowFrames * 100u) / _windowTicks)
                .text(_pbrSurface ? "  pbr" : (_perPixelSurface ? "  per-pixel" : "  flat"))
                .text("  sky/")
                .number(_skyBlock);
            drawShadowedText(pitchPixels, 8u, 134u, line, 0x00A0B4C8u);

            // ⚠ The landmark readout used to live BELOW this branch, and this branch
            // returns — so in endless mode, the only mode that has landmarks in it, it
            // was never drawn at all. Not overdrawn: unreachable. It is the line that
            // answers "I cannot see a cave entrance" against "there are no cave
            // entrances", which is the distinction TerrainStreamer's own docstring says
            // no amount of looking can make, and it had never once been on screen.
            core::u32 navigable = 0u;
            const core::u32 warrens = _streamer.residentWarrens(&navigable);
            line.clear()
                .text("landmarks: ")
                .number(_streamer.residentCaveMouths())
                .text(" mouths  ")
                .number(warrens)
                .text(" caves (")
                .number(navigable)
                .text(" deep)  ")
                .number(_streamer.residentBuildings())
                .text(" buildings");
            drawShadowedText(pitchPixels, 8u, 152u, line, 0x00C8A050u);

            // Where the body IS, vertically. The only readout that separates "the cave
            // is unlit" from "I never got inside it" — two completely different faults
            // that look identical from a dark screen.
            line.clear()
                .text(_body.isEnclosed() ? "UNDERGROUND  y " : "surface      y ")
                .integer(static_cast<core::i32>(_body.y().toFloat()))
                .text("  ceiling ")
                .integer(_body.isEnclosed() ? static_cast<core::i32>(_body.ceilingHeight().toFloat()) : 0)
                .text("  head ")
                .number(_body.headBumpCount())
                .text("  ducked ")
                .number(_body.duckedCount());
            drawShadowedText(pitchPixels, 8u, 170u, line, _body.isEnclosed() ? 0x00E08040u : 0x00708090u);

            image::drawText8x16(
                _surface.buffer, pitchPixels, 8u, _surface.height - 20u,
                "WASD=walk SPACE=jump C=sprint mouse=look V=detach F=view I/K=tilt T/Y/R/G=shading O=bounded X=exit",
                0x00808890u);
            return;
        }

        line.clear()
            .text(shadingName())
            .number(presentBiomes())
            .text(" biomes  ")
            .number(standingPlants())
            .text(" plants of ")
            .number(static_cast<core::u32>(_plants.size()));
        drawShadowedText(pitchPixels, 8u, 26u, line, 0x00C8C8C0u);

        line.clear()
            .text("herd ")
            .number(countSpecies(0u))
            .text(" grazers  ")
            .number(countSpecies(1u))
            .text(" hunters  grazed ")
            .number(_living.grazedCount());
        drawShadowedText(pitchPixels, 8u, 44u, line, 0x0060FF80u);

        line.clear()
            .text("cartridge: ")
            .number(_plotCount)
            .text(" plots  ")
            .number(_roadCells)
            .text(" road cells  ")
            .number(_propCount)
            .text(" props");
        drawShadowedText(pitchPixels, 8u, 62u, line, 0x00A0B4C8u);

        line.clear()
            .text("rivers ")
            .number(_riverCells)
            .text("  cave floor ")
            .number(_caveFloor)
            .text("  gate ")
            .number(_gatePassed ? 1u : 0u);
        drawShadowedText(pitchPixels, 8u, 80u, line, 0x00A0B4C8u);

        image::drawText8x16(_surface.buffer, pitchPixels, 8u, _surface.height - 20u,
                            "WASD=cam Q/E=zoom N=new seed B=shading O=endless X=exit", 0x00808890u);
    }

    [[nodiscard]] const char *shadingName() const noexcept
    {
        switch (_shading)
        {
        case Shading::Height: return "shading height   ";
        case Shading::Moisture: return "shading moisture ";
        case Shading::Biome:
        case Shading::Count: break;
        }
        return "shading biome    ";
    }

    // ── Input ────────────────────────────────────────────────────────────────

    /**
     * @brief The movement bindings this class owns, then the game's own keys.
     *
     * WASD, the turn and tilt pair, the zoom, the view toggle and the shading
     * switches belong here because they act on state this class owns. A game adds
     * its own through @ref onKey rather than re-binding the lot.
     */
    void drainInput(engine::WorldContext &context)
    {
        drainPointer(context);
        sampleHeldKeys(context);

        char key;
        while (context.platform.input().tryPopCharacter(key))
        {
            switch (key)
            {
            // In the endless world WASD WALKS, because there is somewhere to walk
            // to; in the bounded one it orbits, because there is not. Forward is
            // where the camera looks, which is the only reading of "forward" a
            // player ever means.
            // When the body is walking from HELD keys, a typed character must not
            // also nudge it — the two would stack and every keystroke would add a
            // lurch on top of the steady walk. The typed form stays as the fallback
            // for a backend that cannot report key states at all, and keeps driving
            // the bounded world, which has no body to move.
            // No AZERTY aliases in the TYPED path: 'q' is already zoom here, and a
            // key cannot mean two things. The held-key sampler accepts both layouts
            // and is what actually walks the body; this switch is only the fallback
            // for a backend with no key states, where a QWERTY binding is the one
            // that was always there.
            case 'w':
                if (!_infinite)
                    _camera.tilt(0.06f);
                else if (!bodyDrivesMovement(context))
                    nudge(1.0f, 0.0f);
                break;
            case 's':
                if (!_infinite)
                    _camera.tilt(-0.06f);
                else if (!bodyDrivesMovement(context))
                    nudge(-1.0f, 0.0f);
                break;
            // Jump. An EDGE, buffered by the controller, so pressing it a few ticks
            // early still fires on landing instead of being dropped.
            case ' ': _jumpPressed = true; break;
            case 'v':
                _embodied = !_embodied;
                _camera.setEyeHeight(_embodied ? _bodyParams.eyeHeight.toFloat() : kDetachedEyeHeight);
                break;
            case 'a':
                if (!_infinite)
                    _camera.turn(-0.08f);
                else if (!bodyDrivesMovement(context))
                    nudge(0.0f, -1.0f);
                break;
            case 'd':
                if (!_infinite)
                    _camera.turn(0.08f);
                else if (!bodyDrivesMovement(context))
                    nudge(0.0f, 1.0f);
                break;
            case 'j': _camera.turn(-0.10f); break;
            case 'l': _camera.turn(0.10f); break;
            case 'i': _camera.tilt(0.06f); break;
            case 'k': _camera.tilt(-0.06f); break;
            case 'o': setInfinite(!_infinite); break;
            // First person and orbit are one camera at two distances: collapse the
            // orbit onto the eye and the same code stands in the world. Nothing
            // switches, which is why there is no second camera to keep in step.
            case 'r':
                _reflectionProbeOn = !_reflectionProbeOn;
                resetRateWindow();
                break;
            case 'g':
                _pbrSurface = !_pbrSurface;
                resetRateWindow();
                break;
            case 'y':
                _skyBlock = _skyBlock == 1u ? 3u : 1u;
                resetRateWindow();
                break;
            case 't':
                _perPixelSurface = !_perPixelSurface;
                resetRateWindow();
                break;
            case 'f':
                _camera.setFirstPerson(!_camera.isFirstPerson());
                _camera.setPitch(_camera.isFirstPerson() ? kFirstPersonPitch : kOrbitPitch);
                break;
            case 'q': _camera.zoom(0.88f, 1.6f, 160.0f); break;
            case 'e': _camera.zoom(1.14f, 1.6f, 160.0f); break;
            case 'b':
                _shading = static_cast<Shading>((static_cast<core::u32>(_shading) + 1u) %
                                                static_cast<core::u32>(Shading::Count));
                break;
            case 'n':
                // A whole new world, generated in ring 0 on a keystroke. This is
                // the demo's one real claim, and it is the only key that proves it.
                _seed = _seed * 1664525u + 1013904223u;
                generate();
                break;
            case 'x':
            case 27:
                if (context.engine != nullptr)
                    context.engine->requestShutdown();
                break;
            // Anything this class does not bind goes to the game, which is the only
            // one that knows what its own keys mean.
            default: break;
            }
        }
    }

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
     */
    void drainPointer(engine::WorldContext &context)
    {
        core::i32 deltaX = 0;
        core::i32 deltaY = 0;
        core::u32 buttons = 0u;
        core::i32 turnAccumulator = 0;
        core::i32 tiltAccumulator = 0;

        while (context.platform.input().tryPopPointerMotion(deltaX, deltaY, buttons))
        {
            turnAccumulator += deltaX;
            tiltAccumulator += deltaY;
            _pointerButtons = buttons;
            ++_pointerPackets;
        }

        // Reported on the serial console rather than the HUD: the body line is
        // already at the width the overlay can hold, and a counter that is silently
        // truncated reads as a missing field rather than as a zero.
        //
        // Three DISTINCT sentences rather than two counters, because they are three
        // distinct diagnoses and the logger takes a string, not a format. No
        // interrupt at all is a wiring problem; interrupts without packets is a
        // decoding one; packets arriving while the view stays still is a consumer
        // one. Told apart here, they each point at one file.
        if (_pointerPackets != 0u && _pointerReported == 0u)
        {
            _pointerReported = _pointerPackets;
            core::Log::info("pointer: motion received — the device and the driver are fine");
        }
        else if (_pointerPackets == 0u && _frames == 240u)
        {
            core::Log::info(context.platform.input().pointerInterruptCount() != 0u ?
                                "pointer: interrupts arrive but no packet is assembled" :
                                "pointer: no interrupt from the device at all");
        }

        // Turning goes to the BODY when there is one, because the heading picks the
        // walk direction and that makes it authoritative. Tilt stays on the camera:
        // looking up does not move you, so it is presentation and may be float.
        if (turnAccumulator != 0)
        {
            if (_embodied && _infinite)
                _pendingTurn =
                    _pendingTurn - math::Fixed32::fromFloat(static_cast<core::f32>(turnAccumulator) * kLookSensitivity);
            else
                _camera.turn(static_cast<core::f32>(turnAccumulator) * kLookSensitivity);
        }
        if (tiltAccumulator != 0)
            _camera.tilt(static_cast<core::f32>(tiltAccumulator) * kLookSensitivity);
    }

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
     */
    void sampleHeldKeys(engine::WorldContext &context)
    {
        const platform::IInputBackend &input = context.platform.input();
        if (!input.hasKeyStates())
            return;

        // Two characters per direction: the same PHYSICAL keys type 'w'/'a' on
        // QWERTY and 'z'/'q' on AZERTY, and a walker should not care which keyboard
        // is plugged in. Accepting both costs one test and strands nobody.
        const bool ahead = input.isKeyHeld('w') || input.isKeyHeld('z');
        const bool back = input.isKeyHeld('s');
        const bool left = input.isKeyHeld('a') || input.isKeyHeld('q');
        const bool right = input.isKeyHeld('d');

        // Opposite keys held together cancel. Letting the later one win would make a
        // player who rolls their fingers across both drift in whichever order the
        // scancodes happened to arrive.
        _intent.forward = (ahead == back) ? math::Fixed32{} : (ahead ? math::Fixed32::one() : -math::Fixed32::one());
        _intent.strafe = (left == right) ? math::Fixed32{} : (right ? math::Fixed32::one() : -math::Fixed32::one());
        _intent.sprint = input.isKeyHeld('c');
    }

    // ── Small helpers ────────────────────────────────────────────────────────

    /// Walks along the camera's heading, in world cells.

    void walk(core::f32 sign) noexcept { _camera.walk(sign * kWalkStep); }

    /// Sidesteps: the heading rotated a quarter turn.
    void strafe(core::f32 sign) noexcept { _camera.strafe(sign * kWalkStep); }

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
     */
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
    void ensureBodySpawned()
    {
        if (_bodySpawned || _streamer.size() < 9u)
            return;
        spawnBody();
        _bodySpawned = true;
    }

    void spawnBody()
    {
        const auto ground = [this](core::i32 x, core::i32 z) { return _streamer.groundHeightAt(x, z); };
        const auto space = [this](core::i32 x, core::i32 z, math::Fixed32 y) { return _streamer.spanAt(x, z, y); };
        const math::Fixed32 shore = math::Fixed32::fromFloat(seaLevel() + 0.5f);

        const core::i32 reach = spawnSearchCells();
        for (core::i32 radius = 0; radius <= reach; ++radius)
        {
            for (core::i32 offsetZ = -radius; offsetZ <= radius; ++offsetZ)
                for (core::i32 offsetX = -radius; offsetX <= radius; ++offsetX)
                {
                    // Only the RING, not the filled square: the inner cells were
                    // already rejected at a smaller radius, and re-testing them
                    // makes the search quadratic in the radius for nothing.
                    const core::i32 ax = offsetX < 0 ? -offsetX : offsetX;
                    const core::i32 az = offsetZ < 0 ? -offsetZ : offsetZ;
                    if (ax != radius && az != radius)
                        continue;

                    const math::Fixed32 here = ground(offsetX, offsetZ);
                    if (here < shore)
                        continue;
                    // Not on a face it would immediately slide off, either.
                    const math::Fixed32 slopeX = ground(offsetX + 1, offsetZ) - here;
                    const math::Fixed32 slopeZ = ground(offsetX, offsetZ + 1) - here;
                    if (slopeX > _bodyParams.maxSlope || -slopeX > _bodyParams.maxSlope ||
                        slopeZ > _bodyParams.maxSlope || -slopeZ > _bodyParams.maxSlope)
                        continue;

                    // Spawned at the SURFACE and settled from there: the search above
                    // looked at the terrain, so the gap the body means is the one over
                    // the terrain and not whatever gallery happens to run under it.
                    _body.placeAt(math::Fixed32::fromInt(offsetX), math::Fixed32::fromInt(offsetZ), here, space);
                    _camera.setFocus(static_cast<core::f32>(offsetX), static_cast<core::f32>(offsetZ));
                    return;
                }
        }

        // Nothing dry within the search: an all-ocean seed is a legitimate world, and
        // starting at the origin is a better answer than refusing to start. SAID, though
        // — silently drowning the walker and letting the player work out why is how this
        // went unnoticed while the reach was too short to clear a bay.
        core::Log::warn("TerrainWorld: no dry ground within the spawn search — starting in the water");
        _body.placeAt(math::Fixed32{}, math::Fixed32{}, ground(0, 0), space);
    }

    /// Whether the body is what movement keys act on this frame.
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
    void nudge(core::f32 forward, core::f32 strafeAmount) noexcept
    {
        if (!_embodied)
        {
            _camera.walk(forward * kWalkStep);
            _camera.strafe(strafeAmount * kWalkStep);
            return;
        }
        _intent.forward = math::Fixed32::fromFloat(forward);
        _intent.strafe = math::Fixed32::fromFloat(strafeAmount);
    }

    [[nodiscard]] math::Fixed32 cellToWorldX(core::u32 cell) const noexcept
    {
        return math::Fixed32::fromInt(static_cast<core::i32>(cell)) -
               math::Fixed32::fromInt(static_cast<core::i32>(_gridWidth / 2u));
    }

    [[nodiscard]] math::Fixed32 cellToWorldZ(core::u32 cell) const noexcept
    {
        return math::Fixed32::fromInt(static_cast<core::i32>(cell)) -
               math::Fixed32::fromInt(static_cast<core::i32>(_gridDepth / 2u));
    }

    [[nodiscard]] bool worldToCell(math::Fixed32 x, math::Fixed32 z, core::u32 &outX, core::u32 &outZ) const noexcept
    {
        const core::i32 gx = (x + math::Fixed32::fromInt(static_cast<core::i32>(_gridWidth / 2u))).toInt();
        const core::i32 gz = (z + math::Fixed32::fromInt(static_cast<core::i32>(_gridDepth / 2u))).toInt();
        if (gx < 0 || gz < 0 || static_cast<core::u32>(gx) >= _gridWidth || static_cast<core::u32>(gz) >= _gridDepth)
            return false;
        outX = static_cast<core::u32>(gx);
        outZ = static_cast<core::u32>(gz);
        return true;
    }

    /**
     * @brief Whether an animal may stand at a world position.
     *
     * The endless world has no obstacle mask — there is no grid to precompute one
     * over — so the question is answered from the height alone, which
     * @c sampleWorldHeight can answer anywhere, including in chunks nobody has
     * generated. That is exactly what a creature walking toward the horizon needs.
     */
    [[nodiscard]] core::u32 standingPlants() const noexcept
    {
        return _plants.empty() ? 0u : ecology::countStanding(&_plants[0], static_cast<core::u32>(_plants.size()));
    }

    [[nodiscard]] core::u32 countSpecies(core::u32 species) const noexcept
    {
        return _living.herd().countSpecies(species);
    }

    [[nodiscard]] core::u32 presentBiomes() const noexcept
    {
        core::u32 present = 0u;
        for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
            if (_biomeCounts[i] != 0u)
                ++present;
        return present;
    }

    static constexpr core::f32 kFloor = -6.0f;
    static constexpr core::f32 kCeiling = 14.0f;
    /// Where the sea is: the RECIPE's, never a constant here.
    ///
    /// There was a `kSeaLevel = -1.0f` in this block, and with it the shipped
    /// cartridge had three answers to one question: the classifier called everything
    /// at or below -4 (the engine default, since the document said nothing) Ocean, the
    /// view profile drew water at -1, and this constant blocked walking below -1. The
    /// band between them was land you could not walk on, under water, that the scatter
    /// was free to plant trees in.
    [[nodiscard]] core::f32 seaLevel() const noexcept { return _recipe.biomes.seaLevel; }
    /// One fixed tick at 60 Hz, in seconds.
    // The frame buffers live in BSS, like the other sample's: a kernel stack
    // cannot hold 480x300 pixels plus their depth, and the heap should not have to.
    static core::u32 _color[kRenderWidth * kRenderHeight];
    static core::f32 _depth[kRenderWidth * kRenderHeight];

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

    // ── The endless world ────────────────────────────────────────────────────
    /// The resident set: procgen::ChunkResidency owns the policy, this owns the payload.

    procgen::ChunkParams _chunkParams{};
    procgen::StreamingParams _streamParams{};
    procgen::EndlessRiverParams _riverParams{};
    /// World cell the endless stigmergy window's corner sits on.
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

    /// The pheromone substrate, as a window on absolute coordinates.
    // The field's rates are the cartridge's too; keeping a second copy here is how
    // a document ends up describing one ecosystem while the host runs another.

    /// The living population: ecology::Herd owns the movement and the census.
    lpl::pmr::vector<Plant> _plants;

    /// One grown plant per species, and the transforms are the instances.
    static constexpr core::u32 kTreeSpecies = 3u;

    /// The engine's surface layer: sky, sun, grain, shadows, water, reflection.
    /// The props: grown once, placed by a cell hash, drawn batched.
    /// The endless world's chunks: residency, generation, one ground height.
    /// Population, bodies and the scent they read: engine::LivingLayer.
    engine::LivingLayer _living;
    /// Owned by the scheduler; held to keep their window in step with the scent's.
    /// The six systems of an animal's tick, ordered by engine::systems::CreatureStage.
    engine::systems::CreaturePipeline _creatures;

    mutable engine::TerrainStreamer _streamer;

    mutable engine::PropLibrary _props;

    mutable engine::TerrainSurface _terrainSurface;

    render::OrbitCamera _camera{};
    /// The sun, the sky, the water, the grain and the probe live in the facility;
    /// only the probe's BUFFERS stay here, because they must sit in BSS.
    static constexpr core::u32 kProbeWidth = 240u;
    static constexpr core::u32 kProbeHeight = 150u;
    static core::u32 _probeColor[kProbeWidth * kProbeHeight];
    static core::f32 _probeDepth[kProbeWidth * kProbeHeight];
    mutable render::ReflectionProbe _probe{};
    mutable render::Texture _probePixels{};
    bool _reflectionProbeOn{true};
    /// Boulder meshes, swept once from a tessellated profile and instanced.
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
    /// Per-frame prop queue: ordering, batching and level of detail live in it.
    /// Culling, ordering and level of detail: render::ChunkedTerrainView.
    /// The frame: engine::TerrainRenderer owns the passes.
    mutable engine::TerrainRenderer _renderer;

    /// Salts that keep the two scatter layers from agreeing with each other.
    static constexpr core::u32 kPlantSalt = 0x11u;
    static constexpr core::u32 kRockSalt = 0x5Bu;

    core::u32 _skyBlock{1u};
    /// Read from the Config at init; see the comment where they are assigned.
    core::u32 _lodRings{3u};
    core::f32 _viewDistance{70.0f};
    bool _terrainShadows{true};
    core::u32 _shadowChunksPerTick{1u};
    core::u32 _maxResident{56u};
    bool _pbrSurface{false};
    mutable core::u32 _windowFrames{0u};
    core::u32 _windowTicks{0u};
    Shading _shading{Shading::Biome};
    core::u32 _seed{1337u};
    core::u32 _ticks{0u};
    core::u32 _frames{0u};
    /// Buttons held at the last pointer report; bit 0 left, 1 right, 2 middle.
    core::u32 _pointerButtons{0u};
    /**
     * @brief Whether the host reported a pointing device, and what it has sent.
     *
     * On the HUD because "the mouse does not work" has three completely different
     * causes — no device found at boot, a device found but silent, or motion
     * arriving and being ignored — and they are indistinguishable from the picture.
     * A count that climbs while the view stays still says the driver is fine and the
     * consumer is not; a count stuck at zero says the opposite.
     */
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
    /// Sampled on the render side, consumed by the authoritative step.
    engine::CharacterIntent _intent{};
    /// False until terrain exists to stand on; see @ref ensureBodySpawned.
    bool _bodySpawned{false};
    /// Turn accumulated by the pointer since the last authoritative step.
    math::Fixed32 _pendingTurn{};
    /// Jump pressed since the last authoritative step; consumed by it.
    bool _jumpPressed{false};
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
// In BSS, like the frame itself: a 240x150 colour+depth pair is 288 KiB, and the
// kernel stack is nowhere near that. The scene learned this lesson once already.
inline core::u32 TerrainWorld::_probeColor[TerrainWorld::kProbeWidth * TerrainWorld::kProbeHeight];
inline core::f32 TerrainWorld::_probeDepth[TerrainWorld::kProbeWidth * TerrainWorld::kProbeHeight];

} // namespace lpl::samples

#endif // LPL_SAMPLES_TERRAINWORLD_HPP
