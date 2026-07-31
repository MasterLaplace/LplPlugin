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
#    include <lpl/engine/Engine.hpp>
#    include <lpl/engine/LivingLayer.hpp>
#    include <lpl/engine/PropLibrary.hpp>
#    include <lpl/engine/TerrainRenderer.hpp>
#    include <lpl/engine/TerrainStreamer.hpp>
#    include <lpl/engine/TerrainSurface.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/image/Font8x16.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/platform/IPlatform.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/Botany.hpp>
#    include <lpl/procgen/ChunkResidency.hpp>
#    include <lpl/procgen/ChunkTerrain.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Climate.hpp>
#    include <lpl/procgen/FixedMath.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>
#    include <lpl/procgen/Random.hpp>
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
#    include <lpl/render/Pbr.hpp>
#    include <lpl/render/Overlay.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/Reflection.hpp>
#    include <lpl/render/Revolve.hpp>
#    include <lpl/render/Scatter.hpp>
#    include <lpl/render/Topology.hpp>
#    include <lpl/render/Sky.hpp>
#    include <lpl/render/SkyDome.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>
#    include <lpl/render/Water.hpp>

namespace lpl::samples {

/**
 * @class TerrainWorld
 * @brief A world on a heightfield: generated, streamed, lit, walked and grazed.
 */
class TerrainWorld final : public engine::World {
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
        engine::TerrainSurfaceParams surfaceParams;
        surfaceParams.seaLevel = kSeaLevel;
        surfaceParams.fogDensity = kFogDensity;
        surfaceParams.ambient = kAmbient;
        surfaceParams.shadowSteps = kShadowSteps;
        _terrainSurface.configure(context.config, surfaceParams, _seed);
        _terrainSurface.attachProbe(_probeColor, _probeDepth, kProbeWidth, kProbeHeight);
        _maxResident = context.config.maxResidentChunks() == 0u ? kMaxResidentCeiling
                                                               : context.config.maxResidentChunks();

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
            core::Log::info("TerrainWorld: WASD=cam Q/E=zoom N=new seed B=shading X=exit");
        }

        generate();
        return {};
    }

    /// Authoritative: the herd walks, the vegetation regrows, the web steps.
    void onFixedStep(core::f32 dt) override
    {
        engine::World::onFixedStep(dt);
        ++_ticks;
        // A day every four minutes at 60 Hz: long enough that the light reads as
        // moving rather than flickering, short enough to see a sunset without
        // waiting for one.
        ++_windowTicks;
        // One tick of the world's clock: the sun moves, the ripples travel.
        _terrainSurface.advance(kDayStep);

        if (_infinite)
        {
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
            stepHerd();
            _living.scent().field().step(_living.recipe().stigmergy);
            if (_ticks % kWebPeriod == 0u)
            {
                _living.web().step(1u);
                reconcile();
            }
            return;
        }
        stepHerd();
        if (_ticks % kWebPeriod == 0u)
        {
            _living.web().step(1u);
            reconcile();
        }
    }

    /// Non-authoritative: input, camera, rasterize, scale, HUD, present.
    void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override
    {
        drainInput(context);
        if (!_hasSurface)
            return;

        render::RenderTarget target{_color, _depth, kRenderWidth, kRenderHeight};
        renderScene(target);
        render::blitScaled(_surface.buffer, _surface.pitch / 4u, _surface.width, _surface.height, _color,
                           kRenderWidth, kRenderHeight);
        drawHud();
        context.platform.display().present();
        ++_frames;
        ++_windowFrames;
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
    /// One tick of the day: a full cycle takes about four minutes at 60 Hz.
    static constexpr core::f32 kDayStep = 1.0f / 14400.0f;
    /// Cells across the pheromone window that follows the walker.
    static constexpr core::u32 kFieldSpan = 64u;
    /// Thermal iterations each streamed chunk gets. Also its apron, plus one.
    static constexpr core::u32 kEndlessErosion = 6u;
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
    static constexpr core::f32 kPlantFarDistance = 70.0f;
    /// Nearer than this the walker is under the canopy: wood only, no leaves.
    static constexpr core::f32 kPlantCanopyDistance = 11.0f;
    static constexpr core::f32 kChunkCentreY = 8.0f;
    static constexpr core::f32 kChunkHalfHeight = 72.0f;
    /// Relief, in metres rather than in map units: see where this is applied.
    static constexpr core::f32 kReliefScale = 2.8f;
    /// Lower frequency with the taller amplitude, or the world becomes a rasp.
    static constexpr core::f32 kReliefFrequency = 0.55f;

    /// How the surface is coloured.
    enum class Shading : core::u32 {
        Biome = 0,
        Height,
        Moisture,
        Count
    };

    /// One animal is ecology::HerdMember: a flocked body, a genome, an identity.
    using Creature = ecology::HerdMember;

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
            const procgen::WorldSnapshot snapshot =
                procgen::buildSnapshot(recipe, &registry(), &_propIds,
                                       procgen::WalkabilityRule{kSeaLevel, 2.4f});

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
        procgen::Random thin{_seed ^ 0x5EED11u};
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
        _living.openScent(_gridWidth < _gridDepth ? _gridWidth : _gridDepth, 2u);
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

        _chunkParams.size = kChunkSize;
        _chunkParams.worldSeed = _seed;
        _chunkParams.noise = _recipe.terrain;
        _chunkParams.noise.seed = _seed;
        // Real scale: one cell is one metre — the trees measure 6.8 m, the eye
        // stands at 2 m — so the relief has to be metres too. The cartridge's
        // amplitude is tuned for a 64-cell map seen from above, where 16 m of
        // peak-to-trough is a mountain range; walking through it at eye height,
        // the same terrain is a lawn. The gate's parameters are NOT touched:
        // procgen::parityChunkParams stays exactly as it was, so the P9 signature
        // does not move because the viewer changed how tall its hills are.
        _chunkParams.noise.amplitude = _recipe.terrain.amplitude * kReliefScale;
        _chunkParams.noise.frequency = _recipe.terrain.frequency * kReliefFrequency;

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

        // The pheromone window, not a world-sized grid.
        _living.scent().open(kFieldSpan, 2u);
        _living.scent().centreOn(0, 0);

        // The ecosystem comes with: a world you can walk across for an hour and never
        // meet anything is a landscape, not a world. The producer capacity is the
        // recipe's here — a streamed world has no total to count, only what is
        // resident, and tickVegetation corrects it every tick from what it finds.
        _living.configure(livingParams(), _livingRecipe, _seed);
        _living.seedWeb(0u);
        _living.openScent(kFieldSpan, 2u);
        _living.scent().centreOn(0, 0);

        // One place says what the residency policy is: the chunk parameters, the
        // streaming policy (radii, release ratio, direction weight, per-tick budget
        // — procgen::StreamingParams' business, not a copy of it here) and the
        // memory ceiling the host asked for.
        // One call says what the endless world is: the terrain parameters, the
        // streaming policy and the memory ceiling. The rule below is the CONTENT —
        // where the sea and the snow line are, how much erosion, how dense the woods.
        procgen::ChunkTerrainRule rule;
        rule.erosionIterations = kEndlessErosion;
        rule.seaLevel = kSeaLevel;
        rule.vegetationOneIn = 3u;
        _streamer.configure(_chunkParams, _riverParams, _streamParams, _maxResident, rule);

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
            patch,
            [&params](core::i32 x, core::i32 z) { return procgen::sampleWorldHeight(params, x, z).toFloat(); },
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
        procgen::Random stock{_seed ^ 0x57EA11u};
        for (core::u32 species = 0u; species + 1u < _living.web().species.size() && species < 2u; ++species)
        {
            const core::u32 wanted = _living.bodiesFor(species);
            for (core::u32 i = 0u; i < wanted; ++i)
                (void) _living.spawn(stock, species,
                                     [this](procgen::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
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
    [[nodiscard]] bool proposeSpawn(procgen::Random &random, core::u32 attempt, math::Fixed32 &outX,
                                    math::Fixed32 &outZ) const
    {
        (void) attempt;
        if (_infinite)
        {
            const core::f32 offsetX = static_cast<core::f32>(random.range(-20, 20));
            const core::f32 offsetZ = static_cast<core::f32>(random.range(-20, 20));
            outX = math::Fixed32::fromFloat(_camera.focusX() + offsetX);
            outZ = math::Fixed32::fromFloat(_camera.focusZ() + offsetZ);
            return walkable(outX, outZ);
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
     * @brief One tick of the living layer, through ecology::Herd.
     *
     * The flocking, the scent following, the speed from the genome and the
     * walkability slide are the module's. What this world supplies is the three
     * things only it knows: how a world cell maps into the pheromone window (which
     * MOVES with the walker in the endless world), where an animal may stand, and
     * what happens when a grazer eats here.
     */
    /// One tick of the living layer; the two modes differ only in what they read.
    void stepHerd()
    {
        _living.stepHerd(
            kStep,
            [this](core::i32 worldX, core::i32 worldZ, core::u32 &outX, core::u32 &outZ) {
                if (_infinite)
                    return fieldCell(worldX, worldZ, outX, outZ);
                return worldToCell(math::Fixed32::fromInt(worldX), math::Fixed32::fromInt(worldZ), outX, outZ);
            },
            [this](math::Fixed32 x, math::Fixed32 z) { return walkable(x, z); },
            [this](core::i32 worldX, core::i32 worldZ) {
                if (_infinite)
                {
                    grazeEndless(worldX, worldZ);
                    return;
                }
                core::u32 cx = 0u;
                core::u32 cz = 0u;
                if (worldToCell(math::Fixed32::fromInt(worldX), math::Fixed32::fromInt(worldZ), cx, cz))
                    graze(cx, cz);
            });
    }

    /// Eats one plant in the chunk that holds a world cell.
    void grazeEndless(core::i32 worldX, core::i32 worldZ)
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
                return; // the cell's chunk was found; no other holds it
            if (ecology::grazeAt(&chunk.plants[0], static_cast<core::u32>(chunk.plants.size()), worldX, worldZ, 1,
                                 _living.recipe().regrowthTicks))
                _living.countGrazed();
            return;
        }
    }

    /// Eats one plant near a bounded-world cell.
    void graze(core::u32 x, core::u32 z)
    {
        if (_plants.empty())
            return;
        if (ecology::grazeAt(&_plants[0], static_cast<core::u32>(_plants.size()), static_cast<core::i32>(x),
                             static_cast<core::i32>(z), 1, _living.recipe().regrowthTicks))
            _living.countGrazed();
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
        _living.reconcile(_ticks, [this](procgen::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
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
        if (_height.empty() ||
            !worldToCell(math::Fixed32::fromFloat(worldX), math::Fixed32::fromFloat(worldZ), cx, cz))
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
        params.seaLevel = kSeaLevel;
        params.ambient = kAmbient;
        params.skirtDrop = kSkirtDrop;
        params.chunkCentreY = kChunkCentreY;
        params.chunkHalfHeight = kChunkHalfHeight;

        const auto palette = [this](procgen::BiomeId biome) { return biomeColour(biome); };

        if (_infinite)
        {
            _lastTriangles = _renderer.drawStreamed(
                rt, _camera, _streamer, _terrainSurface, _props, _living.herd(), params, _frames, palette,
                [this](core::i32 x, core::i32 z) { return groundAt(x, z); });
            return;
        }

        if (_height.empty())
            return;
        _lastTriangles = _renderer.drawBounded(
            rt, _camera, _terrainSurface, _props, _living.herd(), _gridWidth, _gridDepth,
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
    [[nodiscard]] core::u32 biomeColour(procgen::BiomeId biome) const noexcept
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
                .text("  LOD rings ").number(_lodRings).text("  tris ")
                .number(_lastTriangles);
            drawShadowedText(pitchPixels, 8u, 44u, line, 0x00A0B4C8u);

            line.clear()
                .text("herd ")
                .number(countSpecies(0u))
                .text(" grazers  ")
                .number(countSpecies(1u))
                .text(" hunters  grazed ")
                .number(_living.grazedCount());
            drawShadowedText(pitchPixels, 8u, 62u, line, 0x0060FF80u);

            line.clear()
                .text("scent window ")
                .number(kFieldSpan)
                .text(" cells, recentred ")
                .number(_living.scent().recentres())
                .text(" times");
            drawShadowedText(pitchPixels, 8u, 80u, line, 0x00A0B4C8u);

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
            drawShadowedText(pitchPixels, 8u, 98u, line, 0x00A0B4C8u);

            image::drawText8x16(_surface.buffer, pitchPixels, 8u, _surface.height - 20u,
                                "WASD=walk J/L=turn I/K=tilt Q/E=zoom F=view T/Y/R/G=shading O=bounded X=exit", 0x00808890u);
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
        char key;
        while (context.platform.input().tryPopCharacter(key))
        {
            switch (key)
            {
            // In the endless world WASD WALKS, because there is somewhere to walk
            // to; in the bounded one it orbits, because there is not. Forward is
            // where the camera looks, which is the only reading of "forward" a
            // player ever means.
            case 'w':
                if (_infinite)
                    walk(1.0f);
                else
                    _camera.tilt(0.06f);
                break;
            case 's':
                if (_infinite)
                    walk(-1.0f);
                else
                    _camera.tilt(-0.06f);
                break;
            case 'a':
                if (_infinite)
                    strafe(-1.0f);
                else
                    _camera.turn(-0.08f);
                break;
            case 'd':
                if (_infinite)
                    strafe(1.0f);
                else
                    _camera.turn(0.08f);
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
                _camera.setPitch(_camera.isFirstPerson() ? 0.05f : 0.6f);
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

    // ── Small helpers ────────────────────────────────────────────────────────

    /// Walks along the camera's heading, in world cells.

    void walk(core::f32 sign) noexcept { _camera.walk(sign * kWalkStep); }

    /// Sidesteps: the heading rotated a quarter turn.
    void strafe(core::f32 sign) noexcept { _camera.strafe(sign * kWalkStep); }

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
    [[nodiscard]] bool walkable(math::Fixed32 x, math::Fixed32 z) const noexcept
    {
        if (_infinite)
        {
            const core::f32 height =
                procgen::sampleWorldHeight(_streamer.chunkParams(), x.toInt(), z.toInt()).toFloat();
            return height > kSeaLevel + 0.2f;
        }
        core::u32 cx = 0u;
        core::u32 cz = 0u;
        if (!worldToCell(x, z, cx, cz))
            return false;
        return _blocked.at(cx, cz) == 0u;
    }

    [[nodiscard]] core::u32 standingPlants() const noexcept
    {
        return _plants.empty() ? 0u
                               : ecology::countStanding(&_plants[0], static_cast<core::u32>(_plants.size()));
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
    static constexpr core::f32 kSeaLevel = -1.0f;
    /// One fixed tick at 60 Hz, in seconds.
    static inline const math::Fixed32 kStep = math::Fixed32::fromRaw(1092);

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
    lpl::pmr::vector<ecs::EntityId> _propIds;
    core::u32 _gridWidth{0u};
    core::u32 _gridDepth{0u};
    core::u32 _propCount{0u};
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
};

inline core::u32 TerrainWorld::_color[TerrainWorld::kRenderWidth * TerrainWorld::kRenderHeight];
inline core::f32 TerrainWorld::_depth[TerrainWorld::kRenderWidth * TerrainWorld::kRenderHeight];
// In BSS, like the frame itself: a 240x150 colour+depth pair is 288 KiB, and the
// kernel stack is nowhere near that. The scene learned this lesson once already.
inline core::u32 TerrainWorld::_probeColor[TerrainWorld::kProbeWidth * TerrainWorld::kProbeHeight];
inline core::f32 TerrainWorld::_probeDepth[TerrainWorld::kProbeWidth * TerrainWorld::kProbeHeight];

} // namespace lpl::samples

#endif // LPL_SAMPLES_TERRAINWORLD_HPP
