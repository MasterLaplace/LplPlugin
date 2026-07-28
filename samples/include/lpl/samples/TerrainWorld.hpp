/**
 * @file TerrainWorld.hpp
 * @brief The world viewer, in ring 0: procgen + the living layer, rasterized.
 *
 * This is `apps/mapview` with the desktop taken away. The viewer on Linux draws
 * through X11 and GLX; there is no X11 in a kernel, and there is no OpenGL — so
 * everything below the viewport is the same code and the viewport is the engine's
 * own software rasterizer writing into the framebuffer the HAL exposes.
 *
 * That is the claim the demo exists to make, and it is worth stating plainly
 * because it is easy to mistake for a screenshot: the terrain on screen was
 * generated **by the kernel**, from a seed, by the same `lpl::procgen` passes the
 * host tests fold — value noise, thermal and hydraulic erosion, depression
 * filling and drainage, the climate axes, biome classification. The herd walking
 * on it is `lpl::ai` and `lpl::ecology`, the same modules the P8 parity gate
 * folds. Nothing is baked, nothing is replayed from a recording, and there is no
 * userspace: this is a freestanding i686 kernel doing procedural generation and
 * ecology simulation between an interrupt handler and a page fault.
 *
 * @warning Everything drawn is float and none of it flows back: the authoritative
 *          state is the Fixed32 grids and creature positions, the projection and
 *          the lighting read them and produce pixels. Same discipline as
 *          @ref CubePileWorld.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_TERRAINWORLD_HPP
#    define LPL_SAMPLES_TERRAINWORLD_HPP

#    include <lpl/ai/Personality.hpp>
#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/ai/Swarm.hpp>
#    include <lpl/core/Log.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/ecology/Populations.hpp>
#    include <lpl/engine/Engine.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/image/Font8x16.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/platform/IPlatform.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/FixedMath.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>
#    include <lpl/procgen/Random.hpp>
#    include <lpl/procgen/WorldBuilder.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::samples {

/**
 * @class TerrainWorld
 * @brief engine::World generating a landscape and living on it, on screen.
 */
class TerrainWorld final : public engine::World {
public:
    [[nodiscard]] core::Expected<void> onInit(engine::WorldContext &context) override
    {
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
        stepHerd();
        if (_ticks % kWebPeriod == 0u)
        {
            _web.step(1u);
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
        blitScaled();
        drawHud();
        context.platform.display().present();
        ++_frames;
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
    static constexpr core::u32 kSize = 48u;
    static constexpr core::u32 kRenderWidth = 480u;
    static constexpr core::u32 kRenderHeight = 300u;
    static constexpr core::u32 kMaxCreatures = 48u;
    static constexpr core::u32 kWebPeriod = 60u;
    static constexpr core::u32 kRegrowthTicks = 900u;

    /// How the surface is coloured.
    enum class Shading : core::u32 { Biome = 0, Height, Moisture, Count };

    struct Camera {
        core::f32 yaw{0.7f};
        core::f32 pitch{0.65f};
        core::f32 dist{62.0f};
    };

    struct Creature {
        ai::Boid body{};
        ecology::Genome genome{};
        core::u32 id{0u};
        core::u32 species{0u};                       ///< 0 grazer, 1 hunter.
        math::Fixed32 heading{math::Fixed32::one()}; ///< Unit facing, X.
        math::Fixed32 headingZ{};                    ///< Unit facing, Z.
    };

    struct Plant {
        core::u32 cellX{0u};
        core::u32 cellZ{0u};
        core::u32 regrowth{0u};
        bool standing{true};
    };

    // ── Generation ───────────────────────────────────────────────────────────

    /**
     * @brief Builds a whole world from the current seed.
     *
     * The builder is a local: it holds the intermediate grids of every pass, and
     * on a 4 MiB heap those are worth freeing the moment the few this viewer
     * actually draws have been copied out.
     */
    void generate()
    {
        {
            procgen::NoiseParams noise;
            noise.seed = _seed;
            noise.frequency = 6.0f / static_cast<core::f32>(kSize);
            noise.amplitude = 12.0f;
            noise.octaves = 4u;

            procgen::WorldBuilder builder{_seed};
            builder.terrain(kSize, kSize, noise).normalize(kFloor, kCeiling).erode().rivers().biomes();

            _height = builder.heightfield();
            _biomes = builder.biomeMap();
            _moisture = builder.moisture();
            _rivers = procgen::riverMask(builder.drainage(), 0.02f);
        }

        (void) procgen::heightRange(_height, _low, _high);
        procgen::countBiomes(_biomes, _biomeCounts);

        // ── The living layer ─────────────────────────────────────────────────
        //
        // One obstacle mask, three readers: the pheromone field diffuses around
        // it, the herd refuses to walk into it, and the spawn refuses to start in
        // it. Three separate notions of "blocked" is how an animal ends up
        // standing in a lake that the scent flows around.
        _blocked = procgen::Grid<core::u8>{kSize, kSize, 0u};
        for (core::u32 z = 0u; z < kSize; ++z)
            for (core::u32 x = 0u; x < kSize; ++x)
            {
                const bool drowned = _height.at(x, z).toFloat() < kSeaLevel;
                const bool steep = procgen::slopeAt(_height, x, z).toFloat() > 2.4f;
                _blocked.at(x, z) = (drowned || steep) ? 1u : 0u;
            }

        _field = ai::StigmergyField{kSize, kSize, 2u};
        _field.setObstacles(_blocked);

        // Vegetation: one plant per forested cell, thinned so the map is not a
        // solid canopy. These ARE the producer level — the population is counted,
        // never integrated, so grazing a valley bare moves the number because the
        // plants are gone.
        _plants.clear();
        procgen::Random thin{_seed ^ 0x5EED11u};
        for (core::u32 z = 0u; z < kSize; ++z)
            for (core::u32 x = 0u; x < kSize; ++x)
            {
                const procgen::BiomeId biome = _biomes.at(x, z);
                const bool wooded = biome == procgen::BiomeId::Forest || biome == procgen::BiomeId::Taiga ||
                                    biome == procgen::BiomeId::Rainforest;
                if (!wooded || _blocked.at(x, z) != 0u || thin.below(2u) != 0u)
                    continue;
                Plant plant;
                plant.cellX = x;
                plant.cellZ = z;
                _plants.push_back(plant);
            }

        _web = ecology::TrophicWeb{};
        ecology::SpeciesParams producer{};
        producer.level = ecology::TrophicLevel::Producer;
        producer.capacity = math::Fixed32::fromInt(static_cast<core::i32>(_plants.size() + 1u));
        const core::u32 grass = _web.add(producer, producer.capacity, ecology::Species::kNoPrey);

        ecology::SpeciesParams herbivore{};
        herbivore.level = ecology::TrophicLevel::Primary;
        herbivore.capacity = math::Fixed32::fromInt(64);
        const core::u32 grazers = _web.add(herbivore, math::Fixed32::fromInt(48), grass);

        ecology::SpeciesParams predator{};
        predator.level = ecology::TrophicLevel::Secondary;
        predator.capacity = math::Fixed32::fromInt(16);
        (void) _web.add(predator, math::Fixed32::fromInt(8), grazers);

        _creatures.clear();
        _heredity = _seed ^ 0xA57E22u;
        _nextId = 1u;
        procgen::Random stock{_seed ^ 0xB0D533u};
        for (core::u32 i = 0u; i < 24u; ++i)
            spawn(stock, i < 18u ? 0u : 1u);

        _ticks = 0u;
        _grazed = 0u;

        core::Log::info("TerrainWorld: world generated");
    }

    void spawn(procgen::Random &random, core::u32 species)
    {
        if (_creatures.size() >= kMaxCreatures)
            return;

        Creature creature;
        creature.id = _nextId++;
        creature.species = species;

        ecology::Genome archetype{};
        if (species == 1u)
        {
            archetype.size = math::Fixed32::fromFloat(1.4f);
            archetype.maxSpeed = math::Fixed32::fromFloat(5.0f);
        }
        else
        {
            archetype.size = math::Fixed32::fromFloat(0.9f);
            archetype.maxSpeed = math::Fixed32::fromFloat(3.5f);
        }
        creature.genome = ecology::mutate(archetype, 8u, 0.18f, _heredity);

        for (core::u32 attempt = 0u; attempt < 24u; ++attempt)
        {
            const core::u32 x = random.below(kSize);
            const core::u32 z = random.below(kSize);
            if (_blocked.at(x, z) != 0u)
                continue;
            creature.body.x = cellToWorld(x);
            creature.body.z = cellToWorld(z);
            creature.body.vx = random.unit() - math::Fixed32::half();
            creature.body.vz = random.unit() - math::Fixed32::half();
            creature.heading = creature.body.vx;
            creature.headingZ = creature.body.vz;
            _creatures.push_back(creature);
            return;
        }
    }

    // ── Simulation ───────────────────────────────────────────────────────────

    void stepHerd()
    {
        if (_creatures.empty())
            return;

        for (core::u32 species = 0u; species < 2u; ++species)
        {
            _flock.clear();
            for (core::u32 i = 0u; i < _creatures.size(); ++i)
                if (_creatures[i].species == species)
                    _flock.push_back(_creatures[i].body);
            if (_flock.empty())
                continue;

            ai::BoidParams params;
            params.separationWeight = species == 1u ? 1.1f : 0.9f;
            params.alignmentWeight = species == 1u ? 0.5f : 0.8f;
            params.cohesionWeight = species == 1u ? 0.25f : 0.5f;
            params.neighbourRadius = math::Fixed32::fromInt(species == 1u ? 10 : 6);

            // dt is explicit, and the flock's own integration is discarded: the
            // boid rules decide where an animal WANTS to go, the terrain decides
            // where it may stand. Taking the position back would put bodies inside
            // rock before any check here could refuse it.
            ai::stepBoids(&_flock[0], static_cast<core::u32>(_flock.size()), params, kStep);

            core::u32 cursor = 0u;
            for (core::u32 i = 0u; i < _creatures.size(); ++i)
                if (_creatures[i].species == species)
                {
                    _creatures[i].body.vx = _flock[cursor].vx;
                    _creatures[i].body.vz = _flock[cursor].vz;
                    ++cursor;
                }
        }

        for (core::u32 i = 0u; i < _creatures.size(); ++i)
        {
            Creature &creature = _creatures[i];
            const ai::PersonalityTraits traits = ai::personalityOf(creature.id, creature.species);

            core::u32 cx = 0u;
            core::u32 cz = 0u;
            if (worldToCell(creature.body.x, creature.body.z, cx, cz))
            {
                // One field, two readings: a grazer climbs the scent toward the
                // pasture, a hunter climbs the same field because it leads to the
                // grazers. Nothing tells the hunter where the herd is — the map does.
                const core::u32 direction = _field.gradientDirection(1u, cx, cz, true);
                if (direction != ai::StigmergyField::kNoDirection)
                {
                    const math::Fixed32 pull = math::Fixed32::fromFloat(0.06f) *
                                               (math::Fixed32::half() + traits.energy);
                    creature.body.vx =
                        creature.body.vx + math::Fixed32::fromInt(procgen::kNeighbor8X[direction]) * pull;
                    creature.body.vz =
                        creature.body.vz + math::Fixed32::fromInt(procgen::kNeighbor8Z[direction]) * pull;
                }
                if (creature.species == 0u)
                {
                    _field.deposit(1u, cx, cz, math::Fixed32::fromFloat(0.6f));
                    graze(cx, cz);
                }
            }

            // The steering vector carries a heading; the speed is the genome's.
            // Multiplying by the vector's own length instead is how a settled herd
            // grinds to a halt, and how it reaches mach 20 when it does not.
            const math::Fixed32 lengthSquared =
                creature.body.vx * creature.body.vx + creature.body.vz * creature.body.vz;
            const math::Fixed32 length = procgen::fixedSqrt(lengthSquared);
            if (length.raw() > 256)
            {
                creature.heading = creature.body.vx / length;
                creature.headingZ = creature.body.vz / length;
            }

            const math::Fixed32 pace =
                creature.genome.maxSpeed * kStep * (math::Fixed32::fromFloat(0.7f) + traits.energy * math::Fixed32::half());
            const math::Fixed32 stepX = creature.heading * pace;
            const math::Fixed32 stepZ = creature.headingZ * pace;
            const math::Fixed32 tryX = creature.body.x + stepX;
            const math::Fixed32 tryZ = creature.body.z + stepZ;

            const bool freeX = walkable(tryX, creature.body.z);
            const bool freeZ = walkable(creature.body.x, tryZ);
            if (freeX && freeZ && walkable(tryX, tryZ))
            {
                creature.body.x = tryX;
                creature.body.z = tryZ;
            }
            else if (freeX)
                creature.body.x = tryX;
            else if (freeZ)
                creature.body.z = tryZ;
            else
            {
                creature.heading = -creature.heading;
                creature.headingZ = -creature.headingZ;
            }
        }

        // The field forgets: without this the scent saturates within seconds and
        // stops meaning "the herd went that way".
        _field.step(_stigmergy);
        tickVegetation();
    }

    void graze(core::u32 x, core::u32 z)
    {
        for (core::u32 i = 0u; i < _plants.size(); ++i)
        {
            if (!_plants[i].standing)
                continue;
            const core::i32 dx = static_cast<core::i32>(_plants[i].cellX) - static_cast<core::i32>(x);
            const core::i32 dz = static_cast<core::i32>(_plants[i].cellZ) - static_cast<core::i32>(z);
            if (dx > 1 || dx < -1 || dz > 1 || dz < -1)
                continue;
            _plants[i].standing = false;
            _plants[i].regrowth = kRegrowthTicks;
            ++_grazed;
            return;
        }
    }

    void tickVegetation()
    {
        for (core::u32 i = 0u; i < _plants.size(); ++i)
            if (!_plants[i].standing && _plants[i].regrowth != 0u && --_plants[i].regrowth == 0u)
                _plants[i].standing = true;

        if (!_plants.empty() && !_web.species.empty())
            _web.species[0].population = math::Fixed32::fromInt(static_cast<core::i32>(standingPlants()));
    }

    /// Brings the number of bodies in line with what the web says exists.
    void reconcile()
    {
        procgen::Random stock{_heredity ^ (_ticks * 0x9E3779B9u)};
        for (core::u32 species = 0u; species < 2u; ++species)
        {
            if (_web.species.size() <= species + 1u)
                continue;
            const core::i32 head = _web.species[species + 1u].population.toInt() / 2;
            const core::u32 wanted = head <= 0 ? 0u : static_cast<core::u32>(head);

            core::u32 have = 0u;
            for (core::u32 i = 0u; i < _creatures.size(); ++i)
                if (_creatures[i].species == species)
                    ++have;

            if (have < wanted)
                spawn(stock, species);
            else if (have > wanted && have > 1u)
                for (core::u32 i = 0u; i < _creatures.size(); ++i)
                    if (_creatures[i].species == species)
                    {
                        _creatures[i] = _creatures[_creatures.size() - 1u];
                        _creatures.pop_back();
                        break;
                    }
        }
    }

    // ── Rendering ────────────────────────────────────────────────────────────

    /**
     * @brief Fills a triangle whichever way it faces.
     *
     * @c fillTriangle culls anything whose screen-space area is negative, which is
     * exactly right for a closed solid — a cube's far faces are hidden by its near
     * ones, so drawing them is wasted work. A heightfield is not a solid: it is a
     * single sheet with no inside, and half of it faces away from any given camera.
     * Culled, the ground vanished entirely and only the plant spires and the herd
     * survived, which looks like a broken projection and is in fact a correct
     * rasterizer being asked the wrong question.
     *
     * Swapping two vertices when the area comes out negative costs one edge
     * function and draws the sheet from both sides.
     */
    static void fillSheet(const render::RenderTarget &rt, const render::detail::ScreenVertex &a,
                          const render::detail::ScreenVertex &b, const render::detail::ScreenVertex &c,
                          core::u32 colour) noexcept
    {
        if (render::detail::edge(a, b, c.x, c.y) > 0.0f)
            render::detail::fillTriangle(rt, a, b, c, colour);
        else
            render::detail::fillTriangle(rt, a, c, b, colour);
    }

    void renderScene(const render::RenderTarget &rt) const noexcept
    {
        render::clearTarget(rt, 0x000A0E18u);
        if (_height.empty())
            return;

        math::Fixed32 sy{}, cy{}, sp{}, cp{};
        math::Cordic::sincos(math::Fixed32::fromFloat(_camera.yaw), sy, cy);
        math::Cordic::sincos(math::Fixed32::fromFloat(_camera.pitch), sp, cp);

        using Vec3f = math::Vec3<core::f32>;
        const core::f32 dirx = cp.toFloat() * sy.toFloat();
        const core::f32 diry = sp.toFloat();
        const core::f32 dirz = cp.toFloat() * cy.toFloat();
        const Vec3f eye(_camera.dist * dirx, 6.0f + _camera.dist * diry, _camera.dist * dirz);
        const auto view = math::Mat4<core::f32>::lookAt(eye, Vec3f(0.0f, 2.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f));
        const core::f32 aspect = static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height);
        const auto proj = render::perspectiveFov(math::Fixed32::fromFloat(1.04719755f), aspect, 0.4f, 400.0f);
        const auto mvp = proj * view;

        const core::f32 half = static_cast<core::f32>(kSize) * 0.5f;

        // ── The ground, one quad per cell ────────────────────────────────────
        for (core::u32 z = 0u; z + 1u < kSize; ++z)
        {
            for (core::u32 x = 0u; x + 1u < kSize; ++x)
            {
                const core::f32 y00 = _height.at(x, z).toFloat();
                const core::f32 y10 = _height.at(x + 1u, z).toFloat();
                const core::f32 y11 = _height.at(x + 1u, z + 1u).toFloat();
                const core::f32 y01 = _height.at(x, z + 1u).toFloat();

                const core::f32 x0 = static_cast<core::f32>(x) - half;
                const core::f32 x1 = x0 + 1.0f;
                const core::f32 z0 = static_cast<core::f32>(z) - half;
                const core::f32 z1 = z0 + 1.0f;

                const auto a = render::detail::projectVertex(mvp, x0, y00, z0, rt.width, rt.height);
                const auto b = render::detail::projectVertex(mvp, x1, y10, z0, rt.width, rt.height);
                const auto c = render::detail::projectVertex(mvp, x1, y11, z1, rt.width, rt.height);
                const auto d = render::detail::projectVertex(mvp, x0, y01, z1, rt.width, rt.height);

                // Lambert from the cell's own normal: flat shading on a heightfield
                // is nearly invisible, and the relief is the whole subject.
                core::f32 nx = y00 - y10;
                core::f32 nz = y00 - y01;
                core::f32 lit = 0.34f + 0.66f * (2.0f / (2.0f + nx * nx + nz * nz));
                if (nx + nz > 0.0f)
                    lit += 0.10f;
                if (lit > 1.25f)
                    lit = 1.25f;

                const core::u32 colour = shade(cellColour(x, z), lit);
                fillSheet(rt, a, b, c, colour);
                fillSheet(rt, a, c, d, colour);
            }
        }

        // ── The sea, one flat quad ───────────────────────────────────────────
        {
            const auto a = render::detail::projectVertex(mvp, -half, kSeaLevel, -half, rt.width, rt.height);
            const auto b = render::detail::projectVertex(mvp, half, kSeaLevel, -half, rt.width, rt.height);
            const auto c = render::detail::projectVertex(mvp, half, kSeaLevel, half, rt.width, rt.height);
            const auto d = render::detail::projectVertex(mvp, -half, kSeaLevel, half, rt.width, rt.height);
            fillSheet(rt, a, b, c, 0x00123A6Au);
            fillSheet(rt, a, c, d, 0x00123A6Au);
        }

        // ── Standing vegetation, as small spires ─────────────────────────────
        for (core::u32 i = 0u; i < _plants.size(); ++i)
        {
            if (!_plants[i].standing)
                continue;
            const core::f32 px = static_cast<core::f32>(_plants[i].cellX) - half + 0.5f;
            const core::f32 pz = static_cast<core::f32>(_plants[i].cellZ) - half + 0.5f;
            const core::f32 ground = _height.at(_plants[i].cellX, _plants[i].cellZ).toFloat();
            const auto base0 = render::detail::projectVertex(mvp, px - 0.3f, ground, pz, rt.width, rt.height);
            const auto base1 = render::detail::projectVertex(mvp, px + 0.3f, ground, pz, rt.width, rt.height);
            const auto apex = render::detail::projectVertex(mvp, px, ground + 1.6f, pz, rt.width, rt.height);
            fillSheet(rt, base0, base1, apex, 0x00204A24u);
        }

        // ── The herd ─────────────────────────────────────────────────────────
        for (core::u32 i = 0u; i < _creatures.size(); ++i)
        {
            const Creature &creature = _creatures[i];
            core::u32 cx = 0u;
            core::u32 cz = 0u;
            if (!worldToCell(creature.body.x, creature.body.z, cx, cz))
                continue;
            const core::f32 ground = _height.at(cx, cz).toFloat();
            const core::f32 size = 0.35f * creature.genome.size.toFloat();
            const core::f32 wx = creature.body.x.toFloat();
            const core::f32 wz = creature.body.z.toFloat();

            const ai::PersonalityTraits traits = ai::personalityOf(creature.id, creature.species);
            core::u32 tint = creature.species == 1u ? 0x00C03028u : 0x00D0A852u;
            if (traits.aggression > math::Fixed32::fromFloat(0.75f))
                tint |= 0x00200000u;

            // A billboard rather than a box: at this scale a creature is a handful
            // of pixels, and six lit faces cost six times as much to say the same.
            const auto b0 = render::detail::projectVertex(mvp, wx - size, ground, wz, rt.width, rt.height);
            const auto b1 = render::detail::projectVertex(mvp, wx + size, ground, wz, rt.width, rt.height);
            const auto t1 = render::detail::projectVertex(mvp, wx + size, ground + size * 2.4f, wz, rt.width,
                                                          rt.height);
            const auto t0 = render::detail::projectVertex(mvp, wx - size, ground + size * 2.4f, wz, rt.width,
                                                          rt.height);
            fillSheet(rt, b0, b1, t1, tint);
            fillSheet(rt, b0, t1, t0, tint);
        }
    }

    [[nodiscard]] core::u32 cellColour(core::u32 x, core::u32 z) const noexcept
    {
        if (_shading == Shading::Height)
        {
            const core::f32 span = (_high - _low).toFloat();
            const core::f32 t = span > 0.0f ? (_height.at(x, z) - _low).toFloat() / span : 0.5f;
            return ramp(t);
        }
        if (_shading == Shading::Moisture)
            return _moisture.empty() ? 0x00404040u : ramp(_moisture.at(x, z).toFloat());

        if (!_rivers.empty() && _rivers.at(x, z) != 0u)
            return 0x002A6BB8u;
        return biomeColour(_biomes.at(x, z));
    }

    [[nodiscard]] static core::u32 biomeColour(procgen::BiomeId biome) noexcept
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
        return 0x00FF00FFu;
    }

    /// Amber-on-abyss ramp, so a scalar view reads as instrumentation.
    [[nodiscard]] static core::u32 ramp(core::f32 t) noexcept
    {
        if (t < 0.0f)
            t = 0.0f;
        if (t > 1.0f)
            t = 1.0f;
        const core::u32 r = static_cast<core::u32>((0.05f + 0.95f * t) * 255.0f);
        const core::u32 g = static_cast<core::u32>((0.05f + 0.62f * t * t) * 255.0f);
        const core::u32 b = static_cast<core::u32>((0.10f + 0.15f * t * t * t) * 255.0f);
        return (r << 16) | (g << 8) | b;
    }

    /**
     * @brief Applies a lighting term to a colour, SATURATING each channel.
     *
     * Masking with 0xFF instead of clamping is the difference between a lit
     * highlight and a wrap-around: the lighting term reaches 1.25 on a slope
     * facing the light, and 0xF0 (snow) times 1.25 is 0x12C, which masked becomes
     * 0x2C — dark. On screen that is a snowfield covered in black, red and yellow
     * blocks, each one a channel that overflowed at a different point, and it
     * reads as corrupt memory rather than as the arithmetic mistake it is.
     */
    [[nodiscard]] static core::u32 shade(core::u32 colour, core::f32 lit) noexcept
    {
        const core::u32 scale = static_cast<core::u32>(lit * 256.0f);
        const auto channel = [scale](core::u32 value) -> core::u32 {
            const core::u32 lit8 = (value * scale) >> 8;
            return lit8 > 255u ? 255u : lit8;
        };
        return (channel((colour >> 16) & 0xFFu) << 16) | (channel((colour >> 8) & 0xFFu) << 8) |
               channel(colour & 0xFFu);
    }

    /// Nearest-neighbour scale of the engine frame onto the display surface.
    void blitScaled() const noexcept
    {
        const core::u32 pitchPixels = _surface.pitch / 4u;
        for (core::u32 dy = 0u; dy < _surface.height; ++dy)
        {
            const core::u32 *sourceRow = &_color[((dy * kRenderHeight) / _surface.height) * kRenderWidth];
            core::u32 *destinationRow = &_surface.buffer[dy * pitchPixels];
            for (core::u32 dx = 0u; dx < _surface.width; ++dx)
                destinationRow[dx] = sourceRow[(dx * kRenderWidth) / _surface.width];
        }
    }

    void drawHud() const noexcept
    {
        const core::u32 pitchPixels = _surface.pitch / 4u;
        char line[80];

        format(line, sizeof(line), "LPLKERNEL WORLD VIEWER  seed ", _seed, "  ", kSize, "x", kSize);
        image::drawText8x16(_surface.buffer, pitchPixels, 8u, 8u, line, 0x00FFAA22u);

        format(line, sizeof(line), shadingName(), presentBiomes(), " biomes  ", standingPlants(), " plants of ",
               static_cast<core::u32>(_plants.size()));
        image::drawText8x16(_surface.buffer, pitchPixels, 8u, 26u, line, 0x00C8C8C0u);

        format(line, sizeof(line), "herd ", countSpecies(0u), " grazers  ", countSpecies(1u), " hunters  grazed ",
               _grazed);
        image::drawText8x16(_surface.buffer, pitchPixels, 8u, 44u, line, 0x0060FF80u);

        image::drawText8x16(_surface.buffer, pitchPixels, 8u, _surface.height - 20u,
                            "WASD=cam Q/E=zoom N=new seed B=shading X=exit", 0x00808890u);
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

    /// Tiny formatter: no snprintf in a freestanding build, and none needed.
    static void format(char *out, core::u32 capacity, const char *a, core::u32 v0, const char *b, core::u32 v1,
                       const char *c, core::u32 v2) noexcept
    {
        core::u32 n = 0u;
        appendText(out, capacity, n, a);
        appendNumber(out, capacity, n, v0);
        appendText(out, capacity, n, b);
        appendNumber(out, capacity, n, v1);
        appendText(out, capacity, n, c);
        appendNumber(out, capacity, n, v2);
        if (n < capacity)
            out[n] = '\0';
        else if (capacity != 0u)
            out[capacity - 1u] = '\0';
    }

    static void appendText(char *out, core::u32 capacity, core::u32 &n, const char *text) noexcept
    {
        for (const char *p = text; *p != '\0' && n + 1u < capacity; ++p)
            out[n++] = *p;
    }

    static void appendNumber(char *out, core::u32 capacity, core::u32 &n, core::u32 value) noexcept
    {
        char digits[12];
        core::u32 count = 0u;
        do
        {
            digits[count++] = static_cast<char>('0' + (value % 10u));
            value /= 10u;
        } while (value != 0u && count < sizeof(digits));
        while (count != 0u && n + 1u < capacity)
            out[n++] = digits[--count];
    }

    // ── Input ────────────────────────────────────────────────────────────────

    void drainInput(engine::WorldContext &context) noexcept
    {
        char key;
        while (context.platform.input().tryPopCharacter(key))
        {
            switch (key)
            {
            case 'a': _camera.yaw -= 0.08f; break;
            case 'd': _camera.yaw += 0.08f; break;
            case 'w': _camera.pitch = clampF(_camera.pitch + 0.06f, -1.40f, 1.40f); break;
            case 's': _camera.pitch = clampF(_camera.pitch - 0.06f, -1.40f, 1.40f); break;
            case 'q': _camera.dist = clampF(_camera.dist - 3.0f, 12.0f, 160.0f); break;
            case 'e': _camera.dist = clampF(_camera.dist + 3.0f, 12.0f, 160.0f); break;
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
            default: break;
            }
        }
    }

    // ── Small helpers ────────────────────────────────────────────────────────

    [[nodiscard]] static core::f32 clampF(core::f32 value, core::f32 low, core::f32 high) noexcept
    {
        return value < low ? low : (value > high ? high : value);
    }

    [[nodiscard]] static math::Fixed32 cellToWorld(core::u32 cell) noexcept
    {
        return math::Fixed32::fromInt(static_cast<core::i32>(cell)) -
               math::Fixed32::fromInt(static_cast<core::i32>(kSize / 2u));
    }

    [[nodiscard]] static bool worldToCell(math::Fixed32 x, math::Fixed32 z, core::u32 &outX, core::u32 &outZ) noexcept
    {
        const core::i32 gx = (x + math::Fixed32::fromInt(static_cast<core::i32>(kSize / 2u))).toInt();
        const core::i32 gz = (z + math::Fixed32::fromInt(static_cast<core::i32>(kSize / 2u))).toInt();
        if (gx < 0 || gz < 0 || static_cast<core::u32>(gx) >= kSize || static_cast<core::u32>(gz) >= kSize)
            return false;
        outX = static_cast<core::u32>(gx);
        outZ = static_cast<core::u32>(gz);
        return true;
    }

    [[nodiscard]] bool walkable(math::Fixed32 x, math::Fixed32 z) const noexcept
    {
        core::u32 cx = 0u;
        core::u32 cz = 0u;
        if (!worldToCell(x, z, cx, cz))
            return false;
        return _blocked.at(cx, cz) == 0u;
    }

    [[nodiscard]] core::u32 standingPlants() const noexcept
    {
        core::u32 standing = 0u;
        for (core::u32 i = 0u; i < _plants.size(); ++i)
            if (_plants[i].standing)
                ++standing;
        return standing;
    }

    [[nodiscard]] core::u32 countSpecies(core::u32 species) const noexcept
    {
        core::u32 count = 0u;
        for (core::u32 i = 0u; i < _creatures.size(); ++i)
            if (_creatures[i].species == species)
                ++count;
        return count;
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
    math::Fixed32 _low{};
    math::Fixed32 _high{};
    core::u32 _biomeCounts[static_cast<core::u32>(procgen::BiomeId::Count)] = {};

    ai::StigmergyField _field;
    ai::StigmergyParams _stigmergy{};
    ecology::TrophicWeb _web;
    lpl::pmr::vector<Creature> _creatures;
    lpl::pmr::vector<Plant> _plants;
    lpl::pmr::vector<ai::Boid> _flock;

    Camera _camera{};
    Shading _shading{Shading::Biome};
    core::u32 _seed{1337u};
    core::u32 _heredity{1u};
    core::u32 _nextId{1u};
    core::u32 _ticks{0u};
    core::u32 _frames{0u};
    core::u32 _grazed{0u};
};

inline core::u32 TerrainWorld::_color[TerrainWorld::kRenderWidth * TerrainWorld::kRenderHeight];
inline core::f32 TerrainWorld::_depth[TerrainWorld::kRenderWidth * TerrainWorld::kRenderHeight];

} // namespace lpl::samples

#endif // LPL_SAMPLES_TERRAINWORLD_HPP
