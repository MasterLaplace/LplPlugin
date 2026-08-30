/**
 * @file ChronicleWorld.hpp
 * @brief A region, a corpus, and the people a corpus says crossed it.
 *
 * @warning **The first World any of this session's work has.** `history::JourneySystem`,
 * `engine::systems::TerrainRoutes` and the routing cascade were reachable only from a parity fold
 * -- which proves they agree between two targets and nothing at all about whether they are worth
 * running. A world you can look at is the other half.
 *
 * What it is: the southern Peloponnese as it actually is -- the same SRTM survey
 * @ref makeRealReliefField hands the kernel client -- with a handful of real places at their real
 * coordinates, roads a corpus attests laid across it by least-cost search, and travellers walking
 * those roads across a century. The chronicle records where they arrive, and nobody told them to.
 *
 * @warning **A chart, not a walk.** @ref TerrainWorld already answers "what does it look like from
 * inside"; this answers "what happened here, over years", and those want different views. Drawing
 * this in first person would also have meant a second `TerrainRenderer` -- two answers to what the
 * world looks like, and the one five booted artifacts check would not be this one.
 *
 * @warning **The attested links are a FIXTURE and are marked as such.** Pleiades carries
 * `connectsWith` for exactly this, and `samples/` cannot reach it: LplKnowledge depends on
 * LplPlugin and never the other way round. So the network below stands in for a corpus rather
 * than being one, and the moment a `.lplknow` gazetteer is wired in, these tables go.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_CHRONICLEWORLD_HPP
#    define LPL_SAMPLES_CHRONICLEWORLD_HPP

#    include <lpl/engine/World.hpp>
#    include <lpl/engine/systems/Journey.hpp>
#    include <lpl/engine/systems/TerrainRoutes.hpp>
#    include <lpl/history/Chronicle.hpp>
#    include <lpl/history/Era.hpp>
#    include <lpl/history/HistorySystem.hpp>
#    include <lpl/history/PossibleWorld.hpp>
#    include <lpl/history/Timeline.hpp>
#    include <lpl/procgen/MapShading.hpp>
#    include <lpl/procgen/WorldSnapshot.hpp>
#    include <lpl/render/Overlay.hpp>
#    include <lpl/samples/TerrainWorld.hpp>
#    include <lpl/std/memory.hpp>

namespace lpl::samples {

/**
 * @struct ChroniclePlace
 * @brief A place, where it really is.
 */
struct ChroniclePlace {
    core::u32 id;        ///< Identifier the corpus uses.
    const char *name;    ///< What to write on the map.
    core::f32 latitude;  ///< Degrees north.
    core::f32 longitude; ///< Degrees east.
};

/// Identifiers. Spaced away from the parity fixture's so a log naming one is unambiguous.
enum : core::u32 {
    kChroniclePlaceKalamata = 700u,
    kChroniclePlaceKardamyli = 701u,
    kChroniclePlaceOitylo = 702u,
    kChroniclePlaceAreopoli = 703u,
    kChroniclePlaceGytheio = 704u,
    kChroniclePlaceTainaron = 705u,
    kChroniclePlaceSparta = 706u,

    kChronicleSourceRegister = 720u,
    kChronicleSourcePeriplus = 721u,

    kChronicleSubjectMerchant = 740u,
    kChronicleSubjectPilgrim = 741u,
};

/**
 * @brief The places, at their real coordinates.
 *
 * @warning **Two of these are NOT on the map, and that was measured rather than intended.** The
 * survey is one degree tall, from 37 N southward; Sparta at 37.074 and Kalamata at 37.038 both sit
 * just north of its edge, so the gazetteer knows exactly where they are and the routing grid
 * cannot place either. The first draft said "Sparta is off this survey" and seeded a traveller at
 * Kalamata, who therefore never entered the world -- one body where the log said two seeded, which
 * is what an off-grid birth looks like from outside.
 *
 * They stay, both of them: a gazetteer holding only places inside its own map never exercises the
 * "located, and not on this ground" branch, and it is the honest shape of the region anyway --
 * the two cities everyone walked to are the ones over the northern ridge.
 */
inline constexpr ChroniclePlace kManiPlaces[] = {
    {kChroniclePlaceKalamata,  "Kalamata",  37.038f, 22.113f},
    {kChroniclePlaceKardamyli, "Kardamyli", 36.891f, 22.234f},
    {kChroniclePlaceOitylo,    "Oitylo",    36.706f, 22.400f},
    {kChroniclePlaceAreopoli,  "Areopoli",  36.665f, 22.383f},
    {kChroniclePlaceGytheio,   "Gytheio",   36.759f, 22.565f},
    {kChroniclePlaceTainaron,  "Tainaron",  36.386f, 22.483f},
    {kChroniclePlaceSparta,    "Sparta",    37.074f, 22.429f},
};

inline constexpr core::u32 kManiPlaceCount = sizeof(kManiPlaces) / sizeof(kManiPlaces[0]);

/// The century this world runs, and the window every place is attested across.
inline constexpr core::i32 kChronicleFirstYear = -500;
inline constexpr core::i32 kChronicleLastYear = -400;

/**
 * @class ManiGazetteer
 * @brief Where the places are, and which pairs a source says were connected.
 *
 * @warning Positions are derived from LATITUDE AND LONGITUDE through the survey's own projection,
 * never written as cells. A cell index is a fact about a lattice chosen for a memory budget; a
 * coordinate is a fact about the world, and only one of the two stays true when the survey is
 * re-tiled.
 */
class ManiGazetteer final : public history::IPlaceResolver {
public:
    /**
     * @brief Resolves every place through @p field's projection.
     *
     * @param field The survey the world stands on.
     * @param side  Cells on a side of the world grid.
     */
    void bind(const math::ReliefField &field, core::u32 side) noexcept
    {
        _side = side;
        for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
        {
            const ChroniclePlace &place = kManiPlaces[i];
            const core::i32 cellX = field.projection.cellX(math::Fixed32::fromFloat(place.longitude).raw());
            const core::i32 cellZ = field.projection.cellZ(math::Fixed32::fromFloat(place.latitude).raw());
            _cellX[i] = cellX;
            _cellZ[i] = cellZ;
            // @warning THE one conversion, done once. TerrainRoutes maps a world unit to a cell of a
            // grid CENTRED ON THE ORIGIN, while the survey lives at its true projection cells
            // starting at zero. Written at both ends it would be an off-by-half-a-map that reads
            // as a gazetteer pointing at the wrong valleys.
            _worldX[i] = static_cast<core::f32>(cellX) - static_cast<core::f32>(side / 2u);
            _worldZ[i] = static_cast<core::f32>(cellZ) - static_cast<core::f32>(side / 2u);
        }
    }

    [[nodiscard]] bool resolve(core::u32 id, history::Place &out) const override
    {
        for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
        {
            if (kManiPlaces[i].id != id)
                continue;
            out.id = id;
            out.x = math::Fixed32::fromFloat(_worldX[i]);
            out.z = math::Fixed32::fromFloat(_worldZ[i]);
            // Attested across the whole era this world runs: the fixture makes no claim about
            // when a town began, and inventing one would be a date the corpus does not carry.
            out.minYear = kChronicleFirstYear;
            out.maxYear = kChronicleLastYear;
            // Every one of these has been found on the ground. Whether it falls inside THIS survey
            // is a different question, and the routing grid is the one that answers it.
            out.located = true;
            return true;
        }
        return false;
    }

    [[nodiscard]] core::u32 linkedPlaces(core::u32 id, core::u32 *out, core::u32 capacity) const override
    {
        // The coast road down the Mani, and the pass east to the gulf. A fixture standing in for
        // Pleiades' `connectsWith`; see the file comment.
        static constexpr core::u32 kLinks[][2] = {
            {kChroniclePlaceKalamata,  kChroniclePlaceKardamyli},
            {kChroniclePlaceKardamyli, kChroniclePlaceOitylo   },
            {kChroniclePlaceOitylo,    kChroniclePlaceAreopoli },
            {kChroniclePlaceAreopoli,  kChroniclePlaceTainaron },
            {kChroniclePlaceOitylo,    kChroniclePlaceGytheio  },
            {kChroniclePlaceGytheio,   kChroniclePlaceSparta   },
        };
        core::u32 written = 0u;
        for (const auto &link : kLinks)
        {
            if (written >= capacity)
                break;
            // Both directions from one table: a road is walked either way, and a consumer forced
            // to check two orders ends up checking one.
            if (link[0] == id)
                out[written++] = link[1];
            else if (link[1] == id)
                out[written++] = link[0];
        }
        return written;
    }

    /**
     * @brief Map cell a place occupies, for drawing it.
     *
     * @param index Index into @ref kManiPlaces.
     * @param outX  Receives the column.
     * @param outZ  Receives the row.
     * @return false when the place falls outside the survey.
     */
    [[nodiscard]] bool mapCell(core::u32 index, core::i32 &outX, core::i32 &outZ) const noexcept
    {
        if (index >= kManiPlaceCount)
            return false;
        outX = _cellX[index];
        outZ = _cellZ[index];
        return outX >= 0 && outZ >= 0 && outX < static_cast<core::i32>(_side) && outZ < static_cast<core::i32>(_side);
    }

private:
    core::i32 _cellX[kManiPlaceCount]{};
    core::i32 _cellZ[kManiPlaceCount]{};
    core::f32 _worldX[kManiPlaceCount]{};
    core::f32 _worldZ[kManiPlaceCount]{};
    core::u32 _side{0u};
};

/**
 * @class ChronicleWorld
 * @brief A century of the Mani, drawn as a chart.
 */
class ChronicleWorld final : public engine::World, public engine::ITerrainQuery {
public:
    /**
     * @brief Builds the world from @p recipe, whose size decides the survey window.
     *
     * @param recipe What the region is made of.
     */
    explicit ChronicleWorld(const procgen::WorldRecipe &recipe) noexcept : _recipe(recipe) {}

    ChronicleWorld() = default;

    [[nodiscard]] const char *name() const noexcept override { return "ChronicleWorld"; }

    [[nodiscard]] core::Expected<void> onInit(engine::WorldContext &context) override
    {
        // One grid cell per survey cell: the projection then places a town in the valley it is
        // actually in, rather than in whichever cell a resampling put it.
        _recipe.width = kSide;
        _recipe.depth = kSide;

        const math::ReliefField field = makeRealReliefField(_recipe, _recipe.terrain.amplitude);
        _relief = field;
        _mosaic = math::ReliefMosaic{};
        (void) _mosaic.add(&_relief);

        procgen::ReliefBlend blend{};
        blend.mosaic = &_mosaic;
        blend.detail = _recipe.terrain;
        blend.detail.amplitude = _recipe.terrain.amplitude * 0.25f;

        procgen::WalkabilityRule rule{};
        rule.seaLevel = _recipe.biomes.seaLevel;
        _snapshot = procgen::buildSnapshot(_recipe, nullptr, nullptr, rule, &blend);

        _gazetteer.bind(_relief, kSide);

        // ── The roads ────────────────────────────────────────────────────────
        engine::systems::TerrainRouteParams routeParams;
        routeParams.cellSize = 1u;
        routeParams.cost.waterLevel = _recipe.biomes.seaLevel;
        routeParams.cost.waterPenalty = 40.0f;
        routeParams.cost.slopePenalty = 6.0f;
        // @warning The cascade, on a grid small enough not to need it -- and that is the point of
        // running it here. A road planned on a summary and refined in its corridor must be the
        // road a flat search finds, and the cheapest place to keep watching that is a world
        // somebody looks at every day.
        routeParams.coarseRatio = 8u;
        _routes.bind(_snapshot.height, _gazetteer, routeParams);

        core::u32 order[kManiPlaceCount];
        for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
            order[i] = kManiPlaces[i].id;
        _pavedCells = _routes.paveAttested(order, kManiPlaceCount);

        // ── The corpus ───────────────────────────────────────────────────────
        buildCorpus();
        history::FusionReport report;
        history::WorldView view;
        _timeline = history::buildTimeline(_corpus, view, report);
        _era = history::Era::ofYears(kFirstYear, kLastYear, kDaysPerTick);

        // ── The systems ──────────────────────────────────────────────────────
        // Registered rather than stepped by hand: a step written into a World cannot be ordered
        // against what it touches, cannot be handed a fake in a test, and cannot declare what it
        // reads -- which is exactly why CubePileStepSystem was deleted.
        auto constraints = pmr::make_unique<history::HistorySystem>(_timeline, _era, _chronicle);
        _constraints = constraints.get();
        ecs::Archetype archetype;
        archetype.add(ecs::ComponentId::Position);
        archetype.add(ecs::ComponentId::Historical);
        _constraints->bindWorld(registry(), _gazetteer, archetype);
        if (auto registered = scheduler().registerSystem(std::move(constraints)); !registered)
            return registered;

        for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
            _placeIds[i] = kManiPlaces[i].id;

        engine::systems::JourneyParams walk;
        walk.pacePerYear = math::Fixed32::fromFloat(6.0f);
        walk.arrivalRadius = math::Fixed32::fromFloat(1.2f);
        walk.horizon = math::Fixed32::fromFloat(200.0f);
        auto journey = pmr::make_unique<engine::systems::JourneySystem>(registry(), _gazetteer, _placeIds,
                                                                        kManiPlaceCount, _era, _chronicle, walk);
        _journey = journey.get();
        _journey->useRoutes(_routes);
        _journey->useTerrain(*this);
        _journey->placeBodyAt(kChronicleSubjectMerchant, kChroniclePlaceKardamyli);
        _journey->placeBodyAt(kChronicleSubjectPilgrim, kChroniclePlaceGytheio);
        if (auto registered = scheduler().registerSystem(std::move(journey)); !registered)
            return registered;

        _hasSurface = context.platform.display().querySurface(_surface) && _surface.buffer != nullptr;
        if (_hasSurface)
        {
            _zoom = _surface.width / kSide;
            const core::u32 verticalZoom = _surface.height / kSide;
            _zoom = _zoom < verticalZoom ? _zoom : verticalZoom;
            if (_zoom == 0u)
                _zoom = 1u;
        }

        // What this region is made of, once, at init. A map that comes out one colour is either a
        // classifier that ran on nothing or a region that really is all one thing, and only a
        // count tells those apart.
        char census[96];
        formatLine(census, sizeof(census), "ChronicleWorld: biome grid ",
                   static_cast<core::i32>(_snapshot.biomes.width()), "x",
                   static_cast<core::i32>(_snapshot.biomes.depth()), ", distinct biomes ",
                   static_cast<core::i32>(distinctBiomes()));
        core::Log::info(census);

        char banner[96];
        formatLine(banner, sizeof(banner), "ChronicleWorld: road cells ", static_cast<core::i32>(_pavedCells),
                   ", pairs ", static_cast<core::i32>(_routes.pairs()), ", coarse expansions ",
                   static_cast<core::i32>(_routes.coarseExpanded()));
        core::Log::info(banner);
        return {};
    }

    void onFixedStep(core::f32 dt) override
    {
        tick(dt);
        ++_ticks;
    }

    void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override
    {
        if (!_hasSurface)
            return;
        drawMap();
        drawRoads();
        drawPlaces();
        drawTravellers();
        drawHud();
        context.platform.display().present();
    }

    /**
     * @brief What the run produced, for a caller that wants to assert on it.
     *
     * @warning Exposed because a World nobody can question is a World that only a screenshot
     * checks, and a screenshot cannot tell a road network from a picture of one.
     * @{
     */
    [[nodiscard]] core::u32 pavedCells() const noexcept { return _pavedCells; }
    [[nodiscard]] const engine::systems::TerrainRoutes &routes() const noexcept { return _routes; }
    [[nodiscard]] const history::Chronicle &chronicle() const noexcept { return _chronicle; }
    [[nodiscard]] core::u32 arrivals() const noexcept { return _journey != nullptr ? _journey->arrivals() : 0u; }
    [[nodiscard]] core::u32 seeded() const noexcept { return _constraints != nullptr ? _constraints->seeded() : 0u; }
    /** @} */

    // ── ITerrainQuery ────────────────────────────────────────────────────────

    [[nodiscard]] bool standable(math::Fixed32 x, math::Fixed32 z) const override
    {
        core::i32 cellX = 0;
        core::i32 cellZ = 0;
        if (!toCell(x, z, cellX, cellZ))
            return false;
        return _snapshot.blocked.at(static_cast<core::u32>(cellX), static_cast<core::u32>(cellZ)) == 0u;
    }

    [[nodiscard]] bool consumePlantAt(core::i32 /*x*/, core::i32 /*z*/) override { return false; }

private:
    static constexpr core::u32 kSide = 64u; ///< Matches the survey exactly; see onInit.
    static constexpr core::i32 kFirstYear = kChronicleFirstYear;
    static constexpr core::i32 kLastYear = kChronicleLastYear;
    static constexpr core::u32 kDaysPerTick = 30u;

    /** @brief World units to a map cell; the inverse of the gazetteer's one conversion. */
    [[nodiscard]] static bool toCell(math::Fixed32 x, math::Fixed32 z, core::i32 &outX, core::i32 &outZ) noexcept
    {
        outX = (x.raw() >> 16) + static_cast<core::i32>(kSide / 2u);
        outZ = (z.raw() >> 16) + static_cast<core::i32>(kSide / 2u);
        return outX >= 0 && outZ >= 0 && outX < static_cast<core::i32>(kSide) && outZ < static_cast<core::i32>(kSide);
    }

    /**
     * @brief The corpus: who was born where, and where a source says they went.
     *
     * @warning Two travellers rather than one, because a single body cannot show the thing this
     * world is for: two people, two births, two roads, one region -- and their paths cross at
     * Oitylo, where the coast road meets the pass, because that is what a junction IS.
     */
    void buildCorpus()
    {
        history::SourceProfile register_;
        register_.id = kChronicleSourceRegister;
        register_.kind = history::SourceKind::Notarial;
        register_.yearsAfterEvent = 0u;
        _corpus.sources.push_back(register_);

        history::SourceProfile periplus;
        periplus.id = kChronicleSourcePeriplus;
        periplus.kind = history::SourceKind::Notarial;
        periplus.yearsAfterEvent = 0u;
        _corpus.sources.push_back(periplus);

        // Kardamyli and not Kalamata: a birth off the survey is a body that never appears, and the
        // point of a traveller is that he travels.
        addBirth(kChronicleSubjectMerchant, kChroniclePlaceKardamyli, kChronicleSourceRegister);
        addBirth(kChronicleSubjectPilgrim, kChroniclePlaceGytheio, kChronicleSourcePeriplus);

        // Scored, never forced: forcing would put a traveller at his destination and then
        // congratulate him for being there.
        addJourney(kChronicleSubjectMerchant, kChroniclePlaceTainaron, -480, -460);
        addJourney(kChronicleSubjectPilgrim, kChroniclePlaceKalamata, -470, -440);
    }

    void addBirth(core::u32 subject, core::u32 place, core::u32 source)
    {
        history::Fact born;
        born.subject = subject;
        born.predicate = static_cast<core::u32>(history::Predicate::BornAt);
        born.object = place;
        born.fromDay = history::firstDayOfYear(kFirstYear);
        born.toDay = history::lastDayOfYear(kFirstYear);
        born.source = source;
        born.sigma = math::Fixed32::fromFloat(0.97f);
        _corpus.facts.push_back(born);
    }

    void addJourney(core::u32 subject, core::u32 place, core::i32 fromYear, core::i32 toYear)
    {
        history::Fact went;
        went.subject = subject;
        went.predicate = static_cast<core::u32>(history::Predicate::TravelledTo);
        went.object = place;
        went.fromDay = history::firstDayOfYear(fromYear);
        went.toDay = history::lastDayOfYear(toYear);
        went.source = kChronicleSourceRegister;
        went.sigma = math::Fixed32::fromFloat(0.3f);
        _corpus.facts.push_back(went);
    }

    // ── Drawing ──────────────────────────────────────────────────────────────

    /**
     * @brief Places the gazetteer knows and this grid cannot hold.
     *
     * @warning Counted rather than named, because it is a property of the WINDOW: re-tile the
     * survey a degree further north and Kalamata is on the map. A hard-coded name would go stale
     * the first time the window moved, silently.
     *
     * @return How many fall outside.
     */
    [[nodiscard]] core::u32 offSurveyPlaces() const noexcept
    {
        core::u32 off = 0u;
        for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
        {
            core::i32 x = 0;
            core::i32 z = 0;
            if (!_gazetteer.mapCell(i, x, z))
                ++off;
        }
        return off;
    }

    /** @brief How many biomes the classifier actually produced. */
    [[nodiscard]] core::u32 distinctBiomes() const noexcept
    {
        core::u32 seen = 0u;
        for (core::u32 b = 0u; b < static_cast<core::u32>(procgen::BiomeId::Count); ++b)
            if (_snapshot.biomeCounts[b] != 0u)
                ++seen;
        return seen;
    }

    /**
     * @brief Packs a viewer colour into the surface's word.
     *
     * @warning **`procgen::Rgb` is three floats from 0 to 1**, not three bytes, and casting a
     * channel straight to an integer sends every colour below 1.0 to zero. The first version of
     * this world did exactly that: the roads and towns came out right because their literals
     * happened to be written as 214 and 152, and the entire biome map came out black. Twelve
     * distinct biomes, all of them black -- which looks like a classifier that never ran.
     *
     * @param colour Channels in 0..1.
     * @return The packed 0x00RRGGBB word.
     */
    [[nodiscard]] static core::u32 pack(procgen::Rgb colour) noexcept
    {
        const auto channel = [](float value) {
            const float scaled = value * 255.0f;
            const float clamped = scaled < 0.0f ? 0.0f : scaled > 255.0f ? 255.0f : scaled;
            return static_cast<core::u32>(clamped);
        };
        return (channel(colour.r) << 16) | (channel(colour.g) << 8) | channel(colour.b);
    }

    void put(core::i32 cellX, core::i32 cellZ, procgen::Rgb colour) noexcept
    {
        if (cellX < 0 || cellZ < 0 || cellX >= static_cast<core::i32>(kSide) || cellZ >= static_cast<core::i32>(kSide))
            return;
        const core::u32 packed = pack(colour);
        const core::u32 pitch = _surface.pitch / 4u;
        for (core::u32 dz = 0u; dz < _zoom; ++dz)
        {
            const core::u32 y = static_cast<core::u32>(cellZ) * _zoom + dz;
            if (y >= _surface.height)
                return;
            for (core::u32 dx = 0u; dx < _zoom; ++dx)
            {
                const core::u32 x = static_cast<core::u32>(cellX) * _zoom + dx;
                if (x < _surface.width)
                    _surface.buffer[y * pitch + x] = packed;
            }
        }
    }

    void drawMap() noexcept
    {
        // Cleared first: the map is smaller than the window whenever the zoom does not divide it,
        // and a stale border is indistinguishable from map that is there.
        const core::u32 pitch = _surface.pitch / 4u;
        for (core::u32 y = 0u; y < _surface.height; ++y)
            for (core::u32 x = 0u; x < _surface.width; ++x)
                _surface.buffer[y * pitch + x] = 0x00060810u;

        for (core::u32 z = 0u; z < kSide; ++z)
        {
            for (core::u32 x = 0u; x < kSide; ++x)
            {
                procgen::Rgb colour = procgen::biomeColour(_snapshot.biomes.at(x, z));
                // Shaded by slope, so relief reads on a flat chart. Without it a mountain and a
                // plain of the same biome are the same rectangle, which is a map of the biome
                // classifier rather than of the ground.
                const core::f32 slope = procgen::slopeAt(_snapshot.height, x, z).toFloat();
                const core::f32 lit = slope > 2.0f ? 0.62f : slope > 0.8f ? 0.82f : 1.0f;
                colour.r *= lit;
                colour.g *= lit;
                colour.b *= lit;
                put(static_cast<core::i32>(x), static_cast<core::i32>(z), colour);
            }
        }
    }

    void drawRoads() noexcept
    {
        const procgen::Rgb road{0.84f, 0.60f, 0.23f};
        for (core::u32 z = 0u; z < kSide; ++z)
            for (core::u32 x = 0u; x < kSide; ++x)
                if (_routes.roads().at(x, z) != 0u)
                    put(static_cast<core::i32>(x), static_cast<core::i32>(z), road);
    }

    void drawPlaces() noexcept
    {
        const procgen::Rgb town{0.97f, 0.93f, 0.82f};
        for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
        {
            core::i32 cellX = 0;
            core::i32 cellZ = 0;
            if (!_gazetteer.mapCell(i, cellX, cellZ))
                continue; // off this survey: Sparta. Drawn nowhere, named in the HUD.
            for (core::i32 dz = -1; dz <= 1; ++dz)
                for (core::i32 dx = -1; dx <= 1; ++dx)
                    put(cellX + dx, cellZ + dz, town);
        }
    }

    void drawTravellers() noexcept
    {
        const procgen::Rgb walker{0.93f, 0.33f, 0.24f};
        _bodiesDrawn = 0u;
        for (const auto &partition : registry().partitions())
        {
            if (partition == nullptr || !partition->archetype().has(ecs::ComponentId::Historical))
                continue;
            for (const auto &chunkPtr : partition->chunks())
            {
                if (chunkPtr == nullptr)
                    continue;
                // The WRITE side, which is where the systems that moved these bodies wrote them.
                const auto *positions = static_cast<const math::Vec3<math::Fixed32> *>(
                    chunkPtr->writeComponent(ecs::ComponentId::Position));
                if (positions == nullptr)
                    continue;
                for (core::u32 row = 0u; row < chunkPtr->count(); ++row)
                {
                    core::i32 cellX = 0;
                    core::i32 cellZ = 0;
                    if (!toCell(positions[row].x, positions[row].z, cellX, cellZ))
                        continue;
                    for (core::i32 d = -1; d <= 1; ++d)
                    {
                        put(cellX + d, cellZ, walker);
                        put(cellX, cellZ + d, walker);
                    }
                    ++_bodiesDrawn;
                }
            }
        }
    }

    void drawHud() noexcept
    {
        const core::u32 pitch = _surface.pitch / 4u;
        char line[96];
        core::u32 y = 4u;
        const auto write = [&](const char *text) {
            render::drawShadowedText8x16(_surface.buffer, pitch, 6u, y, text, 0x00F4EFE4u, 0x00101418u);
            y += 17u;
        };

        write("LPLKERNEL CHRONICLE  the Mani, -500 to -400");

        const core::i32 year = kFirstYear + static_cast<core::i32>((_ticks * kDaysPerTick) / 365u);
        formatLine(line, sizeof(line), "year ", year, "   bodies ", static_cast<core::i32>(_bodiesDrawn),
                   "   arrivals ", static_cast<core::i32>(_journey != nullptr ? _journey->arrivals() : 0u));
        write(line);

        formatLine(line, sizeof(line), "road cells ", static_cast<core::i32>(_pavedCells), "   pairs ",
                   static_cast<core::i32>(_routes.pairs()), "   unreachable ",
                   static_cast<core::i32>(_routes.unreachable()));
        write(line);

        // @warning The cascade's own footprint, on screen. Zero coarse expansions would mean the
        // route fell back to a flat search -- which lays an identical network and would look
        // exactly like this.
        formatLine(line, sizeof(line), "coarse ", static_cast<core::i32>(_routes.coarseExpanded()), "   corridor ",
                   static_cast<core::i32>(_routes.corridorCells()), "   fine ",
                   static_cast<core::i32>(_routes.expanded()));
        write(line);

        formatLine(line, sizeof(line), "chronicle ", static_cast<core::i32>(_chronicle.size()), "   seeded ",
                   static_cast<core::i32>(_constraints != nullptr ? _constraints->seeded() : 0u), "   walkers ",
                   static_cast<core::i32>(_journey != nullptr ? _journey->walkers() : 0u));
        write(line);

        // Three values because that is what the formatter takes, and because the third is the one
        // worth reading: a place the gazetteer holds and this grid cannot. A first version tried to
        // end the line on prose and trimmed the unused value back out by searching the buffer,
        // which found the wrong character and printed "this ghoandurvey".
        const core::u32 off = offSurveyPlaces();
        formatLine(line, sizeof(line), "places ", static_cast<core::i32>(kManiPlaceCount), "  on this ground ",
                   static_cast<core::i32>(kManiPlaceCount - off), "  located elsewhere ", static_cast<core::i32>(off));
        write(line);
    }

    /** @brief Three labelled integers on one line, without a formatting library. */
    static void formatLine(char *out, core::u32 capacity, const char *a, core::i32 av, const char *b, core::i32 bv,
                           const char *c, core::i32 cv) noexcept
    {
        core::u32 at = 0u;
        const auto append = [&](const char *text) {
            while (*text != '\0' && at + 1u < capacity)
                out[at++] = *text++;
        };
        const auto appendInt = [&](core::i32 value) {
            char digits[12];
            core::u32 count = 0u;
            const bool negative = value < 0;
            core::u32 magnitude = negative ? static_cast<core::u32>(-value) : static_cast<core::u32>(value);
            do
            {
                digits[count++] = static_cast<char>('0' + (magnitude % 10u));
                magnitude /= 10u;
            } while (magnitude != 0u && count < sizeof(digits));
            if (negative && at + 1u < capacity)
                out[at++] = '-';
            while (count != 0u && at + 1u < capacity)
                out[at++] = digits[--count];
        };
        append(a);
        appendInt(av);
        append(b);
        appendInt(bv);
        append(c);
        appendInt(cv);
        out[at] = '\0';
    }

    procgen::WorldRecipe _recipe{};
    procgen::WorldSnapshot _snapshot{};
    math::ReliefField _relief{};
    math::ReliefMosaic _mosaic{};
    ManiGazetteer _gazetteer{};
    engine::systems::TerrainRoutes _routes{};

    history::Corpus _corpus{};
    history::Timeline _timeline{};
    history::Era _era{history::Era::ofYears(kFirstYear, kLastYear, kDaysPerTick)};
    history::Chronicle _chronicle{};
    history::HistorySystem *_constraints{nullptr};
    engine::systems::JourneySystem *_journey{nullptr};
    core::u32 _placeIds[kManiPlaceCount]{};

    platform::SurfaceDescriptor _surface{};
    bool _hasSurface{false};
    core::u32 _zoom{1u};
    core::u32 _ticks{0u};
    core::u32 _pavedCells{0u};
    core::u32 _bodiesDrawn{0u};
};

} // namespace lpl::samples

#endif // LPL_SAMPLES_CHRONICLEWORLD_HPP
