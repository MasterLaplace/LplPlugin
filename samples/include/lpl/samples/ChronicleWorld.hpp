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
    core::u32 id;        /**< Identifier the corpus uses. */
    const char *name;    /**< What to write on the map. */
    core::f32 latitude;  /**< Degrees north. */
    core::f32 longitude; /**< Degrees east. */
};

/** Identifiers. Spaced away from the parity fixture's so a log naming one is unambiguous. */
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

/**
 * @name The century this world runs, and the window every place is attested across
 * @{
 */
inline constexpr core::i32 kChronicleFirstYear = -500;
inline constexpr core::i32 kChronicleLastYear = -400;
/** @} */

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
     *
     * @warning THE one conversion, done once. TerrainRoutes maps a world unit to a cell of a
     * grid CENTRED ON THE ORIGIN, while the survey lives at its true projection cells
     * starting at zero. Written at both ends it would be an off-by-half-a-map that reads
     * as a gazetteer pointing at the wrong valleys.
     */
    void bind(const math::ReliefField &field, core::u32 side) noexcept;

    /**
     * @brief Looks a place of the fixture up.
     *
     * Every place is attested across the whole era this world runs: the fixture makes no claim
     * about when a town began, and inventing one would be a date the corpus does not carry.
     *
     * Every one of these has been found on the ground, so each is located. Whether it falls inside
     * THIS survey is a different question, and the routing grid is the one that answers it.
     */
    [[nodiscard]] inline bool resolve(core::u32 id, history::Place &out) const override;

    /**
     * @brief The places a link of the fixture connects to @p id.
     *
     * The links are the coast road down the Mani, and the pass east to the gulf. A fixture
     * standing in for Pleiades' `connectsWith`; see the file comment.
     *
     * Both directions come from one table: a road is walked either way, and a consumer forced
     * to check two orders ends up checking one.
     */
    [[nodiscard]] inline core::u32 linkedPlaces(core::u32 id, core::u32 *out, core::u32 capacity) const override;

    /**
     * @brief Map cell a place occupies, for drawing it.
     *
     * @param index Index into @ref kManiPlaces.
     * @param outX  Receives the column.
     * @param outZ  Receives the row.
     * @return false when the place falls outside the survey.
     */
    [[nodiscard]] bool mapCell(core::u32 index, core::i32 &outX, core::i32 &outZ) const noexcept;

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

    /**
     * @brief Builds the region, paves the attested roads, and registers the history and the walk.
     *
     * One grid cell per survey cell: the projection then places a town in the valley it is
     * actually in, rather than in whichever cell a resampling put it.
     *
     * @warning The routes use the cascade, on a grid small enough not to need it -- and that is
     * the point of running it here. A road planned on a summary and refined in its corridor must
     * be the road a flat search finds, and the cheapest place to keep watching that is a world
     * somebody looks at every day.
     *
     * The systems are registered rather than stepped by hand: a step written into a World cannot
     * be ordered against what it touches, cannot be handed a fake in a test, and cannot declare
     * what it reads -- which is exactly why CubePileStepSystem was deleted.
     *
     * What this region is made of is logged once, at init. A map that comes out one colour is
     * either a classifier that ran on nothing or a region that really is all one thing, and only
     * a count tells those apart.
     */
    [[nodiscard]] inline core::Expected<void> onInit(engine::WorldContext &context) override;

    inline void onFixedStep(core::f32 dt) override;

    inline void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override;

    /**
     * @name What the run produced, for a caller that wants to assert on it
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

    /**
     * @name ITerrainQuery
     * @{
     */
    [[nodiscard]] inline bool standable(math::Fixed32 x, math::Fixed32 z) const override;

    [[nodiscard]] bool consumePlantAt(core::i32 /*x*/, core::i32 /*z*/) override { return false; }
    /** @} */

private:
    static constexpr core::u32 kSide = 64u; /**< Matches the survey exactly; see onInit. */
    static constexpr core::i32 kFirstYear = kChronicleFirstYear;
    static constexpr core::i32 kLastYear = kChronicleLastYear;
    static constexpr core::u32 kDaysPerTick = 30u;

    /**
     * @brief World units to a map cell; the inverse of the gazetteer's one conversion.
     */
    [[nodiscard]] static bool toCell(math::Fixed32 x, math::Fixed32 z, core::i32 &outX, core::i32 &outZ) noexcept;

    /**
     * @brief The corpus: who was born where, and where a source says they went.
     *
     * @warning Two travellers rather than one, because a single body cannot show the thing this
     * world is for: two people, two births, two roads, one region -- and their paths cross at
     * Oitylo, where the coast road meets the pass, because that is what a junction IS.
     *
     * The merchant is born at Kardamyli and not Kalamata: a birth off the survey is a body that
     * never appears, and the point of a traveller is that he travels.
     *
     * The journeys are scored, never forced: forcing would put a traveller at his destination and
     * then congratulate him for being there.
     */
    void buildCorpus();

    void addBirth(core::u32 subject, core::u32 place, core::u32 source);

    void addJourney(core::u32 subject, core::u32 place, core::i32 fromYear, core::i32 toYear);

    /**
     * @name Drawing
     * @{
     */

    /**
     * @brief Places the gazetteer knows and this grid cannot hold.
     *
     * @warning Counted rather than named, because it is a property of the WINDOW: re-tile the
     * survey a degree further north and Kalamata is on the map. A hard-coded name would go stale
     * the first time the window moved, silently.
     *
     * @return How many fall outside.
     */
    [[nodiscard]] core::u32 offSurveyPlaces() const noexcept;

    /**
     * @brief How many biomes the classifier actually produced.
     */
    [[nodiscard]] core::u32 distinctBiomes() const noexcept;

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
    [[nodiscard]] static core::u32 pack(procgen::Rgb colour) noexcept;

    void put(core::i32 cellX, core::i32 cellZ, procgen::Rgb colour) noexcept;

    /**
     * @brief Draws the biome map, shaded by slope.
     *
     * The window is cleared first: the map is smaller than the window whenever the zoom does not
     * divide it, and a stale border is indistinguishable from map that is there.
     *
     * Shaded by slope, so relief reads on a flat chart. Without it a mountain and a plain of the
     * same biome are the same rectangle, which is a map of the biome classifier rather than of
     * the ground.
     */
    void drawMap() noexcept;

    void drawRoads() noexcept;

    /**
     * @brief Draws every place on the survey; a place off it is drawn nowhere and counted in the HUD.
     */
    void drawPlaces() noexcept;

    /**
     * @brief Draws every historical body, from the WRITE side of its chunk: where the systems that
     * moved these bodies wrote them.
     */
    void drawTravellers() noexcept;

    /**
     * @brief Writes the readout over the chart.
     *
     * @warning The cascade's own footprint is on screen. Zero coarse expansions would mean the
     * route fell back to a flat search -- which lays an identical network and would look
     * exactly like this.
     *
     * The places line carries three values because that is what the formatter takes, and because
     * the third is the one worth reading: a place the gazetteer holds and this grid cannot. A first
     * version tried to end the line on prose and trimmed the unused value back out by searching
     * the buffer, which found the wrong character and printed "this ghoandurvey".
     */
    void drawHud() noexcept;

    /**
     * @brief Three labelled integers on one line, without a formatting library.
     */
    static void formatLine(char *out, core::u32 capacity, const char *a, core::i32 av, const char *b, core::i32 bv,
                           const char *c, core::i32 cv) noexcept;
    /** @} */

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

#    include "ChronicleWorld.inl"

#endif // LPL_SAMPLES_CHRONICLEWORLD_HPP
