#include <lpl/ecs/Registry.hpp>
#include <lpl/engine/ITerrainQuery.hpp>
#include <lpl/engine/systems/Journey.hpp>
#include <lpl/engine/systems/TerrainRoutes.hpp>
#include <lpl/history/Calendar.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Fold.hpp>
#include <lpl/history/HistorySystem.hpp>
#include <lpl/history/Place.hpp>
#include <lpl/history/PossibleWorld.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(journey);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = lpl::history::kFoldOffsetBasis;

/**
 * @brief Identifiers the journey corpus uses.
 *
 * @warning Fresh numbers, none reused: the corpus of gate P13 history owns 1..3, 10..11, 20..22 and
 *          100..103, and a reused identifier would make one gate's fold depend on the other's
 *          vocabulary.
 */
enum : lpl::core::u32 {
    kSubjectTraveller = 200u, /**< The one body this gate seeds. */
    kPlaceHome = 300u,        /**< Where a source says he was born. */
    kPlaceNear = 301u,        /**< The closest neighbour. He will reach it. */
    kPlaceFar = 302u,         /**< Reachable, later. */
    kPlaceLost = 303u,        /**< Known from texts, never located. Must not be walked to. */
    kPlaceLate = 304u,        /**< Exists only after he is dead. Must not be walked to either. */
    kSourceRegister = 200u,   /**< A register: written to be checked, so trusted enough to seed. */
    /**
     * A local tradition that puts the traveller's birth somewhere else.
     *
     * @warning Its whole purpose is to make a SECOND world possible. Admitted, it seeds the body at
     * the near place instead of home; excluded, the run is the consensus one. Same corpus, same
     * systems, same seed -- one entry more in `admittedSources`, and a different life.
     */
    kSourceTradition = 201u,
};

/**
 * @brief The places the walk may go to, in the order the journey system is given them.
 */
constexpr lpl::core::u32 kJourneyPlaces[] = {kPlaceHome, kPlaceNear, kPlaceFar, kPlaceLost, kPlaceLate};

/**
 * @struct PlaceRow
 * @brief One place of the gazetteer the journey fixture holds in the binary.
 */
struct PlaceRow {
    lpl::core::u32 id;      /**< The gazetteer's identifier. */
    lpl::core::f32 x;       /**< World position. */
    lpl::core::f32 z;       /**< World position. */
    lpl::core::i32 minYear; /**< First year attested. */
    lpl::core::i32 maxYear; /**< Last year attested. */
    bool located;           /**< Whether the coordinates mean anything. */
};

/**
 * @brief The gazetteer of the journey: five places.
 *
 * @warning `kPlaceLost` carries coordinates of zero AND `located` false. The two are not the same
 *          statement, and a resolver that returned (0,0) as a position would send bodies walking
 *          to the origin whenever a corpus mentioned somewhere nobody has found.
 */
constexpr PlaceRow kPlaceRows[] = {
    {kPlaceHome, 0.0f,   0.0f,   -600, 400,  true },
    {kPlaceNear, 300.0f, 0.0f,   -600, 400,  true },
    {kPlaceFar,  300.0f, 600.0f, -600, 400,  true },
    {kPlaceLost, 0.0f,   0.0f,   -600, 400,  false},
    {kPlaceLate, 60.0f,  60.0f,  900,  1200, true },
};

/**
 * @class TableResolver
 * @brief A gazetteer of five places, held in the binary.
 *
 * @warning A table rather than a file, deliberately. The claim this gate tests is that the same
 * arithmetic produces the same journey on a desktop and in ring 0 -- not that a CSV parses. A gate
 * that read a corpus off a disk would go red the day the disk changed, for a reason that has
 * nothing to do with what it measures.
 */
class TableResolver final : public lpl::history::IPlaceResolver {
public:
    bool resolve(lpl::core::u32 id, lpl::history::Place &out) const override
    {
        for (const PlaceRow &row : kPlaceRows)
        {
            if (row.id != id)
                continue;
            out.id = row.id;
            out.x = lpl::math::Fixed32::fromFloat(row.x);
            out.z = lpl::math::Fixed32::fromFloat(row.z);
            out.minYear = row.minYear;
            out.maxYear = row.maxYear;
            out.located = row.located;
            return true;
        }
        return false;
    }

    /**
     * @brief The attested links of the canonical corpus.
     *
     * @warning One link, and it is chosen to CONTRADICT geometry: the far place is further than the
     * near one, so a walk that ignored the corpus would go to the near one first. A fixture whose
     * attested link happens to be the nearest place would pass just as well against a reader that
     * never consulted the corpus at all.
     */
    lpl::core::u32 linkedPlaces(lpl::core::u32 id, lpl::core::u32 *out, lpl::core::u32 capacity) const override
    {
        if (out == nullptr || capacity == 0u)
            return 0u;
        if (id == kPlaceHome)
        {
            out[0] = kPlaceFar;
            return 1u;
        }
        return 0u;
    }
};

/**
 * @class DetourRoutes
 * @brief A road that is deliberately NOT the straight line.
 *
 * @warning The fixture's whole discriminating power. A route that ran straight would be walked
 * identically by a body that ignored the resolver entirely, so the folds would be stable, the
 * arrivals identical, and the gate would prove that roads exist without proving anybody follows
 * one. This road doglegs north before turning east -- the shape a real least-cost route takes
 * around relief -- so a straight walker and a road walker end up at the same place having been in
 * different ones, which is exactly what `position_signature` and `routed_legs` measure.
 */
class DetourRoutes final : public lpl::history::IRouteResolver {
public:
    /**
     * @brief The dogleg from home to the far place; every other pair goes straight, so the
     *        fallback is exercised too.
     */
    lpl::core::u32 route(lpl::core::u32 fromPlace, lpl::core::u32 toPlace, lpl::history::RouteLeg *out,
                         lpl::core::u32 capacity) const override
    {
        if (out == nullptr || capacity < 2u)
            return 0u;
        if (fromPlace != kPlaceHome || toPlace != kPlaceFar)
            return 0u;
        out[0] = lpl::history::RouteLeg{lpl::math::Fixed32::fromFloat(0.0f), lpl::math::Fixed32::fromFloat(300.0f)};
        out[1] = lpl::history::RouteLeg{lpl::math::Fixed32::fromFloat(150.0f), lpl::math::Fixed32::fromFloat(600.0f)};
        return 2u;
    }
};

/**
 * @class RidgeTerrain
 * @brief A boulder beside the road, not a wall across it.
 *
 * @warning **The size is the point, and a first version got it wrong.** A band right across the map
 * stopped the walk dead -- `arrivals=0` -- because greedy avoidance cannot route around a long
 * wall: it turns to the nearest free neighbour and grinds along. That was the fixture being
 * wrong, not the code. **Local avoidance is for what is in front of you; the ROUTE is for what
 * lies between you and your goal**, and asking the step to solve a wall is how bodies end up
 * scraping terrain for a whole run.
 *
 * So: an obstacle a body can see past and step round, placed on the road so a walk that consults
 * the ground and one that does not reach the same places by different paths -- visible in
 * `position_signature` and counted in `avoided`, with both still arriving.
 */
class RidgeTerrain final : public lpl::engine::ITerrainQuery {
public:
    /**
     * @brief Everywhere but a block of about twenty units square, on the road that runs north
     *        from home.
     */
    bool standable(lpl::math::Fixed32 x, lpl::math::Fixed32 z) const override
    {
        const lpl::core::i32 cellZ = z.raw() >> 16;
        const lpl::core::i32 cellX = x.raw() >> 16;

        return cellZ < 140 || cellZ > 160 || cellX < -10 || cellX > 10;
    }

    /**
     * @brief Nothing grows here.
     *
     * @warning A traveller does not graze, and answering "yes there was a plant" would let a walk feed
     * itself on ground the fixture never described. False is the honest answer, not a stub.
     */
    bool consumePlantAt(lpl::core::i32 worldX, lpl::core::i32 worldZ) override
    {
        (void) worldX;
        (void) worldZ;
        return false;
    }
};

/**
 * @class RoadResolver
 * @brief Two places a corpus links, far enough apart that relief gets a say.
 *
 * @warning Its own places rather than the journey's, and its own grid, because the two fixtures
 * measure different things. The walk is calibrated against a hand-written road on purpose (see
 * @ref DetourRoutes); this measures the ROUTER. Folding the router's answer into the walk would
 * mean unifying the two, which needs one description of one world at one resolution -- the ground
 * here is a rule and the relief a field, at different scales -- and that is a design step rather
 * than a fixture swap.
 */
class RoadResolver final : public lpl::history::IPlaceResolver {
public:
    enum : lpl::core::u32 {
        kWest = 400u, /**< South-west of the ridge. */
        kEast = 401u, /**< Due north of it, so the straight line crosses the rock. */
    };

    /**
     * @brief Looks a place up, at the centre of its cell of a 48 by 48 grid of four units a cell:
     *        (cell - 24) * 4 + 2.
     *
     * @param id  The identifier.
     * @param out Receives it.
     * @return false when this table does not carry it.
     */
    [[nodiscard]] bool resolve(lpl::core::u32 id, lpl::history::Place &out) const override
    {
        if (id == kWest)
        {
            out = lpl::history::Place{
                kWest, lpl::math::Fixed32::fromFloat(-46.0f), lpl::math::Fixed32::fromFloat(-54.0f), 0, 0, true};
            return true;
        }
        if (id == kEast)
        {
            out = lpl::history::Place{
                kEast, lpl::math::Fixed32::fromFloat(-46.0f), lpl::math::Fixed32::fromFloat(54.0f), 0, 0, true};
            return true;
        }
        return false;
    }

    /**
     * @brief The one link this corpus attests.
     *
     * @param id       The place.
     * @param out      Receives the neighbours.
     * @param capacity Room in @p out.
     * @return How many were written.
     */
    [[nodiscard]] lpl::core::u32 linkedPlaces(lpl::core::u32 id, lpl::core::u32 *out,
                                              lpl::core::u32 capacity) const override
    {
        if (out == nullptr || capacity == 0u || id != kWest)
            return 0u;
        out[0] = kEast;
        return 1u;
    }
};

/**
 * @struct JourneyFoldResult
 * @brief What a run with BODIES in it produced: what gate P20 journey records.
 *
 * @warning The first fold whose subject is a body that moved on its own. Every earlier history
 * signature folds a corpus being reasoned about; this one folds where somebody ENDED UP -- and the
 * difference matters because history::Divergence only credits a run for events it produced rather
 * than was handed.
 */
struct JourneyFoldResult {
    lpl::core::u32 chronicleSignature{0u}; /**< Fold of every event, caused and earned alike. */
    lpl::core::u32 positionSignature{0u};  /**< Fold of where the bodies finished. */
    lpl::core::u32 deedSignature{0u};      /**< Fold of the emergent arrivals only. */
    lpl::core::u32 seeded{0u};             /**< Bodies a constraint brought into the world. */
    lpl::core::u32 forced{0u};             /**< Times a constraint put a body where it says it was. */
    lpl::core::u32 unplaceable{0u};        /**< Claims naming a place the gazetteer cannot locate. */
    lpl::core::u32 arrivals{0u};           /**< Deeds the run emitted on its own. */
    lpl::core::u32 scoredClaims{0u};       /**< Claims held back to be earned. */
    lpl::core::u32 earned{0u};             /**< Of those, the ones the walk actually reproduced. */
    lpl::core::u32 divergenceScore{0u};    /**< Raw Q16.16 of the score. */

    /**
     * Where the first emergent arrival landed.
     *
     * @warning The number that proves the corpus was consulted at all. The canonical fixture attests a
     * road from the birthplace to the FAR place while a nearer one sits unlinked, so a walk that
     * ignored the corpus would arrive at the near place first -- and every signature above would
     * still be perfectly stable on both targets while measuring a reader that never opened the
     * gazetteer.
     */
    lpl::core::u32 firstArrival{0u};

    /**
     * Waypoints of planned road the walk actually followed.
     *
     * @warning Zero would mean the bodies went straight past every road, which produces the same
     * arrivals at the same places -- so without this number the gate cannot tell a walk that used
     * the terrain from one that ignored it.
     */
    lpl::core::u32 routedLegs{0u};

    /**
     * Times a body turned aside from ground it could not cross.
     *
     * @warning Zero would mean the walk went through the rock -- which it did, silently, until the
     * terrain was wired in. The fixture puts a BOULDER on the road (not a wall across it: see
     * @ref RidgeTerrain for why a wall measured the fixture instead of the code), so a walk that
     * consulted the ground and one that did not reach the same places by different routes, and
     * only this number separates them.
     */
    lpl::core::u32 avoided{0u};

    /**
     * Fold of the road network a corpus's attested links produce on a relief.
     *
     * @warning Folded APART from the walk, and the separation is the point. Everything above is about
     * a body: where it went, what it earned. This is about `procgen::routeLeastCost` itself --
     * a priority queue, Fixed32 costs and an index tie-break, none of which had ever crossed to
     * ring 0 because the pass had no caller at all. Mixing the two would mean a change to the
     * router moved a signature named after the traveller.
     */
    lpl::core::u32 roadSignature{0u};

    /**
     * Fold of the waypoints that road is summarised as.
     *
     * @warning Separate from @ref roadSignature because the two fail differently: the mask says the
     * search found the same cells on both targets, this says they were reduced to the same
     * corners. A router that agreed and a summariser that did not would produce identical roads
     * that bodies walked differently.
     */
    lpl::core::u32 waypointSignature{0u};

    lpl::core::u32 roadCells{0u}; /**< Cells the attested network paved. */
    lpl::core::u32 roadPairs{0u}; /**< Attested pairs a road was laid between. */

    /**
     * The same corpus walked again with one source excluded.
     *
     * @warning **What makes "possible worlds" a mechanism rather than a label.** Gate P13 history
     * already asserts that a timeline built from a restricted view folds differently; this asserts
     * it of a WALK -- a body seeded somewhere else, going somewhere else, ending somewhere else.
     * Nothing forks: it is one run function called twice with one entry more in `admittedSources`.
     */
    lpl::core::u32 alternateChronicle{0u};

    lpl::core::u32 alternateArrivals{0u}; /**< Deeds the alternate world's body emitted on its own. */
    lpl::core::u32 alternateFirst{0u};    /**< Where the alternate world's first emergent arrival landed. */

    /**
     * The same world CLOSED on itself, east-west.
     *
     * @warning One number changes -- how wide the world is -- and every distance the walk measures
     * changes with it: a place beyond the antimeridian is a neighbour rather than most of a
     * circumference away. A walk that ignored `JourneyParams::wrapWidth` would fold identically
     * here, and that is the whole failure it exists to prevent -- a body setting off around the
     * planet the long way, on a road nothing downstream can tell was the wrong one.
     */
    lpl::core::u32 closedChronicle{0u};

    lpl::core::u32 closedArrivals{0u}; /**< Deeds the closed world's body emitted on its own. */
    lpl::core::u32 closedFirst{0u};    /**< Where the closed world's first emergent arrival landed. */

    /**
     * Cells paved when the routing grid CLOSES east-west.
     *
     * @warning The two attested places sit either side of the seam, so a router that could not cross
     * it lays a road most of the way round instead. That road is perfectly valid and perfectly long,
     * and nothing downstream can tell it was the wrong one -- which is why this is folded rather
     * than trusted.
     */
    lpl::core::u32 wrappedRoadCells{0u};

    /**
     * Cells paved with the poles open as well.
     *
     * @warning This grid gives a polar crossing nothing to win, so it must equal
     * @ref wrappedRoadCells. Offering a shortcut must never make a road WORSE -- and a road getting
     * worse is exactly what an inadmissible estimate looks like from the outside, since A* then
     * discards the cheap route before evaluating it.
     */
    lpl::core::u32 polarRoadCells{0u};

    /**
     * Fold of the SAME attested network, planned coarse and refined fine.
     *
     * @warning **It must equal @ref roadSignature, and that equality is the claim.** A cascade is a
     * different search -- a plan on a summary, a corridor, then a confined A* -- so "it returns the
     * road the flat search returns" is a property to prove rather than assume, and proving it on
     * one machine proves it for one machine. Two numbers that must match is how a target that
     * cascaded differently would name itself.
     */
    lpl::core::u32 cascadeRoadSignature{0u};

    lpl::core::u32 cascadeRoadCells{0u}; /**< Cells the cascaded network paved; must equal @ref roadCells. */

    /**
     * Cells the COARSE plans settled.
     *
     * @warning **Zero would mean no cascade happened**, and a run that quietly fell back to a flat
     * search folds an identical road signature and satisfies every check above. This is the number
     * that says the summary was built, planned on, and refined -- and it is folded rather than
     * merely asserted non-zero because a coarse search is arithmetic too, and arithmetic that
     * disagrees between targets is what a gate exists to catch.
     */
    lpl::core::u32 cascadeCoarseExpanded{0u};

    lpl::core::u32 cascadeCorridorCells{0u}; /**< Fine cells the coarse plans opened. */
};

constexpr lpl::core::u32 kRoadGridSize = 48u;
constexpr lpl::core::u32 kRidgeRow = 24u;
constexpr lpl::core::u32 kPassFrom = 40u;
constexpr lpl::core::u32 kMaxWaypoints = 12u;

/**
 * @brief Raises a ridge across row @ref kRidgeRow, from the west edge to column @ref kPassFrom,
 *        which leaves one pass to the east.
 *
 * @details The ridge is what makes the road fold mean anything. Between two places on a plain the
 *          cheapest road IS the straight line, so the signature would be satisfied by a router that
 *          did no search at all, or by one whose priority queue ordered ties differently. A ridge
 *          with one pass forces the search to settle a frontier and break ties, which is the part
 *          that has to agree between a desktop and ring 0.
 */
void raiseRidge(lpl::procgen::Heightfield &field)
{
    for (lpl::core::u32 x = 0u; x < kPassFrom; ++x)
        field.at(x, kRidgeRow) = lpl::math::Fixed32::fromFloat(40.0f);
}

[[nodiscard]] lpl::core::u32 foldPavedCells(const lpl::engine::systems::TerrainRoutes &routes)
{
    lpl::core::u32 signature = kFnv1aOffsetBasis;

    for (lpl::core::u32 cell = 0u; cell < routes.roads().cellCount(); ++cell)
    {
        if (routes.roads()[cell] != 0u)
            signature = lpl::history::foldWord(signature, cell);
    }
    return signature;
}

[[nodiscard]] lpl::core::u32 foldWaypoints(const lpl::engine::systems::TerrainRoutes &routes)
{
    lpl::history::RouteLeg legs[kMaxWaypoints]{};
    const lpl::core::u32 count = routes.route(RoadResolver::kWest, RoadResolver::kEast, legs, kMaxWaypoints);
    lpl::core::u32 signature = lpl::history::foldWord(kFnv1aOffsetBasis, count);

    for (lpl::core::u32 leg = 0u; leg < count; ++leg)
    {
        signature = lpl::history::foldWord(signature, static_cast<lpl::core::u32>(legs[leg].x.raw()));
        signature = lpl::history::foldWord(signature, static_cast<lpl::core::u32>(legs[leg].z.raw()));
    }
    return signature;
}

/**
 * @brief Paves the same network on the same terrain, CLOSED east-west, then open over the poles.
 *
 * @details The two places sit either side of the seam of this grid, so a router that could not
 *          cross it would lay a road most of the way round instead: a perfectly valid, perfectly
 *          long road nothing downstream can question. An implementation that ignored
 *          `wrapColumns` paves both the same. This grid gives a polar crossing nothing to win, so
 *          the polar road must come out the same as the closed one: offering a shortcut must never
 *          make a road worse, because that is what an inadmissible estimate looks like from the
 *          outside.
 */
void paveClosedNetworks(const lpl::procgen::Heightfield &field, const RoadResolver &places,
                        const lpl::engine::systems::TerrainRouteParams &params, const lpl::core::u32 (&order)[2],
                        JourneyFoldResult &out)
{
    lpl::engine::systems::TerrainRouteParams closedParams = params;
    lpl::engine::systems::TerrainRoutes closedRoutes;

    closedParams.cost.wrapColumns = kRoadGridSize;
    closedRoutes.bind(field, places, closedParams);
    out.wrappedRoadCells = closedRoutes.paveAttested(order, 2u);

    lpl::engine::systems::TerrainRouteParams polarParams = closedParams;
    lpl::engine::systems::TerrainRoutes polarRoutes;

    polarParams.cost.wrapPoles = true;
    polarRoutes.bind(field, places, polarParams);
    out.polarRoadCells = polarRoutes.paveAttested(order, 2u);
}

/**
 * @brief Paves the same network planned on a summary and refined inside the corridor that plan
 *        opens.
 *
 * @details A cascade is a genuinely different search, so that it returns the flat search's road is
 *          a claim to prove rather than assume, and proving it on a desktop proves it for a
 *          desktop. The coarse count is folded beside the signature because a run that quietly
 *          fell back to a flat search would produce an identical road.
 */
void paveCascadedNetwork(const lpl::procgen::Heightfield &field, const RoadResolver &places,
                         const lpl::engine::systems::TerrainRouteParams &params, const lpl::core::u32 (&order)[2],
                         JourneyFoldResult &out)
{
    lpl::engine::systems::TerrainRouteParams cascadeParams = params;
    lpl::engine::systems::TerrainRoutes cascadeRoutes;

    cascadeParams.coarseRatio = 4u;
    cascadeRoutes.bind(field, places, cascadeParams);
    out.cascadeRoadCells = cascadeRoutes.paveAttested(order, 2u);
    out.cascadeCoarseExpanded = cascadeRoutes.coarseExpanded();
    out.cascadeCorridorCells = cascadeRoutes.corridorCells();
    out.cascadeRoadSignature = foldPavedCells(cascadeRoutes);
}

/**
 * @brief Folds the road an attested link produces on a relief that refuses the straight line,
 *        then the same network closed, cascaded, and summarised as waypoints.
 *
 * @details The field stands above the routing model's water level, which is zero: a field left at
 *          zero would be sea everywhere, every cell would pay the same water penalty, and the
 *          relief would be flat again under a uniform surcharge.
 *
 * @param out Receives the road signatures and counts.
 */
void foldAttestedRoads(JourneyFoldResult &out)
{
    lpl::procgen::Heightfield field{kRoadGridSize, kRoadGridSize, lpl::math::Fixed32::fromFloat(1.0f)};
    RoadResolver places;
    lpl::engine::systems::TerrainRouteParams params;
    lpl::engine::systems::TerrainRoutes routes;
    const lpl::core::u32 order[2] = {RoadResolver::kWest, RoadResolver::kEast};

    raiseRidge(field);
    params.cellSize = 4u;
    params.cost.slopePenalty = 6.0f;
    routes.bind(field, places, params);
    out.roadCells = routes.paveAttested(order, 2u);
    out.roadPairs = routes.pairs();
    out.roadSignature = foldPavedCells(routes);
    paveClosedNetworks(field, places, params, order, out);
    paveCascadedNetwork(field, places, params, order, out);
    out.waypointSignature = foldWaypoints(routes);
}

[[nodiscard]] lpl::history::Fact travellerFact(lpl::history::Predicate predicate, lpl::core::u32 place,
                                               lpl::core::i32 firstYear, lpl::core::i32 lastYear, lpl::core::u32 source,
                                               lpl::core::f32 sigma)
{
    lpl::history::Fact fact;

    fact.subject = kSubjectTraveller;
    fact.predicate = static_cast<lpl::core::u32>(predicate);
    fact.object = place;
    fact.fromDay = lpl::history::firstDayOfYear(firstYear);
    fact.toDay = lpl::history::lastDayOfYear(lastYear);
    fact.source = source;
    fact.sigma = lpl::math::Fixed32::fromFloat(sigma);
    return fact;
}

[[nodiscard]] lpl::history::SourceProfile notarialSource(lpl::core::u32 id)
{
    lpl::history::SourceProfile profile;

    profile.id = id;
    profile.kind = lpl::history::SourceKind::Notarial;
    profile.yearsAfterEvent = 0u;
    return profile;
}

/**
 * @brief The corpus of the journey: a birth that is FORCED, a source states it and the run must
 *        honour it, and arrivals that are only SCORED, so the walk has something to earn rather
 *        than replay.
 *
 * @warning The register is notarial, and MEASURED rather than chosen. A first fixture used a
 *          chronicle at sigma 0.9 and nothing was ever seeded: the fused confidence came out at
 *          0.493, under the 0.60 seed threshold, so the birth became a Score and no body entered
 *          the world. That is the module being honest -- one partial account is not enough to put
 *          a man somewhere -- and a birthplace is exactly what a register records anyway.
 *
 * @warning The tradition is the dissenting source, and the only reason a second world is possible:
 *          honest and partial, and worth enough on its own to seed a body somewhere else. Its whole
 *          job is to make the two views disagree about where a life began.
 *
 * @warning The register knows the YEAR of the birth, so the window is the whole of it: a single day
 *          would claim a precision no register of this period has, and the interval is where
 *          precision lives.
 *
 * @warning TWO scored claims, one the run can reach and one it cannot, and both are needed. Scored
 *          means never forced: forcing would put the body at its destination and then congratulate
 *          it for being there. A fixture where nothing is earned passes against a walk that never
 *          moves, and one where everything is earned passes against a body teleported to its
 *          destination; only a fixture holding both can tell those apart from a real walk.
 *          Following the attested road and its dogleg, the walk reaches the far place in -494 and
 *          the near one in -488. The first claim dates the near arrival to -496..-490, which the run
 *          misses by two years -- arriving at the wrong time is not the same event, and a source
 *          can simply be wrong about a date. The second covers the far arrival, and is earned.
 *          These windows are calibrated to make the fixture discriminate; they are not evidence.
 */
void buildJourneyCorpus(lpl::history::Corpus &corpus)
{
    corpus.sources.push_back(notarialSource(kSourceRegister));
    corpus.sources.push_back(notarialSource(kSourceTradition));
    corpus.facts.push_back(
        travellerFact(lpl::history::Predicate::BornAt, kPlaceNear, -500, -500, kSourceTradition, 0.99f));
    corpus.facts.push_back(
        travellerFact(lpl::history::Predicate::BornAt, kPlaceHome, -500, -500, kSourceRegister, 0.98f));
    corpus.facts.push_back(
        travellerFact(lpl::history::Predicate::TravelledTo, kPlaceNear, -496, -490, kSourceRegister, 0.3f));
    corpus.facts.push_back(
        travellerFact(lpl::history::Predicate::TravelledTo, kPlaceFar, -496, -492, kSourceRegister, 0.3f));
}

/**
 * @brief Folds the emergent arrivals alone, and notes where the first one landed.
 *
 * @details Apart from the chronicle: a signature over both kinds of event moves whenever the
 *          timeline changes, and would say nothing about the walk, the only thing this gate exists
 *          to measure.
 */
void foldDeeds(const lpl::history::Chronicle &chronicle, JourneyFoldResult &result)
{
    lpl::core::u32 signature = kFnv1aOffsetBasis;

    for (lpl::core::u32 index = 0u; index < chronicle.size(); ++index)
    {
        const lpl::history::Event &event = chronicle.at(index);

        if (event.attestation.cause != lpl::history::Cause::Emergent)
            continue;
        if (result.firstArrival == 0u)
            result.firstArrival = event.fact.object;

        const lpl::core::u32 words[] = {event.fact.subject, event.fact.predicate, event.fact.object,
                                        static_cast<lpl::core::u32>(event.fact.fromDay)};

        for (const lpl::core::u32 word : words)
            signature = lpl::history::foldWord(signature, word);
    }
    result.deedSignature = signature;
}

[[nodiscard]] lpl::core::u32 foldBodies(lpl::ecs::Chunk &chunk, lpl::core::u32 signature)
{
    const auto *positions =
        static_cast<const lpl::math::Vec3<lpl::math::Fixed32> *>(chunk.writeComponent(lpl::ecs::ComponentId::Position));
    const auto *bodies =
        static_cast<const lpl::ecs::HistoricalBody *>(chunk.writeComponent(lpl::ecs::ComponentId::Historical));

    if (positions == nullptr || bodies == nullptr)
        return signature;
    for (lpl::core::u32 row = 0u; row < chunk.count(); ++row)
    {
        signature = lpl::history::foldWord(signature, bodies[row].subject);
        signature = lpl::history::foldWord(signature, static_cast<lpl::core::u32>(positions[row].x.raw()));
        signature = lpl::history::foldWord(signature, static_cast<lpl::core::u32>(positions[row].z.raw()));
    }
    return signature;
}

[[nodiscard]] lpl::core::u32 foldPositions(lpl::ecs::Registry &registry)
{
    lpl::core::u32 signature = kFnv1aOffsetBasis;

    for (const auto &partition : registry.partitions())
    {
        if (partition == nullptr || !partition->archetype().has(lpl::ecs::ComponentId::Historical))
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (chunk != nullptr)
                signature = foldBodies(*chunk, signature);
        }
    }
    return signature;
}

/**
 * @brief Walks the corpus through one possible world, and folds what happened.
 *
 * @details A possible world is the same corpus through the same systems with one entry more in
 *          `admittedSources`, so the run is written once and called per world: copying it would
 *          let two worlds drift apart for reasons that have nothing to do with their sources.
 *          Constraints run before the walk on every tick, because a body must exist before it
 *          moves and a Force must land before the walk reads the position it holds; and the walk
 *          is told once, before the first tick, which place the seeded position is, so the
 *          attested roads out of it are offered from the first leg.
 *
 * @param corpus    What the world is built from.
 * @param worldView Which sources it listens to; empty admits them all.
 * @param wrapWidth Width at which the world closes east-west, or zero for an open world.
 * @param result    Receives the signatures and counters of the walk.
 */
void walkWorld(const lpl::history::Corpus &corpus, const lpl::history::WorldView &worldView,
               lpl::math::Fixed32 wrapWidth, JourneyFoldResult &result)
{
    lpl::history::FusionReport report;
    const lpl::history::Timeline timeline = lpl::history::buildTimeline(corpus, worldView, report);
    const lpl::history::Era era = lpl::history::Era::ofYears(-500, -480, 4u);
    lpl::history::Chronicle chronicle;
    lpl::ecs::Registry registry;
    TableResolver resolver;
    lpl::ecs::Archetype archetype;

    archetype.add(lpl::ecs::ComponentId::Position);
    archetype.add(lpl::ecs::ComponentId::Historical);

    lpl::history::HistorySystem constraints{timeline, era, chronicle};
    lpl::engine::systems::JourneyParams params;

    constraints.bindWorld(registry, resolver, archetype);
    params.pacePerYear = lpl::math::Fixed32::fromFloat(120.0f);
    params.arrivalRadius = lpl::math::Fixed32::fromFloat(8.0f);
    params.horizon = lpl::math::Fixed32::fromFloat(4000.0f);
    params.wrapWidth = wrapWidth;

    lpl::engine::systems::JourneySystem journey{registry, resolver, kJourneyPlaces, 5u, era, chronicle, params};
    DetourRoutes routes;
    RidgeTerrain ground;

    journey.useRoutes(routes);
    journey.useTerrain(ground);
    journey.placeBodyAt(kSubjectTraveller, kPlaceHome);
    for (lpl::core::u32 tick = 0u; tick < era.totalTicks(); ++tick)
    {
        constraints.execute(0.0f);
        journey.execute(0.0f);
        registry.swapAllBuffers();
    }

    result.seeded = constraints.seeded();
    result.forced = constraints.forced();
    result.unplaceable = constraints.unplaceable();
    result.arrivals = journey.arrivals();
    result.routedLegs = journey.routedLegs();
    result.avoided = journey.avoided();
    result.chronicleSignature = chronicle.fold(kFnv1aOffsetBasis);
    foldDeeds(chronicle, result);
    result.positionSignature = foldPositions(registry);

    const lpl::history::Divergence verdict = lpl::history::measureDivergence(chronicle, timeline);

    result.scoredClaims = verdict.scoredClaims;
    result.earned = verdict.earned;
    result.divergenceScore = static_cast<lpl::core::u32>(verdict.score.raw());
}

/**
 * @brief Seeds a body from a corpus, lets it walk in three worlds, and folds what happened, and
 *        the roads an attested link lays on a relief.
 *
 * @details The canonical journey: one traveller, five places, a birth that is forced and an arrival
 *          that is only scored. The places are a table rather than a gazetteer file: a gate that
 *          read a corpus from disk would fail when the disk changed, and the claim is that the same
 *          arithmetic gives the same journey on two targets, not that a CSV parses.
 *
 *          The consensus world admits every source. The alternate world listens to the register
 *          alone: if it folded identically, possible worlds would be a label rather than a
 *          mechanism. The closed world is the consensus one, closed east-west at 120 units, a width
 *          MEASURED against this fixture: the places sit 300 apart, so a wrap of 250, 350, 400,
 *          500 or 620 leaves every distance as it was and folds identically, passing for the wrong
 *          reason. Only a world narrow enough that the far places sit past the seam moves
 *          anything.
 *
 * @param out Receives the signatures.
 */
void foldJourneyState(JourneyFoldResult &out)
{
    out = JourneyFoldResult{};

    lpl::history::Corpus corpus;
    lpl::history::WorldView consensus;
    lpl::history::WorldView restricted;
    lpl::history::WorldView open;
    JourneyFoldResult alternate{};
    JourneyFoldResult closed{};

    buildJourneyCorpus(corpus);
    walkWorld(corpus, consensus, lpl::math::Fixed32{}, out);

    restricted.admittedSources.push_back(kSourceRegister);
    walkWorld(corpus, restricted, lpl::math::Fixed32{}, alternate);
    out.alternateChronicle = alternate.chronicleSignature;
    out.alternateArrivals = alternate.arrivals;
    out.alternateFirst = alternate.firstArrival;

    walkWorld(corpus, open, lpl::math::Fixed32::fromFloat(120.0f), closed);
    out.closedChronicle = closed.chronicleSignature;
    out.closedArrivals = closed.arrivals;
    out.closedFirst = closed.firstArrival;

    foldAttestedRoads(out);
}

} // namespace

/**
 * @brief Gate P20 journey: a body seeded by a constraint walks on its own, along attested roads and
 *        around ground it cannot cross, and earns some of the timeline's claims but not all.
 *
 * @details The fixture holds a place nearer than the attested destination but unlinked, a road
 *          that doglegs, a boulder on the way, and a ridge whose shortest way round is 57 cells:
 *          each number below separates a walk that read the corpus from one that only measured
 *          distances, which every signature would fold just as stably.
 */
LPL_TEST(a_body_walks_the_attested_roads)
{
    JourneyFoldResult journey{};

    foldJourneyState(journey);
    test.check(journey.seeded == 1u, "a constraint seeds a body");
    test.check(journey.arrivals >= 2u, "and the body goes places on its own");
    test.check(journey.earned >= 1u && journey.earned < journey.scoredClaims,
               "it earns some of the scored claims, and not all");
    test.check(journey.unplaceable == 0u, "it never walks to a place the gazetteer cannot locate");
    test.check(journey.firstArrival == 302u, "it follows the attested road, not the nearest place");
    test.check(journey.routedLegs >= 2u, "along the road's waypoints rather than the straight line");
    test.check(journey.avoided >= 1u, "and turns aside from ground it cannot cross");
    test.check(journey.scoredClaims != 0u &&
                   journey.divergenceScore == (65536u * journey.earned) / journey.scoredClaims,
               "the divergence score is the earned fraction");

    test.measureHexadecimal("chronicle_signature", journey.chronicleSignature);
    test.measureHexadecimal("position_signature", journey.positionSignature);
    test.measureHexadecimal("deed_signature", journey.deedSignature);
    test.measure("forced", journey.forced);
    test.measure("arrivals", journey.arrivals);
    test.measure("scored_claims", journey.scoredClaims);
    test.measure("earned", journey.earned);
    test.measureHexadecimal("divergence_score", journey.divergenceScore);
    test.measure("routed_legs", journey.routedLegs);
    test.measure("avoided", journey.avoided);
}

/**
 * @brief The attested link becomes one road, the shortest one round the ridge, and closing or
 *        cascading the routing grid changes the road only where it should.
 *
 * @details 57 cells is derived from the fixture: the ridge fills columns 0 to 39 of its row, so
 *          the road goes 28 columns east, crosses, and comes 28 back. A free climb lays 28 cells
 *          straight through the mountain, and would fold just as stably.
 */
LPL_TEST(roads_take_the_shortest_way_round)
{
    JourneyFoldResult journey{};

    foldJourneyState(journey);
    test.check(journey.roadPairs == 1u, "the attested link becomes exactly one road");
    test.check(journey.roadCells == 57u, "the shortest one that goes round the ridge");
    test.check(journey.roadSignature != 0u && journey.waypointSignature != 0u, "the road and its waypoints fold");
    test.check(journey.wrappedRoadCells > 0u && journey.wrappedRoadCells < journey.roadCells,
               "a grid closed at its seam lays a shorter road");
    test.check(journey.polarRoadCells <= journey.wrappedRoadCells, "and offering the poles never makes it longer");
    test.check(journey.cascadeRoadSignature == journey.roadSignature && journey.cascadeRoadCells == journey.roadCells,
               "a road planned coarse and refined fine is the road found flat, cell for cell");
    test.check(journey.cascadeCoarseExpanded > 0u && journey.cascadeCorridorCells > 0u,
               "because a coarse plan ran and opened a corridor");

    test.measureHexadecimal("road_signature", journey.roadSignature);
    test.measureHexadecimal("waypoint_signature", journey.waypointSignature);
    test.measure("road_cells", journey.roadCells);
    test.measure("wrapped_road_cells", journey.wrappedRoadCells);
    test.measure("polar_road_cells", journey.polarRoadCells);
    test.measure("cascade_coarse_expanded", journey.cascadeCoarseExpanded);
    test.measure("cascade_corridor_cells", journey.cascadeCorridorCells);
}

/**
 * @brief One more admitted source, or a world closed at its seam, gives a body a different life:
 *        "possible worlds" is a mechanism, not a label.
 */
LPL_TEST(other_worlds_are_walked_differently)
{
    JourneyFoldResult journey{};

    foldJourneyState(journey);
    test.check(journey.alternateChronicle != 0u && journey.alternateChronicle != journey.chronicleSignature,
               "the world according to one source is another world");
    test.check(journey.alternateArrivals >= 1u, "where the body still goes somewhere");
    test.check(journey.closedChronicle != journey.chronicleSignature,
               "a closed world folds differently from an open one");
    test.check(journey.closedArrivals >= 1u, "and its body goes somewhere too");

    test.measureHexadecimal("alternate_chronicle_signature", journey.alternateChronicle);
    test.measure("alternate_arrivals", journey.alternateArrivals);
    test.measure("alternate_first_arrival", journey.alternateFirst);
    test.measureHexadecimal("closed_chronicle_signature", journey.closedChronicle);
    test.measure("closed_arrivals", journey.closedArrivals);
    test.measure("closed_first_arrival", journey.closedFirst);
}
