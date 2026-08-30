/**
 * @file JourneyParity.cpp
 * @brief The gate P20 fold, which needs a WORLD and therefore cannot live beside the others.
 *
 * @warning Separate from `Parity.cpp` for a reason measured rather than aesthetic: LplKnowledge links
 * `lpl::history` for its own gate P18 and compiles `Parity.cpp` directly -- but it does NOT link
 * the ECS, because the arithmetic of doubt has no business knowing what an entity is. Folding a
 * journey into the same translation unit dragged `<lpl/ecs/Registry.hpp>` into that build and
 * broke eight LplKnowledge test targets at once, none of which had changed.
 *
 * The layering it restores is the one `history/` already claimed: corpus-only and freestanding
 * up to `Parity.cpp`, and engine-side the moment a Registry is involved.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/systems/Journey.hpp>
#include <lpl/engine/systems/TerrainRoutes.hpp>

#include <lpl/ecs/Registry.hpp>
#include <lpl/history/Calendar.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Fold.hpp>
#include <lpl/history/HistorySystem.hpp>
#include <lpl/history/Parity.hpp>
#include <lpl/history/PossibleWorld.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::engine::systems {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = history::kFoldOffsetBasis;

} // namespace

namespace {

/**
 * @brief Identifiers the journey corpus uses.
 *
 * @warning Fresh numbers, none reused: the P13 corpus owns 1..3, 10..11, 20..22 and 100..103, and a
 * reused identifier would make one gate's fold depend on the other's vocabulary.
 */
enum : core::u32 {
    kSubjectTraveller = 200u, ///< The one body this gate seeds.
    kPlaceHome = 300u,        ///< Where a source says he was born.
    kPlaceNear = 301u,        ///< The closest neighbour. He will reach it.
    kPlaceFar = 302u,         ///< Reachable, later.
    kPlaceLost = 303u,        ///< Known from texts, never located. Must not be walked to.
    kPlaceLate = 304u,        ///< Exists only after he is dead. Must not be walked to either.
    kSourceRegister = 200u,   ///< A register: written to be checked, so trusted enough to seed.
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
 * @class TableResolver
 * @brief A gazetteer of five places, held in the binary.
 *
 * @warning A table rather than a file, deliberately. The claim this gate tests is that the same
 * arithmetic produces the same journey on a desktop and in ring 0 -- not that a CSV parses. A gate
 * that read a corpus off a disk would go red the day the disk changed, for a reason that has
 * nothing to do with what it measures.
 */
class TableResolver final : public history::IPlaceResolver {
public:
    bool resolve(core::u32 id, history::Place &out) const override
    {
        struct Row {
            core::u32 id;
            float x;
            float z;
            core::i32 minYear;
            core::i32 maxYear;
            bool located;
        };
        // @warning `kPlaceLost` carries coordinates of zero AND `located` false. The two are not the
        // same statement, and a resolver that returned (0,0) as a position would send bodies
        // walking to the origin whenever a corpus mentioned somewhere nobody has found.
        static constexpr Row kRows[] = {
            {kPlaceHome, 0.0f,   0.0f,   -600, 400,  true },
            {kPlaceNear, 300.0f, 0.0f,   -600, 400,  true },
            {kPlaceFar,  300.0f, 600.0f, -600, 400,  true },
            {kPlaceLost, 0.0f,   0.0f,   -600, 400,  false},
            {kPlaceLate, 60.0f,  60.0f,  900,  1200, true },
        };
        for (const Row &row : kRows)
        {
            if (row.id != id)
                continue;
            out.id = row.id;
            out.x = math::Fixed32::fromFloat(row.x);
            out.z = math::Fixed32::fromFloat(row.z);
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
    core::u32 linkedPlaces(core::u32 id, core::u32 *out, core::u32 capacity) const override
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
 * different ones, which is exactly what `position_sig` and `routed` measure.
 */
class DetourRoutes final : public history::IRouteResolver {
public:
    core::u32 route(core::u32 fromPlace, core::u32 toPlace, history::RouteLeg *out, core::u32 capacity) const override
    {
        if (out == nullptr || capacity < 2u)
            return 0u;
        if (fromPlace != kPlaceHome || toPlace != kPlaceFar)
            return 0u; // every other pair goes straight, so the fallback is exercised too
        out[0] = history::RouteLeg{math::Fixed32::fromFloat(0.0f), math::Fixed32::fromFloat(300.0f)};
        out[1] = history::RouteLeg{math::Fixed32::fromFloat(150.0f), math::Fixed32::fromFloat(600.0f)};
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
 * `position_sig` and counted in `avoided`, with both still arriving.
 */
class RidgeTerrain final : public ITerrainQuery {
public:
    bool standable(math::Fixed32 x, math::Fixed32 z) const override
    {
        // A block of about twenty units square, sitting on the road that runs north from home.
        const core::i32 zi = z.raw() >> 16;
        const core::i32 xi = x.raw() >> 16;
        return zi < 140 || zi > 160 || xi < -10 || xi > 10;
    }

    /**
     * @brief Nothing grows here.
     *
     * @warning A traveller does not graze, and answering "yes there was a plant" would let a walk feed
     * itself on ground the fixture never described. False is the honest answer, not a stub.
     */
    bool consumePlantAt(core::i32 worldX, core::i32 worldZ) override
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
class RoadResolver final : public history::IPlaceResolver {
public:
    enum : core::u32 {
        kWest = 400u, ///< South-west of the ridge.
        kEast = 401u, ///< Due north of it, so the straight line crosses the rock.
    };

    /**
     * @brief Looks a place up.
     *
     * @param id  The identifier.
     * @param out Receives it.
     * @return false when this table does not carry it.
     */
    [[nodiscard]] bool resolve(core::u32 id, history::Place &out) const override
    {
        // Cell centres of a 48x48 grid at four units a cell: (cell - 24) * 4 + 2.
        if (id == kWest)
        {
            out = history::Place{kWest, math::Fixed32::fromFloat(-46.0f), math::Fixed32::fromFloat(-54.0f), 0, 0, true};
            return true;
        }
        if (id == kEast)
        {
            out = history::Place{kEast, math::Fixed32::fromFloat(-46.0f), math::Fixed32::fromFloat(54.0f), 0, 0, true};
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
    [[nodiscard]] core::u32 linkedPlaces(core::u32 id, core::u32 *out, core::u32 capacity) const override
    {
        if (out == nullptr || capacity == 0u || id != kWest)
            return 0u;
        out[0] = kEast;
        return 1u;
    }
};

/**
 * @brief Folds the road an attested link produces on a relief that refuses the straight line.
 *
 * @warning The ridge is what makes this fold mean anything. Between two places on a plain the
 * cheapest road IS the straight line, so the signature would be satisfied by a router that did no
 * search at all -- and by one whose priority queue ordered ties differently, since on flat ground
 * there are no ties that matter. A ridge with one pass forces the search to settle a frontier and
 * break ties, which is the part that has to agree between a desktop and ring 0.
 *
 * @param out Receives the road signatures and counts.
 */
void foldAttestedRoads(JourneyFoldResult &out)
{
    constexpr core::u32 kSize = 48u;
    constexpr core::u32 kRidgeRow = 24u;
    constexpr core::u32 kPassFrom = 40u;

    // Above the routing model's water level, which is zero: a field left at zero would be sea
    // everywhere, so every cell would pay the same water penalty and the relief would be flat
    // again under a uniform surcharge.
    procgen::Heightfield field{kSize, kSize, math::Fixed32::fromFloat(1.0f)};
    for (core::u32 x = 0u; x < kPassFrom; ++x)
        field.at(x, kRidgeRow) = math::Fixed32::fromFloat(40.0f);

    RoadResolver places;
    TerrainRouteParams params;
    params.cellSize = 4u;
    params.cost.slopePenalty = 6.0f;

    TerrainRoutes routes;
    routes.bind(field, places, params);

    const core::u32 order[2] = {RoadResolver::kWest, RoadResolver::kEast};
    out.roadCells = routes.paveAttested(order, 2u);
    out.roadPairs = routes.pairs();

    core::u32 roadSignature = kFnv1aOffsetBasis;
    for (core::u32 cell = 0u; cell < routes.roads().cellCount(); ++cell)
    {
        if (routes.roads()[cell] != 0u)
            roadSignature = history::foldWord(roadSignature, cell);
    }
    out.roadSignature = roadSignature;

    // @warning **The same terrain, CLOSED east-west and open over the poles, routed again.** The two
    // places sit either side of the seam of this grid, so a router that could not cross it would lay
    // a road most of the way round instead -- a perfectly valid, perfectly long road that nothing
    // downstream can question. Folding both is what says the wrap is doing something: an
    // implementation that ignored `wrapColumns` folds these two identically.
    TerrainRouteParams closedParams = params;
    closedParams.cost.wrapColumns = kSize;
    TerrainRoutes closedRoutes;
    closedRoutes.bind(field, places, closedParams);
    out.wrappedRoadCells = closedRoutes.paveAttested(order, 2u);

    // And once more with the poles open. A grid this shape gives a polar crossing nothing to win,
    // so the road must come out the SAME as the merely-closed one -- which is the claim that
    // matters: offering a shortcut must never make a road worse, because that is what an
    // inadmissible estimate looks like from the outside.
    TerrainRouteParams polarParams = closedParams;
    polarParams.cost.wrapPoles = true;
    TerrainRoutes polarRoutes;
    polarRoutes.bind(field, places, polarParams);
    out.polarRoadCells = polarRoutes.paveAttested(order, 2u);

    // @warning **The same network, planned on a summary and refined inside the corridor that plan
    // opens.** A cascade is a genuinely different search, so that it returns the flat search's road
    // is a claim to prove rather than assume -- and proving it on a desktop proves it for a
    // desktop. The coarse count is folded beside the signature because a run that quietly fell back
    // to a flat search would produce an IDENTICAL road and satisfy everything else here.
    TerrainRouteParams cascadeParams = params;
    cascadeParams.coarseRatio = 4u;
    TerrainRoutes cascadeRoutes;
    cascadeRoutes.bind(field, places, cascadeParams);
    out.cascadeRoadCells = cascadeRoutes.paveAttested(order, 2u);
    out.cascadeCoarseExpanded = cascadeRoutes.coarseExpanded();
    out.cascadeCorridorCells = cascadeRoutes.corridorCells();

    core::u32 cascadeSignature = kFnv1aOffsetBasis;
    for (core::u32 cell = 0u; cell < cascadeRoutes.roads().cellCount(); ++cell)
    {
        if (cascadeRoutes.roads()[cell] != 0u)
            cascadeSignature = history::foldWord(cascadeSignature, cell);
    }
    out.cascadeRoadSignature = cascadeSignature;

    history::RouteLeg legs[12]{};
    const core::u32 count = routes.route(RoadResolver::kWest, RoadResolver::kEast, legs, 12u);
    core::u32 waypointSignature = kFnv1aOffsetBasis;
    waypointSignature = history::foldWord(waypointSignature, count);
    for (core::u32 i = 0u; i < count; ++i)
    {
        waypointSignature = history::foldWord(waypointSignature, static_cast<core::u32>(legs[i].x.raw()));
        waypointSignature = history::foldWord(waypointSignature, static_cast<core::u32>(legs[i].z.raw()));
    }
    out.waypointSignature = waypointSignature;
}

} // namespace

void foldJourneyState(JourneyFoldResult &out)
{
    out = JourneyFoldResult{};

    // The corpus. A birth that is FORCED -- a source states it and the run must honour it -- and
    // an arrival that is only SCORED, so the walk has something to earn rather than replay.
    history::Corpus corpus;
    {
        // @warning Notarial, and MEASURED rather than chosen. A first fixture used a chronicle at
        // sigma 0.9 and nothing was ever seeded: the fused confidence came out at 0.493, under
        // the 0.60 seed threshold, so the birth became a Score and no body entered the world.
        // That is the module being honest -- one partial account is not enough to put a man
        // somewhere -- and a birthplace is exactly what a register records anyway.
        history::SourceProfile register_;
        register_.id = kSourceRegister;
        register_.kind = history::SourceKind::Notarial;
        register_.yearsAfterEvent = 0u;
        corpus.sources.push_back(register_);

        // @warning The dissenting source, and the only reason a second world is possible. A local
        // tradition, so it is a Chronicle rather than a register: honest and partial, and worth
        // enough on its own to seed a body somewhere else. Its whole job is to make the two views
        // disagree about where a life began.
        history::SourceProfile tradition;
        tradition.id = kSourceTradition;
        tradition.kind = history::SourceKind::Notarial;
        tradition.yearsAfterEvent = 0u;
        corpus.sources.push_back(tradition);

        history::Fact bornElsewhere;
        bornElsewhere.subject = kSubjectTraveller;
        bornElsewhere.predicate = static_cast<core::u32>(history::Predicate::BornAt);
        bornElsewhere.object = kPlaceNear;
        bornElsewhere.fromDay = history::firstDayOfYear(-500);
        bornElsewhere.toDay = history::lastDayOfYear(-500);
        bornElsewhere.source = kSourceTradition;
        bornElsewhere.sigma = math::Fixed32::fromFloat(0.99f);
        corpus.facts.push_back(bornElsewhere);

        history::Fact born;
        born.subject = kSubjectTraveller;
        born.predicate = static_cast<core::u32>(history::Predicate::BornAt);
        born.object = kPlaceHome;
        // @warning The register knows the YEAR, so the window is the whole of it. Writing a single
        // day here would claim a precision no register of this period has -- and the interval is
        // where precision now lives, so a narrow window is a positive assertion of sharpness.
        born.fromDay = history::firstDayOfYear(-500);
        born.toDay = history::lastDayOfYear(-500);
        born.source = kSourceRegister;
        born.sigma = math::Fixed32::fromFloat(0.98f);
        corpus.facts.push_back(born);

        // @warning TWO scored claims, one the run can reach and one it cannot, and both are needed.
        // Scored means never forced: forcing would put the body at its destination and then
        // congratulate it for being there. But a fixture where nothing is earned does not
        // discriminate -- it passes just as well against a walk that never moves -- and one where
        // everything is earned does not either.
        //
        // The dates are stated plainly rather than hidden. Following the attested road and its
        // dogleg, the walk reaches the far place in -494 and the near one in -488. The first
        // claim below dates the near arrival to -496..-490, which the run MISSES by two years --
        // arriving at the wrong time is not the same event, and a source can simply be wrong
        // about a date. The second covers the far arrival, and is earned.
        //
        // @warning Calibrated, and said so: these are windows chosen to make the fixture discriminate,
        // not evidence. A gate where nothing is earned passes against a body that never moves,
        // and one where everything is earned passes against a body teleported to its
        // destination; only a fixture holding both can tell those apart from a real walk.
        history::Fact wentNear;
        wentNear.subject = kSubjectTraveller;
        wentNear.predicate = static_cast<core::u32>(history::Predicate::TravelledTo);
        wentNear.object = kPlaceNear;
        wentNear.fromDay = history::firstDayOfYear(-496);
        wentNear.toDay = history::lastDayOfYear(-490);
        wentNear.source = kSourceRegister;
        wentNear.sigma = math::Fixed32::fromFloat(0.3f);
        corpus.facts.push_back(wentNear);

        history::Fact wentFar;
        wentFar.subject = kSubjectTraveller;
        wentFar.predicate = static_cast<core::u32>(history::Predicate::TravelledTo);
        wentFar.object = kPlaceFar;
        wentFar.fromDay = history::firstDayOfYear(-496);
        wentFar.toDay = history::lastDayOfYear(-492);
        wentFar.source = kSourceRegister;
        wentFar.sigma = math::Fixed32::fromFloat(0.3f);
        corpus.facts.push_back(wentFar);
    }

    // @warning **One run, two worlds, and nothing forks.** A possible world is the same corpus through
    // the same systems with one entry more in `admittedSources` -- so the run is written ONCE, as
    // a lambda over the view, and called twice. Copying it would let the consensus world and the
    // dissenting one drift apart in a way that has nothing to do with what their sources say,
    // which is precisely the comparison this gate exists to make.
    const auto runWorld = [&](const history::WorldView &worldView, math::Fixed32 wrapWidth, JourneyFoldResult &result) {
        // An all-listening view: an empty `admittedSources` admits every source, which is what a
        // consensus world means here.
        history::FusionReport report;
        const history::Timeline timeline = history::buildTimeline(corpus, worldView, report);

        const history::Era era = history::Era::ofYears(-500, -480, 4u);

        history::Chronicle chronicle;
        ecs::Registry registry;
        TableResolver resolver;

        ecs::Archetype archetype;
        archetype.add(ecs::ComponentId::Position);
        archetype.add(ecs::ComponentId::Historical);

        history::HistorySystem constraints{timeline, era, chronicle};
        constraints.bindWorld(registry, resolver, archetype);

        static constexpr core::u32 kPlaces[] = {kPlaceHome, kPlaceNear, kPlaceFar, kPlaceLost, kPlaceLate};
        JourneyParams params;
        params.pacePerYear = math::Fixed32::fromFloat(120.0f);
        params.arrivalRadius = math::Fixed32::fromFloat(8.0f);
        params.horizon = math::Fixed32::fromFloat(4000.0f);
        params.wrapWidth = wrapWidth;
        JourneySystem journey{registry, resolver, kPlaces, 5u, era, chronicle, params};
        DetourRoutes routes;
        journey.useRoutes(routes);
        RidgeTerrain ground;
        journey.useTerrain(ground);

        // @warning Constraints BEFORE the walk, every tick: a body must exist before it can move, and a
        // Force must land before the walk reads the position it is meant to hold.
        // @warning The walk is told where the corpus put him, once, before the first tick. The seed
        // constraint positions the body; this tells the WALK which place that position is, so the
        // attested roads out of it are offered from the first leg rather than from the second.
        journey.placeBodyAt(kSubjectTraveller, kPlaceHome);

        for (core::u32 tick = 0u; tick < era.totalTicks(); ++tick)
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

        // -- The three folds ------------------------------------------------------
        // The whole account, through the module's own fold: one function produces this number, so
        // the gate cannot disagree with the chronicle about what a chronicle folds to.
        result.chronicleSignature = chronicle.fold(kFnv1aOffsetBasis);

        // And the earned half on its own. @warning Folded apart because a signature over BOTH kinds moves
        // whenever the timeline changes, and would therefore say nothing about the walk -- which is
        // the only thing this gate exists to measure.
        core::u32 deedSignature = kFnv1aOffsetBasis;
        for (core::u32 i = 0u; i < chronicle.size(); ++i)
        {
            const history::Event &event = chronicle.at(i);
            if (event.attestation.cause != history::Cause::Emergent)
                continue;
            if (result.firstArrival == 0u)
                result.firstArrival = event.fact.object;
            const core::u32 words[] = {event.fact.subject, event.fact.predicate, event.fact.object,
                                       static_cast<core::u32>(event.fact.fromDay)};
            for (const core::u32 word : words)
                deedSignature = history::foldWord(deedSignature, word);
        }
        result.deedSignature = deedSignature;

        core::u32 positionSignature = kFnv1aOffsetBasis;
        for (const auto &partition : registry.partitions())
        {
            if (partition == nullptr || !partition->archetype().has(ecs::ComponentId::Historical))
                continue;
            for (const auto &chunkPtr : partition->chunks())
            {
                if (chunkPtr == nullptr)
                    continue;
                const auto *positions = static_cast<const math::Vec3<math::Fixed32> *>(
                    chunkPtr->writeComponent(ecs::ComponentId::Position));
                const auto *bodies =
                    static_cast<const ecs::HistoricalBody *>(chunkPtr->writeComponent(ecs::ComponentId::Historical));
                if (positions == nullptr || bodies == nullptr)
                    continue;
                for (core::u32 row = 0u; row < chunkPtr->count(); ++row)
                {
                    positionSignature = history::foldWord(positionSignature, bodies[row].subject);
                    positionSignature =
                        history::foldWord(positionSignature, static_cast<core::u32>(positions[row].x.raw()));
                    positionSignature =
                        history::foldWord(positionSignature, static_cast<core::u32>(positions[row].z.raw()));
                }
            }
        }
        result.positionSignature = positionSignature;

        const history::Divergence verdict = history::measureDivergence(chronicle, timeline);
        result.scoredClaims = verdict.scoredClaims;
        result.earned = verdict.earned;
        result.divergenceScore = static_cast<core::u32>(verdict.score.raw());
    };

    // The consensus world: an empty `admittedSources` admits every source.
    history::WorldView consensus;
    runWorld(consensus, math::Fixed32{}, out);

    // @warning And the world according to the register alone. Same corpus, same seed, same systems --
    // one source excluded. If this folded identically, "possible worlds" would be a label rather
    // than a mechanism, which is exactly what the P13 minority signature already asserts about a
    // timeline and what this asserts about a WALK.
    JourneyFoldResult alternate{};
    history::WorldView restricted;
    restricted.admittedSources.push_back(kSourceRegister);
    runWorld(restricted, math::Fixed32{}, alternate);
    out.alternateChronicle = alternate.chronicleSignature;
    out.alternateArrivals = alternate.arrivals;
    out.alternateFirst = alternate.firstArrival;

    // @warning And the SAME world, closed on itself. One number changes -- how wide the world is
    // east-west -- and every distance the walk measures changes with it, because a place beyond the
    // antimeridian is suddenly a neighbour rather than most of a circumference away. A body that
    // ignored the wrap would fold identically here, which is precisely the failure `wrapWidth`
    // exists to prevent: it would set off around the planet the long way, and that road is a
    // perfectly ordinary road.
    //
    // @warning **120 units, and the number was MEASURED against this fixture rather than chosen.** The
    // places sit 300 apart, so a wrap of 250, 350, 400, 500 or 620 leaves every distance exactly as
    // it was and folds identically -- passing for the wrong reason, since a wrap wider than the
    // world it closes changes nothing. Only a world narrow enough that the far places really do sit
    // past the seam moves anything, and 120 is that.
    JourneyFoldResult closed{};
    history::WorldView open;
    runWorld(open, math::Fixed32::fromFloat(120.0f), closed);
    out.closedChronicle = closed.chronicleSignature;
    out.closedArrivals = closed.arrivals;
    out.closedFirst = closed.firstArrival;

    foldAttestedRoads(out);
}

} // namespace lpl::engine::systems
