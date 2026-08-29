/**
 * @file Journey.hpp
 * @brief Bodies that carry a name, and go places on their own.
 *
 * @warning **Engine-side, and it took an observation from the author to see why.** This started in
 * `history/`, which was wrong in a way that only showed when the walk needed to notice the
 * ground: it wants a `Registry`, it wants terrain, and it wants a step that avoids what it
 * cannot cross -- every one of those is the engine's. What it wants from `history/` is a
 * chronicle and a vocabulary. A system that needs three things from one module and two from
 * another belongs to the first; keeping it in the second meant reaching for a new interface
 * every time it grew, which is how a module acquires seams that exist only to avoid a move.
 *
 * @warning **The half that makes a measurement possible at all.** Until this existed, the only producer
 * of `Cause::Emergent` in the whole tree was the parity fixture -- so @ref Divergence, whose entire
 * honesty rests on counting what a run EARNED, had nothing real to count. A timeline that forces
 * every event agrees with itself by construction; that is an animation of the record, not a
 * reconstruction of it.
 *
 * What happens here instead: a constraint seeds someone somewhere and lets go. The body then
 * chooses where to walk from the geography alone -- the nearest place it has not been, that
 * existed while it was alive -- and **arriving is recorded as its own deed**. Whether that
 * reproduces what a corpus says happened is then a question with an answer, which is the whole
 * point.
 *
 * @warning Movement is Fixed32 throughout. A journey's length decides which year an arrival lands in,
 * so a float would make two targets disagree about when someone reached Egypt.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_JOURNEY_HPP
#    define LPL_ENGINE_SYSTEMS_JOURNEY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/ComponentData.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/history/Chronicle.hpp>
#    include <lpl/history/Era.hpp>
#    include <lpl/engine/ITerrainQuery.hpp>
#    include <lpl/engine/systems/GroundStep.hpp>
#    include <lpl/history/Place.hpp>
#    include <lpl/history/Predicate.hpp>

namespace lpl::engine::systems {

/**
 * @struct JourneyParams
 * @brief How fast, how close, and how far to look.
 */
struct JourneyParams {
    /**
     * Width of the world east-west, or zero for a world with real edges.
     *
     * @warning **A closed world without this is worse than an open one.** Two places either side of
     * the antimeridian are neighbours, and a plain subtraction puts them nearly a circumference
     * apart -- so a body picks the wrong destination and walks the long way round the planet, which
     * reads as a long voyage rather than as a defect. Zero, the default, leaves every existing world
     * exactly as it was.
     *
     * @warning East-west only, and in the same world units a `history::Place` carries. North-south
     * is not folded because a pole crossing reverses the direction of travel AND turns the longitude
     * by half a world, so it is not expressible as a separation a caller could walk along -- see
     * `math::shortestDelta`.
     */
    math::Fixed32 wrapWidth{};

    /**
     * Distance covered per year, in world units.
     *
     * @warning Per YEAR, not per tick. A pace in ticks would make the gearing part of the biography:
     * the same corpus would put someone in Egypt at a different date depending on how fast the
     * era was crossed, which is exactly the mistake `HistorySystem` already refuses to make.
     */
    math::Fixed32 pacePerYear{math::Fixed32::fromFloat(120.0f)};

    /// How close counts as arrived.
    math::Fixed32 arrivalRadius{math::Fixed32::fromFloat(8.0f)};

    /// Places further than this are not considered. Keeps a walk local rather than global.
    math::Fixed32 horizon{math::Fixed32::fromFloat(4000.0f)};

    /// Most places one body will visit. Bounds the visited set, which is per-body storage.
    core::u32 maxVisits{16u};
};

/**
 * @class JourneySystem
 * @brief Walks named bodies between places, and records their arrivals.
 *
 * @warning Registered in the `Simulation` phase and reading Position read-write: it moves bodies, so it
 * must be ordered against anything else that does.
 */
class JourneySystem final : public ecs::ISystem {
public:
    /**
     * @brief Binds what a journey needs.
     *
     * @param registry  Where the bodies are.
     * @param resolver  Where the places are.
     * @param places    Identifiers the walk may choose between.
     * @param placeCount How many.
     * @param era       The gearing between ticks and years.
     * @param chronicle Where arrivals are recorded.
     * @param params    Pace and radii.
     */
    JourneySystem(ecs::Registry &registry, const history::IPlaceResolver &resolver, const core::u32 *places,
                  core::u32 placeCount, const history::Era &era, history::Chronicle &chronicle,
                  const JourneyParams &params) noexcept;

    /**
     * @brief Gives the walk a way to ask how a road runs.
     *
     * Optional: without one, bodies go straight, which is what they did before terrain existed.
     *
     * @param routes The resolver.
     */
    void useRoutes(const history::IRouteResolver &routes) noexcept { _routes = &routes; }

    /**
     * @brief Gives the walk ground that can refuse it.
     *
     * @warning Optional, and without it bodies step straight through rock -- which is what they did
     * until this existed, and was a real defect rather than a simplification: a traveller
     * crossing a mountain range at walking pace makes every date downstream of it wrong.
     *
     * @warning The ground is the ONLY hard constraint. A road is cheap ground, never a corridor: see
     * `GroundStep.hpp` for why fencing one would both invent a wall that never existed and stop
     * a run from being able to depart from the plan it is measured against.
     *
     * @param terrain What may be stood on.
     */
    void useTerrain(const ITerrainQuery &terrain) noexcept { _terrain = &terrain; }

    /**
     * @brief How many times a body had to turn aside from ground it could not cross.
     *
     * @return The count.
     */
    [[nodiscard]] core::u32 avoided() const noexcept { return _avoided; }

    /**
     * @brief How many legs of planned road were walked.
     *
     * @warning Reported because it is what separates a walk that followed a road from one that went
     * straight past it -- and both produce arrivals at the same places.
     *
     * @return The count.
     */
    [[nodiscard]] core::u32 routedLegs() const noexcept { return _routedLegs; }

    /**
     * @brief Advances every named body one tick.
     *
     * @param dt Ignored: an era's clock is ticks, not seconds.
     */
    void execute(core::f32 dt) override;

    /**
     * @brief What this system reads and writes.
     *
     * @return Its descriptor.
     */
    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief How many arrivals were recorded.
     *
     * @return The count.
     */
    [[nodiscard]] core::u32 arrivals() const noexcept { return _arrivals; }

    /**
     * @brief How many bodies were stepped on the last tick.
     *
     * Reported apart from @ref arrivals because "nobody moved" and "everybody moved and nobody
     * got anywhere" are different failures that produce the same arrival count.
     *
     * @return The count.
     */
    [[nodiscard]] core::u32 walkers() const noexcept { return _walkers; }

    /**
     * @brief Tells the walk where a body starts, so the corpus is consulted from the first leg.
     *
     * @warning Without it the first destination is chosen by geometry even when the corpus states a
     * road out of the birthplace -- and the one leg a source is most likely to describe is the
     * one the walk would ignore.
     *
     * @param subject Who.
     * @param place   Where they are, as a gazetteer identifier.
     */
    void placeBodyAt(core::u32 subject, core::u32 place);

private:
    /**
     * One body's memory: where it has been, and where it is currently headed.
     *
     * @warning `goal` is the field that makes a journey a journey. Without it a body re-decided its
     * destination every tick, and since the place it had just left was no longer "where it is
     * standing" the moment it took one step, that place immediately became the nearest unvisited
     * one -- so the traveller oscillated between two cells for ever, walking hard and arriving
     * nowhere. A goal is chosen once and held until it is reached.
     *
     * This is also the cheap half of the hierarchy a real walk wants: decide the destination
     * rarely, move toward it every tick.
     */
    struct Visited {
        core::u32 subject{0u};
        core::u32 goal{0u}; ///< history::Place being walked to, or zero when none is held.
        core::u32 from{0u}; ///< history::Place last stood at, whose attested links are offered next.
        /// Waypoints of the road being walked, and how far along it the body is.
        ///
        /// @warning Bounded, and a route longer than this is walked as far as it reaches and then
        /// finished straight -- which is a shorter road, not a wrong destination. Truncating
        /// toward the goal is the only truncation that cannot strand a body.
        history::RouteLeg legs[12]{};
        core::u32 legCount{0u};
        core::u32 legAt{0u};
        /// Facing carried between ticks.
        ///
        /// @warning Without it, avoidance cannot work at all. A body that recomputes its heading from
        /// the destination every tick undoes the turn the previous tick made to get round
        /// something, so it zigzags into the obstacle face and never passes -- measured, with
        /// `arrivals` at zero against a boulder twenty units wide. Creatures have carried a
        /// `Heading` component for this reason since the herd systems existed; the journey had
        /// none because nothing had ever refused it a step.
        math::Fixed32 headingX{};
        math::Fixed32 headingZ{};
        core::u32 count{0u};
        core::u32 places[16]{};
    };

    /**
     * @brief Chooses where a body goes next.
     *
     * The nearest place inside the horizon that existed in this year and that this body has not
     * already been to. @warning Ties are broken by the LOWER identifier, never by iteration order: two
     * places equidistant from a body must send it to the same one on both targets.
     *
     * @param from    Where the body is.
     * @param year    The current year.
     * @param visited What it has already seen.
     * @param out     Receives the goal.
     * @return false when nothing is reachable.
     */
    [[nodiscard]] bool chooseGoal(math::Fixed32 x, math::Fixed32 z, core::i32 year,
                                  const Visited &visited, history::Place &out) const;

    /**
     * @brief Finds or makes a body's visit record.
     *
     * @param subject Who.
     * @return The record.
     */
    [[nodiscard]] Visited &visitsOf(core::u32 subject);

    ecs::Registry *_registry{nullptr};
    const history::IPlaceResolver *_resolver{nullptr};
    const core::u32 *_places{nullptr};
    core::u32 _placeCount{0u};
    history::Era _era{};
    history::Chronicle *_chronicle{nullptr};
    JourneyParams _params{};
    core::u32 _tick{0u};
    core::u32 _arrivals{0u};
    core::u32 _walkers{0u};
    core::u32 _routedLegs{0u};
    const history::IRouteResolver *_routes{nullptr};
    const ITerrainQuery *_terrain{nullptr};
    core::u32 _avoided{0u};
    static constexpr core::u32 kMaxBodies = 64u;
    Visited _visited[kMaxBodies]{};
    core::u32 _bodyCount{0u};
};

/**
 * @struct JourneyFoldResult
 * @brief What a run with BODIES in it produced. Gate P20.
 *
 * @warning The first fold in this repository whose subject is a body that moved on its own. Every
 * earlier history signature folds a corpus being reasoned about; this one folds where somebody
 * ENDED UP -- and the difference matters because @ref Divergence only credits a run for events it
 * produced rather than was handed.
 */
struct JourneyFoldResult {
    core::u32 chronicleSignature{0u}; ///< Fold of every event, caused and earned alike.
    core::u32 positionSignature{0u};  ///< Fold of where the bodies finished.
    core::u32 deedSignature{0u};      ///< Fold of the emergent arrivals only.
    core::u32 seeded{0u};             ///< Bodies a constraint brought into the world.
    core::u32 forced{0u};             ///< Times a constraint put a body where it says it was.
    core::u32 unplaceable{0u};        ///< Claims naming a place the gazetteer cannot locate.
    core::u32 arrivals{0u};           ///< Deeds the run emitted on its own.
    core::u32 scoredClaims{0u};       ///< Claims held back to be earned.
    core::u32 earned{0u};             ///< Of those, the ones the walk actually reproduced.
    core::u32 divergenceScore{0u};    ///< Raw Q16.16 of the score.

    /**
     * Where the first emergent arrival landed.
     *
     * @warning The number that proves the corpus was consulted at all. The canonical fixture attests a
     * road from the birthplace to the FAR place while a nearer one sits unlinked, so a walk that
     * ignored the corpus would arrive at the near place first -- and every signature above would
     * still be perfectly stable on both targets while measuring a reader that never opened the
     * gazetteer.
     */
    core::u32 firstArrival{0u};

    /**
     * Waypoints of planned road the walk actually followed.
     *
     * @warning Zero would mean the bodies went straight past every road, which produces the same
     * arrivals at the same places -- so without this number the gate cannot tell a walk that used
     * the terrain from one that ignored it.
     */
    core::u32 routedLegs{0u};

    /**
     * Times a body turned aside from ground it could not cross.
     *
     * @warning Zero would mean the walk went through the rock -- which it did, silently, until the
     * terrain was wired in. The fixture puts a BOULDER on the road (not a wall across it: see
     * `RidgeTerrain` for why a wall measured the fixture instead of the code), so a walk that
     * consulted the ground and one that did not reach the same places by different routes, and
     * only this number separates them.
     */
    core::u32 avoided{0u};

    /**
     * Fold of the road network a corpus's attested links produce on a relief.
     *
     * @warning Folded APART from the walk, and the separation is the point. Everything above is about
     * a body: where it went, what it earned. This is about `procgen::routeLeastCost` itself --
     * a priority queue, Fixed32 costs and an index tie-break, none of which had ever crossed to
     * ring 0 because the pass had no caller at all. Mixing the two would mean a change to the
     * router moved a signature named after the traveller.
     */
    core::u32 roadSignature{0u};

    /**
     * Fold of the waypoints that road is summarised as.
     *
     * @warning Separate from @ref roadSignature because the two fail differently: the mask says the
     * search found the same cells on both targets, this says they were reduced to the same
     * corners. A router that agreed and a summariser that did not would produce identical roads
     * that bodies walked differently.
     */
    core::u32 waypointSignature{0u};

    /// Cells the attested network paved.
    core::u32 roadCells{0u};

    /// Attested pairs a road was laid between.
    core::u32 roadPairs{0u};

    /**
     * The same corpus walked again with one source excluded.
     *
     * @warning **What makes "possible worlds" a mechanism rather than a label.** The P13 gate already
     * asserts that a timeline built from a restricted view folds differently; this asserts it of a
     * WALK -- a body seeded somewhere else, going somewhere else, ending somewhere else. Nothing
     * forks: it is one run function called twice with one entry more in `admittedSources`.
     */
    core::u32 alternateChronicle{0u};

    /// Deeds the alternate world's body emitted on its own.
    core::u32 alternateArrivals{0u};

    /// Where the alternate world's first emergent arrival landed.
    core::u32 alternateFirst{0u};

    /**
     * The same world CLOSED on itself, east-west.
     *
     * @warning One number changes -- how wide the world is -- and every distance the walk measures
     * changes with it: a place beyond the antimeridian is a neighbour rather than most of a
     * circumference away. A walk that ignored `JourneyParams::wrapWidth` would fold identically
     * here, and that is the whole failure it exists to prevent -- a body setting off around the
     * planet the long way, on a road nothing downstream can tell was the wrong one.
     */
    core::u32 closedChronicle{0u};

    /// Deeds the closed world's body emitted on its own.
    core::u32 closedArrivals{0u};

    /// Where the closed world's first emergent arrival landed.
    core::u32 closedFirst{0u};

    /**
     * Cells paved when the routing grid CLOSES east-west.
     *
     * @warning The two attested places sit either side of the seam, so a router that could not cross
     * it lays a road most of the way round instead. That road is perfectly valid and perfectly long,
     * and nothing downstream can tell it was the wrong one -- which is why this is folded rather
     * than trusted.
     */
    core::u32 wrappedRoadCells{0u};

    /**
     * Cells paved with the poles open as well.
     *
     * @warning This grid gives a polar crossing nothing to win, so it must equal
     * @ref wrappedRoadCells. Offering a shortcut must never make a road WORSE -- and a road getting
     * worse is exactly what an inadmissible estimate looks like from the outside, since A* then
     * discards the cheap route before evaluating it.
     */
    core::u32 polarRoadCells{0u};

    /**
     * Fold of the SAME attested network, planned coarse and refined fine.
     *
     * @warning **It must equal @ref roadSignature, and that equality is the claim.** A cascade is a
     * different search -- a plan on a summary, a corridor, then a confined A* -- so "it returns the
     * road the flat search returns" is a property to prove rather than assume, and proving it on
     * one machine proves it for one machine. Two numbers that must match is how a target that
     * cascaded differently would name itself.
     */
    core::u32 cascadeRoadSignature{0u};

    /// Cells the cascaded network paved; must equal @ref roadCells.
    core::u32 cascadeRoadCells{0u};

    /**
     * Cells the COARSE plans settled.
     *
     * @warning **Zero would mean no cascade happened**, and a run that quietly fell back to a flat
     * search folds an identical road signature and satisfies every check above. This is the number
     * that says the summary was built, planned on, and refined -- and it is folded rather than
     * merely asserted non-zero because a coarse search is arithmetic too, and arithmetic that
     * disagrees between targets is what a gate exists to catch.
     */
    core::u32 cascadeCoarseExpanded{0u};

    /// Fine cells the coarse plans opened.
    core::u32 cascadeCorridorCells{0u};
};

/**
 * @brief Seeds a body from a corpus, lets it walk, and folds what happened.
 *
 * The canonical journey: one traveller, five places, a birth that is FORCED and an arrival that
 * is only SCORED -- so the run has something to earn rather than something to replay.
 *
 * @warning The places are a fixed table rather than a gazetteer file. A gate that read a corpus from
 * disk would be a gate that fails when the disk changes, and the claim under test is that the
 * same arithmetic gives the same journey on two targets -- not that a CSV parses.
 *
 * @param out Receives the signatures.
 */
void foldJourneyState(JourneyFoldResult &out);

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_JOURNEY_HPP
