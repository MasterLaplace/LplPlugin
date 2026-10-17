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
#    include <lpl/engine/ITerrainQuery.hpp>
#    include <lpl/engine/systems/GroundStep.hpp>
#    include <lpl/history/Chronicle.hpp>
#    include <lpl/history/Era.hpp>
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

    math::Fixed32 arrivalRadius{math::Fixed32::fromFloat(8.0f)}; /**< How close counts as arrived. */

    /** Places further than this are not considered. Keeps a walk local rather than global. */
    math::Fixed32 horizon{math::Fixed32::fromFloat(4000.0f)};

    core::u32 maxVisits{
        16u}; /**< Most places one body will visit. Bounds the visited set, which is per-body storage. */
};

/**
 * @class JourneySystem
 * @brief Walks named bodies between places, and records their arrivals.
 *
 * @warning Registered in the `PrePhysics` phase and reading Position read-write: it moves bodies, so it
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
     * Places are dated in YEARS and stay that way: a gazetteer's attestation window really
     * is year-resolution, so converting it to days would invent precision Pleiades does not
     * have. The distance covered this tick is the pace scaled by the fraction of a year a
     * tick covers, since a pace is per year and a tick covers a known number of DAYS.
     *
     * @warning Scaled by days rather than by "ticks per year", which is the same number only
     * while an era's rate divides a year evenly. It is the gearing that must not reach the
     * biography: a traveller has to cover the same ground per YEAR whether the era is crossed
     * a day at a time or a century at a time, or the same corpus would put him in Egypt at
     * different dates depending on how fast the run was configured.
     *
     * @warning Positions are read and written on the WRITE side, as everything else in this
     * repository reads: `swapBuffers` copies back to front and then swaps, so the write
     * buffer already holds this frame's values and a read-modify-write on it is both current
     * and published at the next swap.
     *
     * @warning The dead do not travel. A body whose source dates its death before this year
     * must stop producing deeds, or the run would earn agreements for a man who was not there
     * to earn them.
     *
     * @warning The held goal first, and only choose a new one when there is none. A body that
     * re-chooses every tick walks back the way it came the instant it has taken one step --
     * measured: the canonical traveller ping-ponged between two cells for the whole run, with
     * `walkers=1` every tick and `arrivals=0`, because the place it had just left stopped
     * being "where it stands" and became the nearest unvisited one.
     *
     * @warning The road is planned ONCE, when the goal is chosen, and then walked. Asking
     * every tick would re-plan a continental route sixty times a second, and a planner that
     * runs that often is one nobody can afford to make good.
     *
     * @warning The body steers at the next WAYPOINT, not at the destination -- that is the
     * whole difference between following a road and walking past one. Arrival is still judged
     * against the destination, so a road that ends short does not count as having got there.
     * One waypoint per tick at most, so a road is walked, not skipped.
     *
     * @warning An arrival frees the body to choose again, and the place it reached becomes the
     * one whose attested links are offered next. Without this the corpus is consulted once,
     * from wherever the body was seeded, and every leg after the first falls back to geometry
     * -- which reads as a walk that ignores the roads it is standing on.
     *
     * The deed. @warning `history::Cause::Emergent` because nothing told this body to come
     * here: the timeline seeded it somewhere and let go, and the geography did the rest. It is
     * dated to the day, not the year: the run knows exactly when it got there, and a scored
     * claim dated to a whole year still matches because the intervals overlap. Recording the
     * year instead would throw away the one thing the simulation is better at than its
     * sources. No source asserts it; the run did it.
     *
     * Otherwise the body steps toward the goal, normalised by the larger axis rather than by a
     * true length: no square root, and the direction is exact in fixed point.
     *
     * @warning Through the shared stepper when there is ground to consult, so a traveller
     * turns aside from a cliff by exactly the rule a herd does -- one implementation, one
     * answer to "may a body go there". Without terrain the body steps straight, which is what
     * it did before and is still right for a fixture with no world.
     *
     * @warning Re-aim at the target only when the way there is clear. Aiming every tick
     * regardless is what makes greedy avoidance useless: it cancels the turn the last tick
     * made, and the body zigzags into the obstacle instead of past it. When the direct way is
     * blocked the carried heading is kept, so the detour survives long enough to become one.
     *
     * @warning Divide BEFORE multiplying, and this is the third place the same overflow bit.
     * `dx * step` is 300 x 30 at world scale, and Q16.16 saturates at 32767 -- so the step
     * came out as noise and the body stood still while reporting that it was walking.
     * Normalising first bounds the left operand to one, after which the product cannot leave
     * the format. Any pairwise product of two world-scale quantities in Q16.16 is a bug
     * waiting for a big enough map.
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
     * The place also counts as visited: you have been where you were born.
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
        core::u32 goal{0u}; /**< history::Place being walked to, or zero when none is held. */
        core::u32 from{0u}; /**< history::Place last stood at, whose attested links are offered next. */
        /**
         * Waypoints of the road being walked, and how far along it the body is.
         *
         * @warning Bounded, and a route longer than this is walked as far as it reaches and then
         * finished straight -- which is a shorter road, not a wrong destination. Truncating
         * toward the goal is the only truncation that cannot strand a body.
         */
        history::RouteLeg legs[12]{};
        core::u32 legCount{0u};
        core::u32 legAt{0u};
        /**
         * Facing carried between ticks.
         *
         * @warning Without it, avoidance cannot work at all. A body that recomputes its heading from
         * the destination every tick undoes the turn the previous tick made to get round
         * something, so it zigzags into the obstacle face and never passes -- measured, with
         * `arrivals` at zero against a boulder twenty units wide. Creatures have carried a
         * `Heading` component for this reason since the herd systems existed; the journey had
         * none because nothing had ever refused it a step.
         */
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
     * places equidistant from a body must send it to the same one on both targets, and array order
     * is a fact about the caller rather than about the world.
     *
     * @warning **Attested links first, geometry only as a fallback.** A corpus that says two places
     * were connected is stating a fact about roads, sea lanes and passes that no distance measure
     * can recover -- the nearest place across a mountain range is not the place anybody actually
     * went to next. Only when the corpus is silent does the walk fall back to proximity, which is
     * deterministic fill rather than evidence.
     *
     * @warning You cannot travel to where you already are. Without this a body seeded AT a place
     * immediately "arrives" there -- distance zero is inside any arrival radius -- and records a
     * deed nobody performed. Measured: the canonical fixture emitted a `travelled-to` for the
     * traveller's own birthplace on his first tick, which then counted toward a divergence score.
     * A tautology that flatters the run is worse than a missing event.
     *
     * @param from    Where the body is.
     * @param year    The current year.
     * @param visited What it has already seen.
     * @param out     Receives the goal.
     * @return false when nothing is reachable.
     */
    [[nodiscard]] bool chooseGoal(math::Fixed32 x, math::Fixed32 z, core::i32 year, const Visited &visited,
                                  history::Place &out) const;

    /**
     * @brief Finds or makes a body's visit record.
     *
     * @warning Bounded, and the last slot is REUSED rather than the write being dropped: a body
     * whose memory silently vanished would revisit one place for ever and emit the same arrival
     * on every pass, which reads as a very busy traveller rather than as an overflow.
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

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_JOURNEY_HPP
