/**
 * @file Journey.cpp
 * @brief Implementation of named bodies that walk and record where they got to.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/systems/Journey.hpp>

#include <lpl/engine/systems/GroundStep.hpp>

#include <lpl/ecs/Component.hpp>
#include <lpl/math/Geo.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::engine::systems {

namespace {

constexpr ecs::ComponentAccess kComponents[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Historical, ecs::AccessMode::ReadOnly},
};

constexpr ecs::SystemDescriptor kDescriptor{
    "history.journey",
    ecs::SchedulePhase::PrePhysics,
    kComponents,
    {},
};

/**
 * @brief Distance on the ground plane, as the larger axis.
 *
 * @warning **NOT a squared distance, and that was a real bug rather than a preference.** A first
 * version compared `dx*dx + dz*dz` -- and Q16.16 saturates at +/-32767, so any two places more than
 * about 181 units apart produced a squared distance that overflowed. Measured: two places 300
 * apart came out closer than two places 60 apart, so the walk chose nonsense goals and then never
 * reached them. Real gazetteer spacings are thousands of units; the metric was unusable for
 * everything except a fixture small enough to hide it.
 *
 * Chebyshev -- the larger of the two axes -- needs no multiplication and therefore cannot overflow.
 * It is a different metric from Euclidean, not an approximation of one, and for "which place is
 * nearest to walk to" that difference is smaller than the error in any ancient coordinate.
 *
 * @param ax First x.
 * @param az First z.
 * @param bx Second x.
 * @param bz Second z.
 * @return The distance.
 */
[[nodiscard]] math::Fixed32 planarDistance(math::Fixed32 ax, math::Fixed32 az, math::Fixed32 bx,
                                           math::Fixed32 bz, math::Fixed32 wrapWidth) noexcept
{
    // @warning **On a closed world the east-west separation is the SHORTER way round, and without
    // this a closed world is worse than an open one.** Two places either side of the antimeridian
    // are neighbours; a plain subtraction makes them nearly a circumference apart, so "walk to the
    // nearest place" picks the wrong one and a body sets off around the planet the long way -- which
    // looks exactly like a body on a long journey rather than like a defect. The rule itself lives
    // in math::foldOntoShorterWay, shared with the cell-space caller, so the distance that CHOOSES a
    // destination and the one that walks to it cannot drift apart.
    math::Fixed32 dx = math::foldOntoShorterWay(wrapWidth, bx - ax);
    math::Fixed32 dz = bz - az;
    if (dx < math::Fixed32{})
        dx = math::Fixed32{} - dx;
    if (dz < math::Fixed32{})
        dz = math::Fixed32{} - dz;
    return dx > dz ? dx : dz;
}

} // namespace

JourneySystem::JourneySystem(ecs::Registry &registry, const history::IPlaceResolver &resolver,
                             const core::u32 *places, core::u32 placeCount, const history::Era &era,
                             history::Chronicle &chronicle, const JourneyParams &params) noexcept
    : _registry(&registry), _resolver(&resolver), _places(places), _placeCount(placeCount), _era(era),
      _chronicle(&chronicle), _params(params)
{
}

const ecs::SystemDescriptor &JourneySystem::descriptor() const noexcept { return kDescriptor; }

JourneySystem::Visited &JourneySystem::visitsOf(core::u32 subject)
{
    for (core::u32 i = 0u; i < _bodyCount; ++i)
        if (_visited[i].subject == subject)
            return _visited[i];
    // @warning Bounded, and the last slot is REUSED rather than the write being dropped: a body whose
    // memory silently vanished would revisit one place for ever and emit the same arrival on
    // every pass, which reads as a very busy traveller rather than as an overflow.
    const core::u32 slot = _bodyCount < kMaxBodies ? _bodyCount++ : kMaxBodies - 1u;
    _visited[slot] = Visited{};
    _visited[slot].subject = subject;
    return _visited[slot];
}

void JourneySystem::placeBodyAt(core::u32 subject, core::u32 place)
{
    Visited &visited = visitsOf(subject);
    visited.from = place;
    if (visited.count < 16u)
        visited.places[visited.count++] = place; // you have been where you were born
}

bool JourneySystem::chooseGoal(math::Fixed32 x, math::Fixed32 z, core::i32 year,
                               const Visited &visited, history::Place &out) const
{
    // @warning **Attested links first, geometry only as a fallback.** A corpus that says two places
    // were connected is stating a fact about roads, sea lanes and passes that no distance
    // measure can recover -- the nearest place across a mountain range is not the place anybody
    // actually went to next. Only when the corpus is silent does the walk fall back to
    // proximity, which is deterministic fill rather than evidence.
    if (visited.from != 0u)
    {
        core::u32 neighbours[16];
        const core::u32 count = _resolver->linkedPlaces(visited.from, neighbours, 16u);
        bool found = false;
        history::Place best{};
        math::Fixed32 bestDistance{};
        for (core::u32 i = 0u; i < count; ++i)
        {
            history::Place candidate{};
            if (!_resolver->resolve(neighbours[i], candidate) || !candidate.located)
                continue;
            if (!history::existsInYear(candidate, year))
                continue;
            bool seen = false;
            for (core::u32 k = 0u; k < visited.count; ++k)
                seen = seen || visited.places[k] == candidate.id;
            if (seen)
                continue;
            const math::Fixed32 d = planarDistance(x, z, candidate.x, candidate.z, _params.wrapWidth);
            // Ties by the LOWER identifier, as below: array order is a fact about the caller.
            if (!found || d < bestDistance || (d == bestDistance && candidate.id < best.id))
            {
                found = true;
                bestDistance = d;
                best = candidate;
            }
        }
        if (found)
        {
            out = best;
            return true;
        }
    }

    const math::Fixed32 horizon = _params.horizon;

    bool found = false;
    math::Fixed32 best{};
    history::Place bestPlace{};

    for (core::u32 i = 0u; i < _placeCount; ++i)
    {
        history::Place candidate{};
        if (!_resolver->resolve(_places[i], candidate) || !candidate.located)
            continue;
        if (!history::existsInYear(candidate, year))
            continue;

        bool seen = false;
        for (core::u32 k = 0u; k < visited.count; ++k)
            seen = seen || visited.places[k] == candidate.id;
        if (seen)
            continue;

        const math::Fixed32 d = planarDistance(x, z, candidate.x, candidate.z, _params.wrapWidth);
        if (d > horizon)
            continue;
        // @warning You cannot travel to where you already are. Without this a body seeded AT a place
        // immediately "arrives" there -- distance zero is inside any arrival radius -- and records
        // a deed nobody performed. Measured: the canonical fixture emitted a `travelled-to` for
        // the traveller's own birthplace on his first tick, which then counted toward a
        // divergence score. A tautology that flatters the run is worse than a missing event.
        if (d <= _params.arrivalRadius)
            continue;
        // @warning Ties go to the LOWER identifier, never to whichever came first in the array: two
        // places equidistant from a body must send it the same way on both targets, and array
        // order is a fact about the caller rather than about the world.
        if (!found || d < best || (d == best && candidate.id < bestPlace.id))
        {
            found = true;
            best = d;
            bestPlace = candidate;
        }
    }
    out = bestPlace;
    return found;
}

void JourneySystem::execute(core::f32 dt)
{
    (void) dt;
    _walkers = 0u;
    if (_registry == nullptr || _resolver == nullptr || _chronicle == nullptr)
        return;

    const core::i32 today = _era.dayOfTick(_tick);
    // Places are dated in YEARS and stay that way: a gazetteer's attestation window really is
    // year-resolution, so converting it to days would invent precision Pleiades does not have.
    const core::i32 year = history::yearOfDay(today);
    ++_tick;

    // Distance covered this tick: a pace is per year, and a tick now covers a known number of
    // DAYS, so the step is the pace scaled by that fraction of a year.
    //
    // @warning Scaled by days rather than by "ticks per year", which is the same number only while
    // an era's rate divides a year evenly. It is the gearing that must not reach the biography:
    // a traveller has to cover the same ground per YEAR whether the era is crossed a day at a
    // time or a century at a time, or the same corpus would put him in Egypt at different dates
    // depending on how fast the run was configured.
    const core::u32 daysPerTick = _era.daysPerTick != 0u ? _era.daysPerTick : 1u;
    const math::Fixed32 step =
        (_params.pacePerYear * math::Fixed32::fromInt(static_cast<core::i32>(daysPerTick))) /
        math::Fixed32::fromInt(365);


    for (const auto &partition : _registry->partitions())
    {
        if (partition == nullptr)
            continue;
        const ecs::Archetype &archetype = partition->archetype();
        if (!archetype.has(ecs::ComponentId::Position) || !archetype.has(ecs::ComponentId::Historical))
            continue;

        for (const auto &chunkPtr : partition->chunks())
        {
            if (chunkPtr == nullptr)
                continue;
            ecs::Chunk &chunk = *chunkPtr;
            // @warning The WRITE side, as everything else in this repository reads: `swapBuffers` copies
            // back to front and then swaps, so the write buffer already holds this frame's values
            // and a read-modify-write on it is both current and published at the next swap.
            auto *positions = static_cast<math::Vec3<math::Fixed32> *>(
                chunk.writeComponent(ecs::ComponentId::Position));
            const auto *bodies =
                static_cast<const ecs::HistoricalBody *>(chunk.writeComponent(ecs::ComponentId::Historical));
            if (positions == nullptr || bodies == nullptr)
                continue;

            for (core::u32 row = 0u; row < chunk.count(); ++row)
            {
                const ecs::HistoricalBody &body = bodies[row];
                // @warning The dead do not travel. A body whose source dates its death before this year
                // must stop producing deeds, or the run would earn agreements for a man who was
                // not there to earn them.
                if (body.diedDay != 0 && today > body.diedDay)
                    continue;
                if (body.bornDay != 0 && today < body.bornDay)
                    continue;

                ++_walkers;
                Visited &visited = visitsOf(body.subject);

                history::Place goal{};
                // @warning The held goal first, and only choose a new one when there is none. A body
                // that re-chooses every tick walks back the way it came the instant it has taken
                // one step -- measured: the canonical traveller ping-ponged between two cells for
                // the whole run, with `walkers=1` every tick and `arrivals=0`, because the place
                // it had just left stopped being "where it stands" and became the nearest
                // unvisited one.
                const bool haveGoal = visited.goal != 0u && _resolver->resolve(visited.goal, goal) &&
                                      goal.located && history::existsInYear(goal, year);
                if (!haveGoal)
                {
                    if (!chooseGoal(positions[row].x, positions[row].z, year, visited, goal))
                    {
                        visited.goal = 0u;
                        continue;
                    }
                    visited.goal = goal.id;
                    // @warning The road is planned ONCE, when the goal is chosen, and then walked. Asking
                    // every tick would re-plan a continental route sixty times a second, and a
                    // planner that runs that often is one nobody can afford to make good.
                    visited.legCount = 0u;
                    visited.legAt = 0u;
                    if (_routes != nullptr && visited.from != 0u)
                        visited.legCount = _routes->route(visited.from, goal.id, visited.legs, 12u);
                }

                // @warning The body steers at the next WAYPOINT, not at the destination -- that is the
                // whole difference between following a road and walking past one. Arrival is
                // still judged against the destination, so a road that ends short does not count
                // as having got there.
                math::Fixed32 aimX = goal.x;
                math::Fixed32 aimZ = goal.z;
                if (visited.legAt < visited.legCount)
                {
                    aimX = visited.legs[visited.legAt].x;
                    aimZ = visited.legs[visited.legAt].z;
                    if (planarDistance(positions[row].x, positions[row].z, aimX, aimZ, _params.wrapWidth) <=
                        _params.arrivalRadius)
                    {
                        positions[row].x = aimX;
                        positions[row].z = aimZ;
                        ++visited.legAt;
                        ++_routedLegs;
                        continue; // one waypoint per tick at most, so a road is walked, not skipped
                    }
                }

                const math::Fixed32 remaining =
                    planarDistance(positions[row].x, positions[row].z, goal.x, goal.z, _params.wrapWidth);
                if (remaining <= _params.arrivalRadius)
                {
                    positions[row].x = goal.x;
                    positions[row].z = goal.z;
                    if (visited.count < _params.maxVisits && visited.count < 16u)
                        visited.places[visited.count++] = goal.id;
                    visited.goal = 0u; // arrived: free to choose again
                    // @warning And it becomes the place whose attested links are offered next. Without
                    // this the corpus is consulted once, from wherever the body was seeded, and
                    // every leg after the first falls back to geometry -- which reads as a walk
                    // that ignores the roads it is standing on.
                    visited.from = goal.id;
                    visited.legCount = 0u;
                    visited.legAt = 0u;

                    // The deed. @warning `history::Cause::Emergent` because nothing told this body to come here:
                    // the timeline seeded it somewhere and let go, and the geography did the rest.
                    history::Fact arrival{};
                    arrival.subject = body.subject;
                    arrival.predicate = static_cast<core::u32>(history::Predicate::TravelledTo);
                    arrival.object = goal.id;
                    // The day, not the year: the run knows exactly when it got there, and a
                    // scored claim dated to a whole year still matches because the intervals
                    // overlap. Recording the year instead would throw away the one thing the
                    // simulation is better at than its sources.
                    arrival.fromDay = today;
                    arrival.toDay = today;
                    arrival.source = 0u; // no source asserts it; the run did it
                    arrival.sigma = math::Fixed32::one();

                    history::Attestation attestation;
                    attestation.cause = history::Cause::Emergent;
                    attestation.agent = body.subject;
                    _chronicle->record(arrival, attestation);
                    ++_arrivals;
                    continue;
                }

                // Step toward the goal. Normalised by the larger axis rather than by a true
                // length: no square root, and the direction is exact in fixed point.
                const math::Fixed32 dx = aimX - positions[row].x;
                const math::Fixed32 dz = aimZ - positions[row].z;
                const math::Fixed32 ax = dx < math::Fixed32{} ? math::Fixed32{} - dx : dx;
                const math::Fixed32 az = dz < math::Fixed32{} ? math::Fixed32{} - dz : dz;
                const math::Fixed32 scale = ax > az ? ax : az;
                if (scale <= math::Fixed32{})
                    continue;

                // @warning Through the shared stepper when there is ground to consult, so a traveller
                // turns aside from a cliff by exactly the rule a herd does -- one implementation,
                // one answer to "may a body go there". Without terrain the body steps straight,
                // which is what it did before and is still right for a fixture with no world.
                if (_terrain != nullptr)
                {
                    const math::Fixed32 wantX = dx / scale;
                    const math::Fixed32 wantZ = dz / scale;

                    // @warning Re-aim at the target only when the way there is clear. Aiming every tick
                    // regardless is what makes greedy avoidance useless: it cancels the turn the
                    // last tick made, and the body zigzags into the obstacle instead of past it.
                    // When the direct way is blocked the carried heading is kept, so the detour
                    // survives long enough to become one.
                    const math::Fixed32 look = _params.arrivalRadius;
                    const bool clear = _terrain->standable(positions[row].x + wantX * look,
                                                           positions[row].z + wantZ * look);
                    if (clear || (visited.headingX == math::Fixed32{} && visited.headingZ == math::Fixed32{}))
                    {
                        visited.headingX = wantX;
                        visited.headingZ = wantZ;
                    }

                    const GroundStepResult stepped =
                        stepOnGround(*_terrain, positions[row], visited.headingX, visited.headingZ,
                                     step, look);
                    if (stepped.avoided)
                        ++_avoided;
                    continue;
                }
                // @warning Divide BEFORE multiplying, and this is the third place the same overflow
                // bit. `dx * step` is 300 x 30 at world scale, and Q16.16 saturates at 32767 --
                // so the step came out as noise and the body stood still while reporting that it
                // was walking. Normalising first bounds the left operand to one, after which the
                // product cannot leave the format. Any pairwise product of two world-scale
                // quantities in Q16.16 is a bug waiting for a big enough map.
                positions[row].x = positions[row].x + (dx / scale) * step;
                positions[row].z = positions[row].z + (dz / scale) * step;
            }
        }
    }
}

} // namespace lpl::engine::systems
