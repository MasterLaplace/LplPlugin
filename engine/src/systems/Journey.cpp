#include <lpl/engine/systems/Journey.hpp>

#include <lpl/engine/systems/GroundStep.hpp>

#include <lpl/ecs/Component.hpp>
#include <lpl/math/Geo.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::engine::systems {

namespace {

constexpr ecs::ComponentAccess kComponents[] = {
    {ecs::ComponentId::Position,   ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Historical, ecs::AccessMode::ReadOnly },
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
 * @warning **On a closed world the east-west separation is the SHORTER way round, and without
 * this a closed world is worse than an open one.** Two places either side of the antimeridian
 * are neighbours; a plain subtraction makes them nearly a circumference apart, so "walk to the
 * nearest place" picks the wrong one and a body sets off around the planet the long way -- which
 * looks exactly like a body on a long journey rather than like a defect. The rule itself lives
 * in math::foldOntoShorterWay, shared with the cell-space caller, so the distance that CHOOSES a
 * destination and the one that walks to it cannot drift apart.
 *
 * @param ax First x.
 * @param az First z.
 * @param bx Second x.
 * @param bz Second z.
 * @return The distance.
 */
[[nodiscard]] math::Fixed32 planarDistance(math::Fixed32 ax, math::Fixed32 az, math::Fixed32 bx, math::Fixed32 bz,
                                           math::Fixed32 wrapWidth) noexcept
{
    math::Fixed32 dx = math::foldOntoShorterWay(wrapWidth, bx - ax);
    math::Fixed32 dz = bz - az;
    if (dx < math::Fixed32{})
        dx = math::Fixed32{} - dx;
    if (dz < math::Fixed32{})
        dz = math::Fixed32{} - dz;
    return dx > dz ? dx : dz;
}

} // namespace

JourneySystem::JourneySystem(ecs::Registry &registry, const history::IPlaceResolver &resolver, const core::u32 *places,
                             core::u32 placeCount, const history::Era &era, history::Chronicle &chronicle,
                             const JourneyParams &params) noexcept
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
        visited.places[visited.count++] = place;
}

bool JourneySystem::chooseGoal(math::Fixed32 x, math::Fixed32 z, core::i32 year, const Visited &visited,
                               history::Place &out) const
{
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
        if (d <= _params.arrivalRadius)
            continue;
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
    const core::i32 year = history::yearOfDay(today);
    ++_tick;

    const core::u32 daysPerTick = _era.daysPerTick != 0u ? _era.daysPerTick : 1u;
    const math::Fixed32 step = (_params.pacePerYear * math::Fixed32::fromInt(static_cast<core::i32>(daysPerTick))) /
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
            auto *positions =
                static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position));
            const auto *bodies =
                static_cast<const ecs::HistoricalBody *>(chunk.latestComponent(ecs::ComponentId::Historical));
            if (positions == nullptr || bodies == nullptr)
                continue;

            for (core::u32 row = 0u; row < chunk.count(); ++row)
            {
                const ecs::HistoricalBody &body = bodies[row];
                if (body.diedDay != 0 && today > body.diedDay)
                    continue;
                if (body.bornDay != 0 && today < body.bornDay)
                    continue;

                ++_walkers;
                Visited &visited = visitsOf(body.subject);

                history::Place goal{};
                const bool haveGoal = visited.goal != 0u && _resolver->resolve(visited.goal, goal) && goal.located &&
                                      history::existsInYear(goal, year);
                if (!haveGoal)
                {
                    if (!chooseGoal(positions[row].x, positions[row].z, year, visited, goal))
                    {
                        visited.goal = 0u;
                        continue;
                    }
                    visited.goal = goal.id;
                    visited.legCount = 0u;
                    visited.legAt = 0u;
                    if (_routes != nullptr && visited.from != 0u)
                        visited.legCount = _routes->route(visited.from, goal.id, visited.legs, 12u);
                }

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
                        continue;
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
                    visited.goal = 0u;
                    visited.from = goal.id;
                    visited.legCount = 0u;
                    visited.legAt = 0u;

                    history::Fact arrival{};
                    arrival.subject = body.subject;
                    arrival.predicate = static_cast<core::u32>(history::Predicate::TravelledTo);
                    arrival.object = goal.id;
                    arrival.fromDay = today;
                    arrival.toDay = today;
                    arrival.source = 0u;
                    arrival.sigma = math::Fixed32::one();

                    history::Attestation attestation;
                    attestation.cause = history::Cause::Emergent;
                    attestation.agent = body.subject;
                    _chronicle->record(arrival, attestation);
                    ++_arrivals;
                    continue;
                }

                const math::Fixed32 dx = aimX - positions[row].x;
                const math::Fixed32 dz = aimZ - positions[row].z;
                const math::Fixed32 ax = dx < math::Fixed32{} ? math::Fixed32{} - dx : dx;
                const math::Fixed32 az = dz < math::Fixed32{} ? math::Fixed32{} - dz : dz;
                const math::Fixed32 scale = ax > az ? ax : az;
                if (scale <= math::Fixed32{})
                    continue;

                if (_terrain != nullptr)
                {
                    const math::Fixed32 wantX = dx / scale;
                    const math::Fixed32 wantZ = dz / scale;

                    const math::Fixed32 look = _params.arrivalRadius;
                    const bool clear =
                        _terrain->standable(positions[row].x + wantX * look, positions[row].z + wantZ * look);
                    if (clear || (visited.headingX == math::Fixed32{} && visited.headingZ == math::Fixed32{}))
                    {
                        visited.headingX = wantX;
                        visited.headingZ = wantZ;
                    }

                    const GroundStepResult stepped =
                        stepOnGround(*_terrain, positions[row], visited.headingX, visited.headingZ, step, look);
                    if (stepped.avoided)
                        ++_avoided;
                    continue;
                }
                positions[row].x = positions[row].x + (dx / scale) * step;
                positions[row].z = positions[row].z + (dz / scale) * step;
            }
        }
    }
}

} // namespace lpl::engine::systems
