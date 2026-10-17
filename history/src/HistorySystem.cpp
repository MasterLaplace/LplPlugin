#include <lpl/history/HistorySystem.hpp>

#include <lpl/ecs/ComponentData.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::history {

namespace {

/**
 * @brief The resources a timeline touches when it seeds or forces a fact.
 *
 * Terrain and vegetation, both written: a constraint that places a settlement changes
 * what the ground is, and one that records a famine changes what grows. Declared so
 * the scheduler can order this against whatever else reads them -- which is the whole
 * reason this is a system rather than a call inside a World.
 */
constexpr ecs::ResourceAccess kResources[] = {
    {ecs::ResourceId::Terrain,    ecs::AccessMode::ReadWrite},
    {ecs::ResourceId::Vegetation, ecs::AccessMode::ReadWrite},
};

constexpr ecs::SystemDescriptor kDescriptor{
    "history.timeline",
    ecs::SchedulePhase::PrePhysics,
    {},
    kResources,
};

} // namespace

const ecs::SystemDescriptor &HistorySystem::descriptor() const noexcept { return kDescriptor; }

void HistorySystem::bindWorld(ecs::Registry &registry, const IPlaceResolver &resolver,
                              const ecs::Archetype &archetype) noexcept
{
    _registry = &registry;
    _resolver = &resolver;
    _archetype = archetype;
}

bool HistorySystem::bodyOf(core::u32 subject, ecs::EntityId &out) const noexcept
{
    for (core::u32 i = 0u; i < _bindingCount; ++i)
    {
        if (_bindings[i].subject != subject)
            continue;
        out = _bindings[i].body;
        return true;
    }
    return false;
}

bool HistorySystem::bodyFor(core::u32 subject, const Fact &fact, ecs::EntityId &out, bool &outCreated)
{
    outCreated = false;
    if (bodyOf(subject, out))
        return true;
    if (_registry == nullptr || _bindingCount >= kMaxBodies)
        return false;

    auto created = _registry->createEntity(_archetype);
    if (!created)
        return false;
    out = created.value();

    core::u32 row = 0u;
    if (ecs::Chunk *chunk = _registry->chunkOf(out, row); chunk != nullptr)
    {
        if (auto *bodies = static_cast<ecs::HistoricalBody *>(chunk->writeComponent(ecs::ComponentId::Historical)))
        {
            bodies[row].subject = subject;
            bodies[row].bornDay = fact.fromDay;
            bodies[row].diedDay = 0;
        }
    }

    _bindings[_bindingCount].subject = subject;
    _bindings[_bindingCount].body = out;
    ++_bindingCount;
    ++_seeded;
    outCreated = true;
    return true;
}

void HistorySystem::placeBody(ecs::EntityId body, const Place &place)
{
    if (_registry == nullptr)
        return;
    core::u32 row = 0u;
    ecs::Chunk *chunk = _registry->chunkOf(body, row);
    if (chunk == nullptr)
        return;
    auto *positions = static_cast<math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Position));
    if (positions == nullptr)
        return;
    positions[row].x = place.x;
    positions[row].z = place.z;
}

void HistorySystem::execute(core::f32 dt)
{
    (void) dt;
    if (_timeline == nullptr || _chronicle == nullptr)
        return;

    {
        core::i32 spanFrom = 0;
        core::i32 spanTo = 0;
        _era.spanOfTick(_tick, spanFrom, spanTo);
        core::u32 first = 0u;
        core::u32 count = 0u;
        if (_timeline->constraintsStartingIn(spanFrom, spanTo, first, count))
        {
            for (core::u32 i = 0u; i < count; ++i)
            {
                const Constraint &constraint = _timeline->at(first + i);
                if (constraint.kind == ConstraintKind::Score)
                    continue;

                Attestation attestation;
                attestation.cause = Cause::Constraint;
                attestation.agent = constraint.fact.source;
                _chronicle->record(constraint.fact, attestation);
                ++_applied;

                if (_registry == nullptr || _resolver == nullptr)
                    continue;
                const Predicate predicate = static_cast<Predicate>(constraint.fact.predicate);
                if (!objectIsPlace(predicate))
                    continue;

                Place place{};
                if (!_resolver->resolve(constraint.fact.object, place) || !place.located)
                {
                    ++_unplaceable;
                    continue;
                }

                ecs::EntityId body{};
                bool created = false;
                if (!bodyFor(constraint.fact.subject, constraint.fact, body, created))
                    continue;

                if (constraint.kind == ConstraintKind::Force)
                {
                    placeBody(body, place);
                    ++_forced;
                }
                else if (created)
                {
                    placeBody(body, place);
                }
            }
        }
    }

    ++_tick;
}

} // namespace lpl::history
