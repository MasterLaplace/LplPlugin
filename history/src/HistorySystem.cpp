/**
 * @file HistorySystem.cpp
 * @brief Applying a timeline as the era advances.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

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

    // The body learns WHO it is and WHEN it lived. @warning The dates come from the claim rather than
    // from the tick: a source dates a birth, and deriving it from the clock would make the
    // gearing part of the biography.
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

    // A constraint fires on the tick whose span contains the START of its window, exactly once.
    //
    // @warning Both halves of that sentence are load-bearing, and they used to be enforced by a
    // "year boundary" test that only worked at one gearing:
    //  - ONCE, because a claim applied on every tick its window overlaps would be applied 365
    //    times at a daily gearing and once at a yearly one, so the same corpus would fold
    //    differently depending on how fast the era was crossed -- the gearing would become part
    //    of the history. A window has exactly one start, so keying on the start gives one firing
    //    at any rate, with no per-constraint state to remember.
    //  - NEVER SKIPPED, because consecutive spans are contiguous. The old test asked whether the
    //    current year EQUALLED the fact's year, so an era geared to cross a silent century in a
    //    step would pass over every constraint inside it in silence, and produce a chronicle that
    //    looked complete.
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
                    continue; // a scored claim is measured against, never applied

                Attestation attestation;
                attestation.cause = Cause::Constraint;
                attestation.agent = constraint.fact.source;
                _chronicle->record(constraint.fact, attestation);
                ++_applied;

                // ── And now it ACTS ──────────────────────────────────────────
                // @warning The half that was missing. Recording a constraint and never letting it touch
                // the world made every run an animation of its own record: `Divergence` then
                // measured a timeline against itself, which agrees by construction.
                if (_registry == nullptr || _resolver == nullptr)
                    continue;
                const Predicate predicate = static_cast<Predicate>(constraint.fact.predicate);
                if (!objectIsPlace(predicate))
                    continue; // the gazetteer must only be asked about objects that ARE places

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

                // @warning A Seed positions ONCE and lets go; a Force keeps putting the body back. That
                // is the whole difference between a condition the run may leave behind and a
                // script it cannot depart from -- and a Seed that kept re-placing would be a
                // Force wearing the other name.
                if (constraint.kind == ConstraintKind::Force)
                {
                    placeBody(body, place);
                    ++_forced;
                }
                else if (created)
                {
                    // @warning ONLY on the tick the body is made, and a first version re-placed it at
                    // every year boundary -- which meant a Seed put the traveller back at his
                    // birthplace four times a year for ever. The comment above already said
                    // "positions once and lets go" while the code did the opposite, so the
                    // measurement showed a body that walked every tick and never got anywhere.
                    placeBody(body, place);
                }
            }
        }
    }

    ++_tick;
}

} // namespace lpl::history
