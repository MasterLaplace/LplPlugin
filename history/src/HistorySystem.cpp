/**
 * @file HistorySystem.cpp
 * @brief Applying a timeline as the era advances.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/HistorySystem.hpp>

namespace lpl::history {

namespace {

/**
 * @brief The resources a timeline touches when it seeds or forces a fact.
 *
 * Terrain and vegetation, both written: a constraint that places a settlement changes
 * what the ground is, and one that records a famine changes what grows. Declared so
 * the scheduler can order this against whatever else reads them — which is the whole
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

void HistorySystem::execute(core::f32 dt)
{
    (void) dt;
    if (_timeline == nullptr || _chronicle == nullptr)
        return;

    // Constraints land on a year BOUNDARY, once. A fact dated to a year applied on
    // every tick of that year would be applied four times at four ticks per year and
    // once at one, so the same corpus would fold differently depending on how fast the
    // era was crossed — which would make the gearing part of the history.
    if (_era.isYearBoundary(_tick))
    {
        const core::i32 year = _era.yearOfTick(_tick);
        core::u32 first = 0u;
        core::u32 count = 0u;
        if (_timeline->constraintsOfYear(year, first, count))
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
            }
        }
    }

    ++_tick;
}

} // namespace lpl::history
