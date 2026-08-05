/**
 * @file Divergence.cpp
 * @brief Scoring a reconstruction, with the self-fulfilment rule.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/Divergence.hpp>

namespace lpl::history {

namespace {

/**
 * @brief Do an event and a claim assert the same thing about the same window?
 * @param event The run's event.
 * @param claim The record's claim.
 * @return true when they agree.
 */
[[nodiscard]] bool agrees(const Fact &event, const Fact &claim) noexcept
{
    return event.subject == claim.subject && event.predicate == claim.predicate && event.object == claim.object &&
           event.fromYear <= claim.toYear && claim.fromYear <= event.toYear;
}

} // namespace

Divergence measureDivergence(const Chronicle &chronicle, const Timeline &timeline) noexcept
{
    Divergence out{};

    for (core::u32 c = 0u; c < timeline.size(); ++c)
    {
        const Constraint &claim = timeline.at(c);
        if (claim.kind != ConstraintKind::Score)
            continue; // seeded and forced claims were given to the run, not asked of it
        ++out.scoredClaims;

        bool earned = false;
        bool refused = false;
        for (core::u32 e = 0u; e < chronicle.size(); ++e)
        {
            const Event &event = chronicle.at(e);
            if (!agrees(event.fact, claim.fact))
                continue;
            // THE rule. An event the timeline caused cannot be evidence for the
            // timeline. Counted separately rather than ignored, because a run whose
            // matches are all self-fulfilled looks identical to one that reconstructed
            // nothing, and the two deserve different reports.
            if (event.attestation.cause == Cause::Constraint)
            {
                refused = true;
                continue;
            }
            earned = true;
            break;
        }

        if (earned)
            ++out.earned;
        else
            ++out.missed;
        if (refused && !earned)
            ++out.selfFulfilled;
    }

    for (core::u32 e = 0u; e < chronicle.size(); ++e)
    {
        const Event &event = chronicle.at(e);
        if (event.attestation.cause != Cause::Emergent)
            continue;
        bool known = false;
        for (core::u32 c = 0u; c < timeline.size() && !known; ++c)
            known = agrees(event.fact, timeline.at(c).fact);
        if (!known)
            ++out.unattested;
    }

    if (out.scoredClaims != 0u)
        out.score = math::Fixed32::fromInt(static_cast<core::i32>(out.earned)) /
                    math::Fixed32::fromInt(static_cast<core::i32>(out.scoredClaims));
    return out;
}

} // namespace lpl::history
