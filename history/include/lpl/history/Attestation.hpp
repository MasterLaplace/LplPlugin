/**
 * @file Attestation.hpp
 * @brief Provenance of a simulated event.
 *
 * Every event the chronicle emits carries why it happened: a constraint, an
 * emergent system, or a decision of the sovereign. Without this the chronicle is
 * unfalsifiable.
 *
 * Unfalsifiable is the exact word. A chronicle that only says WHAT happened cannot be
 * scored, because every event in it might have been forced by the timeline it is
 * about to be compared against — and a run that reproduces its own inputs has
 * reproduced nothing. The attestation is what makes divergence a measurement.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_ATTESTATION_HPP
#    define LPL_LPL_HISTORY_ATTESTATION_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::history {

/**
 * @enum Cause
 * @brief Why an event is in the chronicle.
 */
enum class Cause : core::u32 {
    /**
     * @brief The timeline put it there.
     *
     * Cannot count towards agreement with that same timeline. This is the whole reason
     * the enum exists.
     */
    Constraint = 0u,

    /**
     * @brief A system produced it.
     *
     * The only kind that is evidence. An event the ecology or the settlement pass
     * produced, that the record also contains, is a prediction the run earned.
     */
    Emergent = 1u,

    /**
     * @brief A director asked for it.
     *
     * Caine, or a human. Recorded rather than hidden so a chronicle stays honest about
     * which parts of it were authored.
     */
    Sovereign = 2u,

    Count = 3u
};

/**
 * @brief The word a report spells @p cause with.
 * @param cause The cause.
 * @return Its name.
 */
[[nodiscard]] constexpr const char *causeName(Cause cause) noexcept
{
    switch (cause)
    {
    case Cause::Constraint: return "constraint";
    case Cause::Emergent: return "emergent";
    case Cause::Sovereign: return "sovereign";
    case Cause::Count: break;
    }
    return "?";
}

/**
 * @struct Attestation
 * @brief Why one event happened, and by what.
 */
struct Attestation {
    Cause cause{Cause::Emergent}; ///< Which of the three.
    core::u32 agent{0u};          ///< Source id, system id, or director id, per @c cause.
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_ATTESTATION_HPP
