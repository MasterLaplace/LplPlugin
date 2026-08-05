/**
 * @file Divergence.hpp
 * @brief The measured gap between chronicle and timeline.
 *
 * The demon's score, and the only honest answer to 'does it work'. A prediction
 * is worth exactly what the reconstruction of a known past is worth, which is why
 * this number exists before any claim about the future.
 *
 * The measurement has one rule that makes it a measurement rather than a
 * congratulation: **a constrained event cannot count as a match.** The run was told to
 * produce it, so reproducing it demonstrates nothing at all. Only events the systems
 * emitted on their own — @ref Cause::Emergent — can agree with the record, and a
 * chronicle made entirely of constraints scores zero however closely it resembles the
 * timeline it was built from.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_DIVERGENCE_HPP
#    define LPL_LPL_HISTORY_DIVERGENCE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/Chronicle.hpp>
#    include <lpl/history/Timeline.hpp>

namespace lpl::history {

/**
 * @struct Divergence
 * @brief How far the run's account is from the record.
 */
struct Divergence {
    core::u32 scoredClaims{0u};  ///< Claims the run had to earn: kind Score.
    core::u32 earned{0u};        ///< Of those, the ones an emergent event matched.
    core::u32 missed{0u};        ///< Of those, the ones nothing matched.
    core::u32 unattested{0u};    ///< Emergent events the record says nothing about.
    core::u32 selfFulfilled{0u}; ///< Matches refused because the run was told to make them.

    /**
     * @brief Share of the scored claims the run reconstructed, in [0, 1].
     *
     * Zero scored claims yields zero, not one: a timeline that asks nothing of a run
     * has not been reconstructed by it, and returning a perfect score for an empty
     * question is the shape of a verification that cannot fail.
     */
    math::Fixed32 score{};

    /**
     * @brief Is the reconstruction good enough to be worth anything?
     * @param threshold Minimum score.
     * @return true when the score reaches it AND anything was actually asked.
     */
    [[nodiscard]] bool acceptable(math::Fixed32 threshold) const noexcept
    {
        return scoredClaims != 0u && score >= threshold;
    }
};

/**
 * @brief Scores a chronicle against the record it was supposed to reconstruct.
 *
 * @param chronicle What the run says happened.
 * @param timeline  What the record says happened.
 * @return The measurement.
 */
[[nodiscard]] Divergence measureDivergence(const Chronicle &chronicle, const Timeline &timeline) noexcept;

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_DIVERGENCE_HPP
