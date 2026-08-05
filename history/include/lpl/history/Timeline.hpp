/**
 * @file Timeline.hpp
 * @brief Dated constraints, ordered, replayable.
 *
 * The ordering is part of the contract exactly as pass order is for a world
 * recipe: it fixes the order in which constraints are applied, therefore the fold.
 *
 * Ordered by year first, then by subject, predicate, object and source — a total
 * order over the contents, not merely a chronological one. Two corpora with the same
 * facts in a different file order must produce the same timeline, or the fold would
 * depend on how a curator happened to type them in.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_TIMELINE_HPP
#    define LPL_LPL_HISTORY_TIMELINE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/Constraint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::history {

/**
 * @class Timeline
 * @brief The constraints a run must honour, in the order it honours them.
 */
class Timeline {
public:
    Timeline() = default;

    /**
     * @brief Adds a constraint. The timeline is re-sorted by @ref finalise.
     * @param constraint What to add.
     */
    void add(const Constraint &constraint);

    /**
     * @brief Puts the constraints in their canonical order.
     *
     * Must be called before a run. An unsorted timeline still contains the same
     * facts and folds differently, which is the class of bug this project spends the
     * most effort refusing.
     */
    void finalise();

    /**
     * @brief Constraints in the timeline.
     * @return The count.
     */
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_constraints.size()); }

    /**
     * @brief One constraint.
     * @param index Position in the canonical order.
     * @return The constraint; a default one when the index is out of range.
     */
    [[nodiscard]] const Constraint &at(core::u32 index) const noexcept;

    /**
     * @brief Constraints dated to a year.
     *
     * The timeline is sorted, so a year is a contiguous run and this is a pair of
     * bounds rather than a scan.
     *
     * @param year      The year.
     * @param outFirst  Receives the first index.
     * @param outCount  Receives how many.
     * @return false when the year carries nothing.
     */
    [[nodiscard]] bool constraintsOfYear(core::i32 year, core::u32 &outFirst, core::u32 &outCount) const noexcept;

    /**
     * @brief FNV-1a over every constraint, in order.
     *
     * The order is folded with the contents precisely because it is part of the
     * contract: a timeline that sorted differently would fold differently and say so.
     *
     * @param seed Fold seed.
     * @return The signature.
     */
    [[nodiscard]] core::u32 fold(core::u32 seed) const noexcept;

    /**
     * @brief Earliest and latest year the timeline mentions.
     * @param outFirst Receives the earliest.
     * @param outLast  Receives the latest.
     * @return false when the timeline is empty.
     */
    [[nodiscard]] bool span(core::i32 &outFirst, core::i32 &outLast) const noexcept;

private:
    lpl::pmr::vector<Constraint> _constraints{};
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_TIMELINE_HPP
