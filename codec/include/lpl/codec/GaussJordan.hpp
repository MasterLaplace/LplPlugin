/**
 * @file GaussJordan.hpp
 * @brief Reduced row echelon form over GF(2).
 *
 * One entry point, two kernels behind it: a vectorised path on the host and a
 * scalar path on i686. The result must be identical, so the vector path may only
 * reorder XORs — an operation that is associative and commutative, hence safe.
 * Word skipping exploits the fact that columns left of the pivot are already zero.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_GAUSSJORDAN_HPP
#    define LPL_LPL_CODEC_GAUSSJORDAN_HPP

#    include <lpl/codec/BitMatrix.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::codec {

/** @brief A column that no pivot occupies. */
inline constexpr core::u32 kNoPivot = 0xFFFFFFFFu;

/**
 * @struct EliminationResult
 * @brief What the elimination found.
 */
struct EliminationResult {
    core::u32 rank{0u};                             /**< Independent rows. */
    lpl::pmr::vector<core::u32> pivotColumnOfRow{}; /**< Pivot column per row, @ref kNoPivot when none. */
    lpl::pmr::vector<core::u32> rowOfPivotColumn{}; /**< Row holding each column's pivot, or @ref kNoPivot. */

    /**
     * @brief Tells whether the system determines every unknown.
     *
     * @return Whether every column has a pivot, i.e. the system determines every unknown.
     */
    [[nodiscard]] bool fullColumnRank(core::u32 columns) const noexcept { return rank == columns; }
};

/**
 * @brief Reduces @p matrix to reduced row echelon form in place.
 *
 * Augmented columns are supported by simply making the matrix wider than the system:
 * pass @p systemColumns smaller than `matrix.columns()` and the columns past it are
 * carried along by the row operations without ever being chosen as a pivot. That is
 * how the decoder solves for the payload — the right-hand side is not a separate
 * array to keep in step, it is the tail of every row.
 *
 * A column no row at or below the pivot row has set is free: it gets no pivot, and its
 * unknown stays undetermined. A pivot column is cleared everywhere else, in both
 * directions, because this is REDUCED echelon form: leaving the rows above alone would
 * give plain echelon form and force a second back-substitution pass over the same memory,
 * which is the pass this loop is already paying for.
 *
 * Word skipping (SIM-096): every column strictly left of the pivot's is already zero in
 * the pivot row, so the XOR that clears a row starts at the pivot's word.
 *
 * @param matrix        System to reduce; modified.
 * @param systemColumns Columns eligible to hold a pivot. 0 means all of them.
 * @return The rank and the pivot mapping.
 */
[[nodiscard]] EliminationResult gaussJordan(BitMatrix &matrix, core::u32 systemColumns = 0u);

/**
 * @brief Does the reduced system have a solution at all?
 *
 * A row with no pivot is all zeroes over the unknowns. If its augmented tail is not
 * also zero, the system states 0 = 1 and there is nothing to solve. Reporting that is
 * the point: a decoder that returned a payload here would be handing back one it
 * invented, which for an archival format is strictly worse than admitting the loss.
 *
 * @param reduced       A matrix @ref gaussJordan has already reduced.
 * @param result        What that call returned.
 * @param systemColumns Unknowns; the columns past this are the right-hand side.
 * @return true when every pivot-free row has a zero tail.
 */
[[nodiscard]] bool isConsistent(const BitMatrix &reduced, const EliminationResult &result, core::u32 systemColumns);

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_GAUSSJORDAN_HPP
