/**
 * @file GaussJordan.cpp
 * @brief Reduced row echelon form over GF(2), with word skipping.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/GaussJordan.hpp>

#include <lpl/codec/XorKernel.hpp>

namespace lpl::codec {

EliminationResult gaussJordan(BitMatrix &matrix, core::u32 systemColumns)
{
    EliminationResult result{};
    const core::u32 rows = matrix.rows();
    const core::u32 columns = matrix.columns();
    const core::u32 eligible = (systemColumns == 0u || systemColumns > columns) ? columns : systemColumns;

    result.pivotColumnOfRow.resize(rows, kNoPivot);
    result.rowOfPivotColumn.resize(eligible, kNoPivot);
    if (rows == 0u || columns == 0u)
        return result;

    const core::u32 words = matrix.rowWords();
    core::u32 pivotRow = 0u;

    for (core::u32 column = 0u; column < eligible && pivotRow < rows; ++column)
    {
        // Find a row at or below pivotRow with this column set.
        core::u32 found = rows;
        for (core::u32 r = pivotRow; r < rows; ++r)
        {
            if (matrix.test(r, column))
            {
                found = r;
                break;
            }
        }
        if (found == rows)
            continue; // free column: no pivot, and the unknown stays undetermined

        // O(1): the indirection table moves, the kilobytes do not.
        matrix.swapRows(pivotRow, found);

        // Clear the column everywhere else. Both directions, because this is
        // REDUCED echelon form: leaving the rows above alone would give plain
        // echelon form and force a second back-substitution pass over the same
        // memory, which is the pass this loop is already paying for.
        core::u64 *const pivot = matrix.row(pivotRow);
        for (core::u32 r = 0u; r < rows; ++r)
        {
            if (r == pivotRow || !matrix.test(r, column))
                continue;
            // Word skipping (SIM-096): every column strictly left of this one is
            // already zero in both rows, so the XOR starts at the pivot's word.
            const core::u32 firstWord = column / kBitsPerWord;
            xorRow(matrix.row(r) + firstWord, pivot + firstWord, words - firstWord);
        }

        result.pivotColumnOfRow[pivotRow] = column;
        result.rowOfPivotColumn[column] = pivotRow;
        ++pivotRow;
    }

    result.rank = pivotRow;
    return result;
}

bool isConsistent(const BitMatrix &reduced, const EliminationResult &result, core::u32 systemColumns)
{
    const core::u32 columns = reduced.columns();
    const core::u32 unknowns = (systemColumns == 0u || systemColumns > columns) ? columns : systemColumns;

    for (core::u32 r = 0u; r < reduced.rows(); ++r)
    {
        const core::u32 pivot = r < result.pivotColumnOfRow.size() ? result.pivotColumnOfRow[r] : kNoPivot;
        if (pivot != kNoPivot)
            continue;
        for (core::u32 c = unknowns; c < columns; ++c)
            if (reduced.test(r, c))
                return false;
    }
    return true;
}

} // namespace lpl::codec
