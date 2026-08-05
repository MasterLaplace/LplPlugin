/**
 * @file FourRussians.cpp
 * @brief M4RI elimination and its Gray-code table.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/FourRussians.hpp>

#include <lpl/codec/XorKernel.hpp>

namespace lpl::codec {

namespace {

/**
 * @brief Index of the single set bit of @p value. Only called on exact powers of two.
 */
[[nodiscard]] core::u32 loneBitIndex(core::u32 value) noexcept
{
    core::u32 index = 0u;
    while ((value & 1u) == 0u && index < 31u)
    {
        value >>= 1;
        ++index;
    }
    return index;
}

/**
 * @brief The Gray code of @p i: consecutive values differ in exactly one bit.
 */
[[nodiscard]] constexpr core::u32 grayCode(core::u32 i) noexcept { return i ^ (i >> 1); }

} // namespace

core::u32 chooseBlockWidth(core::u32 rowBytes, core::u32 byteBudget) noexcept
{
    if (rowBytes == 0u)
        return 1u;
    core::u32 width = 1u;
    while (width < kMaxBlockWidth)
    {
        const core::u64 nextCost = (core::u64{1} << (width + 1u)) * rowBytes;
        if (nextCost > byteBudget)
            break;
        ++width;
    }
    return width;
}

void GrayCodeTable::build(const BitMatrix &source, core::u32 firstRow, core::u32 width)
{
    _width = width > kMaxBlockWidth ? kMaxBlockWidth : width;
    _rowWords = source.rowWords();
    _entries = _width == 0u ? 0u : (1u << _width);
    _xorsPerformed = 0u;

    if (_entries == 0u || _rowWords == 0u)
    {
        _storage.clear();
        return;
    }

    _storage.resize(static_cast<core::usize>(_entries) * _rowWords, core::u64{0});

    // Entry 0 is the empty combination and stays zero. Every other entry is the
    // PREVIOUS Gray entry plus one row — 2^k XORs for the whole table instead of
    // 2^k * k (SIM-103). Without this the table costs more than the XORs it saves.
    for (core::u32 i = 1u; i < _entries; ++i)
    {
        const core::u32 current = grayCode(i);
        const core::u32 previous = grayCode(i - 1u);
        const core::u32 changed = loneBitIndex(current ^ previous);

        const core::u64 *const addend = source.row(firstRow + changed);
        core::u64 *const destination = _storage.data() + static_cast<core::usize>(current) * _rowWords;
        const core::u64 *const base = _storage.data() + static_cast<core::usize>(previous) * _rowWords;

        if (addend == nullptr)
            continue;
        xorRowInto(destination, base, addend, _rowWords);
        ++_xorsPerformed;
    }
}

const core::u64 *GrayCodeTable::combination(core::u32 index) const noexcept
{
    if (index >= _entries || _storage.empty())
        return nullptr;
    return _storage.data() + static_cast<core::usize>(index) * _rowWords;
}

EliminationResult fourRussiansEliminate(BitMatrix &matrix, core::u32 systemColumns, core::u32 blockWidth)
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
    const core::u32 rowBytes = words * static_cast<core::u32>(sizeof(core::u64));
    core::u32 k = blockWidth == 0u ? chooseBlockWidth(rowBytes) : blockWidth;
    if (k == 0u)
        k = 1u;
    if (k > kMaxBlockWidth)
        k = kMaxBlockWidth;

    lpl::pmr::vector<core::u32> blockPivotColumn;
    GrayCodeTable table;
    core::u32 pivotRow = 0u;

    for (core::u32 blockStart = 0u; blockStart < eligible && pivotRow < rows; blockStart += k)
    {
        const core::u32 blockEnd = (blockStart + k) < eligible ? (blockStart + k) : eligible;
        const core::u32 blockFirstRow = pivotRow;
        blockPivotColumn.clear();

        // ── Phase 1: reduce the block among the rows at or below pivotRow ────
        //
        // Restricted to those rows ON PURPOSE. The pivot search in the plain
        // algorithm only ever looks at rows >= pivotRow, and those rows see exactly
        // the same bits here as they would there — every column left of this block
        // was cleared from every row by the previous block's phase 2. So the two
        // algorithms choose the same pivots, perform the same swaps, and end with
        // the same row permutation. That is what lets the parity test assert they
        // fold identically rather than merely "agree up to a permutation".
        for (core::u32 column = blockStart; column < blockEnd && pivotRow < rows; ++column)
        {
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
                continue;

            matrix.swapRows(pivotRow, found);

            const core::u64 *const pivot = matrix.row(pivotRow);
            const core::u32 firstWord = column / kBitsPerWord;
            for (core::u32 r = blockFirstRow; r < rows; ++r)
            {
                if (r == pivotRow || !matrix.test(r, column))
                    continue;
                xorRow(matrix.row(r) + firstWord, pivot + firstWord, words - firstWord);
            }

            result.pivotColumnOfRow[pivotRow] = column;
            result.rowOfPivotColumn[column] = pivotRow;
            blockPivotColumn.push_back(column);
            ++pivotRow;
        }

        const core::u32 pivotsInBlock = static_cast<core::u32>(blockPivotColumn.size());
        if (pivotsInBlock == 0u)
            continue;

        // ── Phase 2: clear the block from every row ABOVE it, in one read ────
        //
        // The rows below were already cleared by phase 1, which had to touch them to
        // find its pivots. What is left is the back-substitution half, and that is
        // where the table pays: one lookup plus one XOR removes all k columns from a
        // row at once (SIM-104), instead of up to k separate row XORs.
        table.build(matrix, blockFirstRow, pivotsInBlock);

        const core::u32 firstWord = blockStart / kBitsPerWord;
        for (core::u32 r = 0u; r < blockFirstRow; ++r)
        {
            core::u32 index = 0u;
            for (core::u32 b = 0u; b < pivotsInBlock; ++b)
                if (matrix.test(r, blockPivotColumn[b]))
                    index |= (1u << b);

            if (index == 0u)
                continue; // SIM-096: test before calling the kernel

            const core::u64 *const combination = table.combination(index);
            if (combination == nullptr)
                continue;
            xorRow(matrix.row(r) + firstWord, combination + firstWord, words - firstWord);
        }
    }

    result.rank = pivotRow;
    return result;
}

} // namespace lpl::codec
