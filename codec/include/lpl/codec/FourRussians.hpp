/**
 * @file FourRussians.hpp
 * @brief M4RI: eliminate k columns per pass via a lookup table.
 *
 * The space-time trade that takes GF(2) elimination below the cube: precompute
 * the 2^k linear combinations of k pivot rows, then replace k XORs by one table
 * read. The table is built along a Gray code so consecutive entries differ by a
 * single row, costing exactly one vector XOR each. k is bounded by the L2 cache,
 * not by the theory.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_FOURRUSSIANS_HPP
#    define LPL_LPL_CODEC_FOURRUSSIANS_HPP

#    include <lpl/codec/BitMatrix.hpp>
#    include <lpl/codec/GaussJordan.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::codec {

/**
 * @brief Widest block M4RI will take.
 *
 * @warning k is bounded by the cache, not by the theory (SIM-105). Textbook M4RI takes
 * k = log2(N); in practice the table is 2^k rows of N/8 bytes and the whole gain
 * evaporates the moment it stops fitting in L2. Twelve is the upper end of the
 * useful range; @ref chooseBlockWidth picks the actual value from a byte budget.
 */
inline constexpr core::u32 kMaxBlockWidth = 12u;

/** @brief Table budget assumed for the host's L2, in bytes. Conservative on purpose. */
inline constexpr core::u32 kDefaultTableBudget = 256u * 1024u;

/**
 * @brief Largest k in [1, @ref kMaxBlockWidth] whose table fits @p byteBudget.
 *
 * @param rowBytes    Bytes one row occupies.
 * @param byteBudget  Cache the table may occupy.
 */
[[nodiscard]] core::u32 chooseBlockWidth(core::u32 rowBytes, core::u32 byteBudget = kDefaultTableBudget) noexcept;

/**
 * @class GrayCodeTable
 * @brief The 2^k linear combinations of k rows, built in 2^k XORs.
 *
 * Naively the table costs 2^k * k row-XORs. Along a Gray code two consecutive
 * indices differ in exactly one bit, so each entry is the previous one plus a single
 * row (SIM-103): 2^k XORs total, and k disappears from the construction cost. That
 * is the difference between the table being an optimisation and being a tax.
 */
class GrayCodeTable {
public:
    GrayCodeTable() noexcept = default;

    /**
     * @brief Builds the combinations of @p width rows starting at @p firstRow.
     *
     * Entry 0 is the empty combination and stays zero. Every other entry is the
     * PREVIOUS Gray entry plus one row, which is what keeps the table at 2^k XORs.
     *
     * @param source   Matrix holding the pivot rows.
     * @param firstRow Index of the first pivot row.
     * @param width    Rows to combine; clamped to @ref kMaxBlockWidth.
     */
    void build(const BitMatrix &source, core::u32 firstRow, core::u32 width);

    /**
     * @brief Looks up one combination.
     *
     * @return Combination @p index, or nullptr when the table is empty.
     */
    [[nodiscard]] const core::u64 *combination(core::u32 index) const noexcept;

    [[nodiscard]] core::u32 width() const noexcept { return _width; }
    [[nodiscard]] core::u32 entries() const noexcept { return _entries; }
    [[nodiscard]] core::u32 rowWords() const noexcept { return _rowWords; }

    /**
     * @brief XORs the entries taken, for a deterministic count the test can compare.
     */
    [[nodiscard]] core::u32 xorsPerformed() const noexcept { return _xorsPerformed; }

private:
    lpl::pmr::vector<core::u64> _storage{};
    core::u32 _width{0u};
    core::u32 _entries{0u};
    core::u32 _rowWords{0u};
    core::u32 _xorsPerformed{0u};
};

/**
 * @brief Reduced row echelon form by blocks of k columns.
 *
 * Same contract as @ref gaussJordan and the same result, bit for bit — that is what
 * makes it a refactor rather than a second algorithm, and the parity test asserts it.
 * What changes is that a row is cleared of k columns by one table read and one XOR
 * instead of up to k XORs (SIM-104).
 *
 * Each block runs in two phases.
 *
 * Phase 1 reduces the block among the rows at or below the pivot row, restricted to those
 * rows ON PURPOSE. The pivot search in the plain algorithm only ever looks at rows at or
 * below the pivot row, and those rows see exactly the same bits here as they would there —
 * every pivot column left of this block was cleared from every row by the blocks before it,
 * below by their phase 1 and above by their phase 2. So the two algorithms choose the same
 * pivots, perform the same swaps, and end with the same row permutation. That is what lets
 * the parity test assert they fold identically rather than merely "agree up to a
 * permutation".
 *
 * Phase 2 clears the block from every row ABOVE it, in one read. The rows below were
 * already cleared by phase 1, which had to touch them to find its pivots. What is left is
 * the back-substitution half, and that is where the table pays: one lookup plus one XOR
 * removes all k columns from a row at once (SIM-104), instead of up to k separate row XORs.
 * A row whose index into the table is zero is skipped (SIM-096: test before calling the
 * kernel).
 *
 * @param matrix        System to reduce; modified.
 * @param systemColumns Columns eligible to hold a pivot; 0 means all.
 * @param blockWidth    k, or 0 to let @ref chooseBlockWidth decide.
 */
[[nodiscard]] EliminationResult fourRussiansEliminate(BitMatrix &matrix, core::u32 systemColumns = 0u,
                                                      core::u32 blockWidth = 0u);

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_FOURRUSSIANS_HPP
