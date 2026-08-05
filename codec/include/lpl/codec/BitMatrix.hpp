/**
 * @file BitMatrix.hpp
 * @brief A bit-packed matrix over GF(2).
 *
 * Rows are arrays of 64-bit words aligned to 64 bytes so a vector load never
 * straddles a cache line. Pivoting swaps POINTERS, never row contents, which
 * keeps the operation O(1) and the cache warm.
 *
 * Bit order inside a word is least-significant-first: column c of a row lives at bit
 * `c % 64` of word `c / 64`. Stated because it is observable — @ref fold walks the
 * words, so the signature the gate compares depends on it, and a reader tempted to
 * "fix" the order would be changing the format.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_BITMATRIX_HPP
#    define LPL_LPL_CODEC_BITMATRIX_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::codec {

/// Bits in one storage word. The whole module is written in these units.
inline constexpr core::u32 kBitsPerWord = 64u;

/// Bytes a row is aligned to: one cache line on every target this runs on.
inline constexpr core::u32 kRowAlignment = 64u;

/**
 * @class BitMatrix
 * @brief Rows of packed bits over GF(2), with O(1) pivoting.
 */
class BitMatrix {
public:
    BitMatrix() noexcept = default;

    /**
     * @brief Allocates a zeroed @p rows by @p columns matrix.
     *
     * The row stride is rounded up to the alignment, so every row starts on a cache
     * line and the padding words past the last column stay zero for the lifetime of
     * the matrix. That is not tidiness: the elimination XORs whole rows, so a padding
     * word that ever became non-zero would leak into every row it touched and the
     * matrix would stop meaning what it says.
     */
    BitMatrix(core::u32 rows, core::u32 columns);

    [[nodiscard]] core::u32 rows() const noexcept { return _rows; }
    [[nodiscard]] core::u32 columns() const noexcept { return _columns; }

    /**
     * @brief Words a single row occupies, padding included.
     * @return Words per row, rounded up to the alignment.
     */
    [[nodiscard]] core::u32 rowWords() const noexcept { return _rowWords; }

    /**
     * @brief Returns a pointer to the mutable row @p row, or nullptr when out of range.
     * @param row Row index.
     * @return Pointer to the mutable row, or nullptr when out of range.
     */
    [[nodiscard]] core::u64 *row(core::u32 row) noexcept;

    /**
     * @brief Returns a pointer to the immutable row @p row, or nullptr when out of range.
     * @param row Row index.
     * @return Pointer to the immutable row, or nullptr when out of range.
     */
    [[nodiscard]] const core::u64 *row(core::u32 row) const noexcept;

    [[nodiscard]] bool test(core::u32 row, core::u32 column) const noexcept;
    void set(core::u32 row, core::u32 column) noexcept;
    void clear(core::u32 row, core::u32 column) noexcept;
    void flip(core::u32 row, core::u32 column) noexcept;

    /**
     * @brief Exchanges two rows in O(1).
     *
     * Swaps entries in the indirection table, never the bits (SIM-099). On a matrix
     * whose rows are kilobytes this is the difference between a pivot costing a
     * pointer write and a pivot costing a memcpy of the whole row — and elimination
     * pivots once per column.
     *
     * @param a Row index to swap.
     * @param b Row index to swap.
     */
    void swapRows(core::u32 a, core::u32 b) noexcept;

    /**
     * @brief Adds the source row to the destination row.
     * @param destination Row to add to.
     * @param source      Row to add.
     */
    void addRow(core::u32 destination, core::u32 source) noexcept;

    /**
     * @brief Returns the column of the first set bit at or after @p fromColumn in @p row, or
     *        @ref columns() when the rest of the row is zero.
     * @param row         Row index.
     * @param fromColumn  Column to start searching from.
     * @return Column of the first set bit, or @ref columns() if none found.
     */
    [[nodiscard]] core::u32 firstSetColumn(core::u32 row, core::u32 fromColumn) const noexcept;

    /**
     * @brief Returns the number of set bits in @p row.
     * @param row Row index.
     * @return Set bits in @p row.
     */
    [[nodiscard]] core::u32 rowWeight(core::u32 row) const noexcept;

    /**
     * @brief Zeroes every bit, keeping the allocation.
     */
    void reset() noexcept;

    /**
     * @brief FNV-1a over every row, in row order, words included.
     *
     * Folds the padding too, deliberately: the padding is asserted to be zero, so
     * folding it makes a leak into it a signature change rather than an invisible
     * one. The seed is the offset basis the whole project uses.
     */
    [[nodiscard]] core::u32 fold(core::u32 seed) const noexcept;

private:
    [[nodiscard]] core::u32 wordIndex(core::u32 column) const noexcept { return column / kBitsPerWord; }
    [[nodiscard]] static core::u64 bitMask(core::u32 column) noexcept
    {
        return core::u64{1} << (column % kBitsPerWord);
    }

    lpl::pmr::vector<core::u64> _storage{};       ///< Backing words, over-allocated for alignment.
    lpl::pmr::vector<core::u64 *> _rowPointers{}; ///< Indirection: pivoting rewrites this, not the bits.
    core::u32 _rows{0u};
    core::u32 _columns{0u};
    core::u32 _rowWords{0u};
};

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_BITMATRIX_HPP
