/**
 * @file BitMatrix.cpp
 * @brief Implementation of the bit-packed GF(2) matrix.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/BitMatrix.hpp>

#include <lpl/codec/XorKernel.hpp>

namespace lpl::codec {

namespace {

constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Words per row, rounded up so each row starts on a @ref kRowAlignment boundary.
 */
[[nodiscard]] core::u32 alignedRowWords(core::u32 columns) noexcept
{
    const core::u32 wordsPerLine = kRowAlignment / sizeof(core::u64);
    const core::u32 needed = (columns + kBitsPerWord - 1u) / kBitsPerWord;
    const core::u32 rounded = ((needed + wordsPerLine - 1u) / wordsPerLine) * wordsPerLine;
    return rounded == 0u ? wordsPerLine : rounded;
}

} // namespace

BitMatrix::BitMatrix(core::u32 rows, core::u32 columns) : _rows(rows), _columns(columns)
{
    _rowWords = alignedRowWords(columns);
    if (_rows == 0u)
        return;

    // Over-allocate by one line so the base can be walked forward to an aligned
    // address. lpl::pmr has no aligned allocation and the kernel heap makes no
    // alignment promise past a word, so the alignment is established here rather
    // than assumed — an assumption is what makes a vector load straddle a line on
    // exactly one target and nowhere else.
    const core::u32 wordsPerLine = kRowAlignment / sizeof(core::u64);
    _storage.resize(static_cast<core::usize>(_rows) * _rowWords + wordsPerLine, core::u64{0});

    core::u64 *base = _storage.data();
    const core::usize misalignment = reinterpret_cast<core::usize>(base) % kRowAlignment;
    if (misalignment != 0u)
        base += (kRowAlignment - misalignment) / sizeof(core::u64);

    _rowPointers.resize(_rows, nullptr);
    for (core::u32 i = 0u; i < _rows; ++i)
        _rowPointers[i] = base + static_cast<core::usize>(i) * _rowWords;
}

core::u64 *BitMatrix::row(core::u32 row) noexcept
{
    return row < _rows ? _rowPointers[row] : nullptr;
}

const core::u64 *BitMatrix::row(core::u32 row) const noexcept
{
    return row < _rows ? _rowPointers[row] : nullptr;
}

bool BitMatrix::test(core::u32 row, core::u32 column) const noexcept
{
    if (row >= _rows || column >= _columns)
        return false;
    return (_rowPointers[row][wordIndex(column)] & bitMask(column)) != 0u;
}

void BitMatrix::set(core::u32 row, core::u32 column) noexcept
{
    if (row >= _rows || column >= _columns)
        return;
    _rowPointers[row][wordIndex(column)] |= bitMask(column);
}

void BitMatrix::clear(core::u32 row, core::u32 column) noexcept
{
    if (row >= _rows || column >= _columns)
        return;
    _rowPointers[row][wordIndex(column)] &= ~bitMask(column);
}

void BitMatrix::flip(core::u32 row, core::u32 column) noexcept
{
    if (row >= _rows || column >= _columns)
        return;
    _rowPointers[row][wordIndex(column)] ^= bitMask(column);
}

void BitMatrix::swapRows(core::u32 a, core::u32 b) noexcept
{
    if (a >= _rows || b >= _rows || a == b)
        return;
    core::u64 *const held = _rowPointers[a];
    _rowPointers[a] = _rowPointers[b];
    _rowPointers[b] = held;
}

void BitMatrix::addRow(core::u32 destination, core::u32 source) noexcept
{
    if (destination >= _rows || source >= _rows)
        return;
    xorRow(_rowPointers[destination], _rowPointers[source], _rowWords);
}

core::u32 BitMatrix::firstSetColumn(core::u32 row, core::u32 fromColumn) const noexcept
{
    if (row >= _rows || fromColumn >= _columns)
        return _columns;

    const core::u64 *const words = _rowPointers[row];
    core::u32 word = wordIndex(fromColumn);

    // The first word is masked so bits before fromColumn do not answer.
    core::u64 value = words[word] & ~((core::u64{1} << (fromColumn % kBitsPerWord)) - core::u64{1});
    const core::u32 lastWord = wordIndex(_columns - 1u);
    while (true)
    {
        if (value != 0u)
        {
            core::u32 bit = 0u;
            while ((value & core::u64{1}) == 0u)
            {
                value >>= 1;
                ++bit;
            }
            const core::u32 column = word * kBitsPerWord + bit;
            return column < _columns ? column : _columns;
        }
        if (word >= lastWord)
            return _columns;
        ++word;
        value = words[word];
    }
}

core::u32 BitMatrix::rowWeight(core::u32 row) const noexcept
{
    if (row >= _rows)
        return 0u;
    const core::u64 *const words = _rowPointers[row];
    core::u32 weight = 0u;
    for (core::u32 i = 0u; i < _rowWords; ++i)
    {
        core::u64 value = words[i];
        while (value != 0u)
        {
            value &= value - core::u64{1};
            ++weight;
        }
    }
    return weight;
}

void BitMatrix::reset() noexcept
{
    for (core::u32 i = 0u; i < _rows; ++i)
        for (core::u32 w = 0u; w < _rowWords; ++w)
            _rowPointers[i][w] = 0u;
}

core::u32 BitMatrix::fold(core::u32 seed) const noexcept
{
    core::u32 hash = seed;
    for (core::u32 i = 0u; i < _rows; ++i)
    {
        const core::u64 *const words = _rowPointers[i];
        for (core::u32 w = 0u; w < _rowWords; ++w)
        {
            hash = (hash ^ static_cast<core::u32>(words[w] & 0xFFFFFFFFu)) * kFnv1aPrime;
            hash = (hash ^ static_cast<core::u32>(words[w] >> 32)) * kFnv1aPrime;
        }
    }
    return hash;
}

} // namespace lpl::codec
