/**
 * @file BitSet.hpp
 * @brief Compact fixed-size bitset with fast iteration over set bits.
 *
 * Backed by an array of u64 words.  Provides set/clear/test in O(1)
 * and a forEachSet() iterator that skips zero words entirely.
 *
 * @tparam N Maximum number of bits.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CONTAINER_BIT_SET_HPP
    #define LPL_CONTAINER_BIT_SET_HPP

    #include <lpl/core/Types.hpp>

    #include <array>
    #include <bit>

namespace lpl::container {

/**
 * @brief Fixed-capacity bitset with word-level skip iteration.
 * @tparam N Maximum number of bits.
 */
template <core::u32 N>
class BitSet final {
public:
    static constexpr core::u32 kBitsPerWord = 64;
    static constexpr core::u32 kWordCount   = (N + kBitsPerWord - 1) / kBitsPerWord;

    constexpr BitSet() = default;

    constexpr void set(core::u32 index)
    {
        _words[index / kBitsPerWord] |= (core::u64{1} << (index % kBitsPerWord));
    }

    constexpr void clear(core::u32 index)
    {
        _words[index / kBitsPerWord] &= ~(core::u64{1} << (index % kBitsPerWord));
    }

    [[nodiscard]] constexpr bool test(core::u32 index) const
    {
        return (_words[index / kBitsPerWord] >> (index % kBitsPerWord)) & 1;
    }

    constexpr void clearAll()
    {
        for (auto &w : _words)
            w = 0;
    }

    [[nodiscard]] constexpr core::u32 count() const
    {
        core::u32 total = 0;
        for (auto w : _words)
            total += static_cast<core::u32>(std::popcount(w));
        return total;
    }

    /**
     * @brief Iterate over all set bit indices.
     * @tparam Fn Callable(core::u32 bitIndex).
     * @param fn  Visitor.
     */
    template <typename Fn>
    constexpr void forEachSet(Fn &&fn) const
    {
        for (core::u32 w = 0; w < kWordCount; ++w) {
            auto word = _words[w];
            while (word) {
                core::u32 bit = static_cast<core::u32>(std::countr_zero(word));
                fn(w * kBitsPerWord + bit);
                word &= word - 1;
            }
        }
    }

private:
    std::array<core::u64, kWordCount> _words{};
};

} // namespace lpl::container

#endif // LPL_CONTAINER_BIT_SET_HPP
