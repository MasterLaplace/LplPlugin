#ifndef LPL_CONTAINER_BIT_SET_INL
#define LPL_CONTAINER_BIT_SET_INL

namespace lpl::container {

template <core::u32 N> constexpr void BitSet<N>::clearAll()
{
    for (auto &w : _words)
        w = 0;
}

template <core::u32 N> constexpr core::u32 BitSet<N>::count() const
{
    core::u32 total = 0;
    for (auto w : _words)
        total += static_cast<core::u32>(std::popcount(w));
    return total;
}

template <core::u32 N> template <typename Fn> constexpr void BitSet<N>::forEachSet(Fn &&fn) const
{
    for (core::u32 w = 0; w < kWordCount; ++w)
    {
        auto word = _words[w];
        while (word)
        {
            core::u32 bit = static_cast<core::u32>(std::countr_zero(word));
            fn(w * kBitsPerWord + bit);
            word &= word - 1;
        }
    }
}

} // namespace lpl::container

#endif // LPL_CONTAINER_BIT_SET_INL
