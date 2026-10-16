#ifndef LPL_MATH_GEO_INL
#define LPL_MATH_GEO_INL

namespace lpl::math {

template <typename T> constexpr T foldOntoShorterWay(T span, T separation) noexcept
{
    if (!(span > T{}))
        return separation;
    T folded = separation;
    for (core::u32 guard = 0u; guard < 64u; ++guard)
    {
        const T doubled = folded + folded;
        if (doubled > span)
            folded = folded - span;
        else if (T{} - doubled > span)
            folded = folded + span;
        else
            break;
    }
    return folded;
}

inline bool ReliefMosaic::add(const ReliefField *tile) noexcept
{
    if (tile == nullptr || !tile->valid() || count >= kMaxResidentReliefTiles)
        return false;
    tiles[count++] = tile;
    return true;
}

} // namespace lpl::math

#endif // LPL_MATH_GEO_INL
