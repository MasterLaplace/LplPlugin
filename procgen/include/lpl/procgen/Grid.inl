#ifndef LPL_PROCGEN_GRID_INL
#define LPL_PROCGEN_GRID_INL

namespace lpl::procgen {

template <typename T> inline const T &Grid<T>::clamped(core::i32 x, core::i32 z) const noexcept
{
    const core::u32 cx = x < 0 ? 0u : (static_cast<core::u32>(x) >= _width ? _width - 1u : static_cast<core::u32>(x));
    const core::u32 cz = z < 0 ? 0u : (static_cast<core::u32>(z) >= _depth ? _depth - 1u : static_cast<core::u32>(z));
    return _cells[index(cx, cz)];
}

template <typename T> inline void Grid<T>::fill(T value) noexcept
{
    for (core::u32 i = 0u; i < cellCount(); ++i)
        _cells[i] = value;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_GRID_INL
