#ifndef LPL_MATH_MORTON_INL
#define LPL_MATH_MORTON_INL

namespace lpl::math::morton {

namespace detail {

LPL_HD constexpr core::u32 part1by1(core::u32 n)
{
    core::u32 v = n & 0x0000FFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

LPL_HD constexpr core::u64 part1by2(core::u64 n)
{
    core::u64 v = n & 0x1FFFFF;
    v = (v | (v << 32)) & 0x1F00000000FFFF;
    v = (v | (v << 16)) & 0x1F0000FF0000FF;
    v = (v | (v << 8)) & 0x100F00F00F00F00F;
    v = (v | (v << 4)) & 0x10C30C30C30C30C3;
    v = (v | (v << 2)) & 0x1249249249249249;
    return v;
}

LPL_HD constexpr core::u32 compact1by1(core::u32 v)
{
    v &= 0x55555555;
    v = (v | (v >> 1)) & 0x33333333;
    v = (v | (v >> 2)) & 0x0F0F0F0F;
    v = (v | (v >> 4)) & 0x00FF00FF;
    v = (v | (v >> 8)) & 0x0000FFFF;
    return v;
}

LPL_HD constexpr core::u64 compact1by2(core::u64 v)
{
    v &= 0x1249249249249249;
    v = (v | (v >> 2)) & 0x10C30C30C30C30C3;
    v = (v | (v >> 4)) & 0x100F00F00F00F00F;
    v = (v | (v >> 8)) & 0x1F0000FF0000FF;
    v = (v | (v >> 16)) & 0x1F00000000FFFF;
    v = (v | (v >> 32)) & 0x1FFFFF;
    return v;
}

} // namespace detail

LPL_HD constexpr core::u32 encode2D(core::i32 x, core::i32 y)
{
    auto ux = static_cast<core::u32>(x + core::kMortonBias);
    auto uy = static_cast<core::u32>(y + core::kMortonBias);
    return detail::part1by1(ux) | (detail::part1by1(uy) << 1);
}

LPL_HD constexpr core::u64 encode3D(core::i32 x, core::i32 y, core::i32 z)
{
    auto ux = static_cast<core::u64>(x + core::kMortonBias);
    auto uy = static_cast<core::u64>(y + core::kMortonBias);
    auto uz = static_cast<core::u64>(z + core::kMortonBias);
    return detail::part1by2(ux) | (detail::part1by2(uy) << 1) | (detail::part1by2(uz) << 2);
}

LPL_HD constexpr core::u64 encode3DGrid(core::u32 x, core::u32 y, core::u32 z)
{
    return detail::part1by2(static_cast<core::u64>(x)) | (detail::part1by2(static_cast<core::u64>(y)) << 1) |
           (detail::part1by2(static_cast<core::u64>(z)) << 2);
}

LPL_HD constexpr void decode2D(core::u32 code, core::i32 &x, core::i32 &y)
{
    x = static_cast<core::i32>(detail::compact1by1(code)) - core::kMortonBias;
    y = static_cast<core::i32>(detail::compact1by1(code >> 1)) - core::kMortonBias;
}

LPL_HD constexpr void decode3D(core::u64 code, core::i32 &x, core::i32 &y, core::i32 &z)
{
    x = static_cast<core::i32>(detail::compact1by2(code)) - core::kMortonBias;
    y = static_cast<core::i32>(detail::compact1by2(code >> 1)) - core::kMortonBias;
    z = static_cast<core::i32>(detail::compact1by2(code >> 2)) - core::kMortonBias;
}

} // namespace lpl::math::morton

#endif // LPL_MATH_MORTON_INL
