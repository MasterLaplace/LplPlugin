#ifndef LPL_CODEC_GALOISFIELD_INL
#define LPL_CODEC_GALOISFIELD_INL

namespace lpl::codec {

constexpr core::u8 gf256Mul(core::u8 a, core::u8 b) noexcept
{
    if (a == 0u || b == 0u)
        return 0u;
    return kGf256.exp[static_cast<core::u32>(kGf256.log[a]) + static_cast<core::u32>(kGf256.log[b])];
}

constexpr core::u8 gf256Div(core::u8 a, core::u8 b) noexcept
{
    if (a == 0u || b == 0u)
        return 0u;
    return kGf256.exp[static_cast<core::u32>(kGf256.log[a]) + 255u - static_cast<core::u32>(kGf256.log[b])];
}

constexpr core::u8 gf256Inv(core::u8 a) noexcept
{
    if (a == 0u)
        return 0u;
    return kGf256.exp[255u - static_cast<core::u32>(kGf256.log[a])];
}

constexpr core::u8 gf256PowOf(core::u8 base, core::u32 exponent) noexcept
{
    if (base == 0u)
        return exponent == 0u ? 1u : 0u;
    return kGf256.exp[(static_cast<core::u32>(kGf256.log[base]) * exponent) % 255u];
}

constexpr core::u8 gf256Evaluate(const core::u8 *coefficients, core::u32 count, core::u8 x) noexcept
{
    core::u8 accumulator = 0u;
    for (core::u32 i = count; i > 0u; --i)
        accumulator = gf256Add(gf256Mul(accumulator, x), coefficients[i - 1u]);
    return accumulator;
}

} // namespace lpl::codec

#endif // LPL_CODEC_GALOISFIELD_INL
