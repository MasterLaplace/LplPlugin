#ifndef LPL_MATH_FIXEDMATH_INL
#define LPL_MATH_FIXEDMATH_INL

namespace lpl::math {

constexpr core::u32 integerSqrt(core::u32 value) noexcept
{
    core::u32 remainder = value;
    core::u32 root = 0u;
    core::u32 bit = core::u32{1} << 30;

    while (bit > remainder)
        bit >>= 2;

    while (bit != 0u)
    {
        if (remainder >= root + bit)
        {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        }
        else
        {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

constexpr Fixed32 fixedLog2(core::u32 value) noexcept
{
    if (value == 0u)
        return Fixed32::zero();

    core::u32 exponent = 0u;
    core::u32 rest = value;
    while (rest > 1u)
    {
        rest >>= 1;
        ++exponent;
    }

    const core::u32 floorPower = core::u32{1} << exponent;
    const core::u32 fraction =
        exponent == 0u ? 0u : static_cast<core::u32>(((static_cast<core::u64>(value - floorPower)) << 16) >> exponent);

    return Fixed32::fromRaw(static_cast<core::i32>((exponent << 16) + fraction));
}

constexpr Fixed32 fixedExp2(Fixed32 value) noexcept
{
    if (value.raw() <= -(31 << 16))
        return Fixed32::zero();
    if (value.raw() >= (15 << 16))
        return Fixed32::max();

    const core::i32 integerPart = value.raw() >> 16;
    const core::u32 fraction = static_cast<core::u32>(value.raw()) & 0xFFFFu;

    constexpr core::i64 c1 = 45426;
    constexpr core::i64 c2 = 15743;
    constexpr core::i64 c3 = 3638;
    constexpr core::i64 c4 = 630;
    constexpr core::i64 c5 = 87;
    constexpr core::i64 c6 = 10;

    const core::i64 f = static_cast<core::i64>(fraction);
    core::i64 series = c6;
    series = ((series * f) >> 16) + c5;
    series = ((series * f) >> 16) + c4;
    series = ((series * f) >> 16) + c3;
    series = ((series * f) >> 16) + c2;
    series = ((series * f) >> 16) + c1;
    series = ((series * f) >> 16) + Fixed32::kOne;

    if (integerPart >= 0)
    {
        const core::i64 scaled = series << integerPart;
        return scaled >= static_cast<core::i64>(Fixed32::max().raw()) ?
                   Fixed32::max() :
                   Fixed32::fromRaw(static_cast<core::i32>(scaled));
    }
    return Fixed32::fromRaw(static_cast<core::i32>(series >> (-integerPart)));
}

constexpr Fixed32 fixedSqrt(Fixed32 value) noexcept
{
    if (value.raw() <= 0)
        return Fixed32::zero();

    core::u64 remainder = static_cast<core::u64>(static_cast<core::u32>(value.raw())) << 16;
    core::u64 root = 0u;
    core::u64 bit = core::u64{1} << 46;

    while (bit > remainder)
        bit >>= 2;

    while (bit != 0u)
    {
        if (remainder >= root + bit)
        {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        }
        else
        {
            root >>= 1;
        }
        bit >>= 2;
    }
    return Fixed32::fromRaw(static_cast<core::i32>(root));
}

} // namespace lpl::math

#endif // LPL_MATH_FIXEDMATH_INL
