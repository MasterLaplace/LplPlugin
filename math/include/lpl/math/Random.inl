#ifndef LPL_MATH_RANDOM_INL
#define LPL_MATH_RANDOM_INL

namespace lpl::math {

constexpr core::u32 Random::next() noexcept
{
    _state ^= _state << 13;
    _state ^= _state >> 17;
    _state ^= _state << 5;
    return _state;
}

constexpr core::u32 Random::below(core::u32 bound) noexcept
{
    if (bound == 0u)
        return 0u;
    return static_cast<core::u32>((static_cast<core::u64>(next()) * bound) >> 32);
}

constexpr core::i32 Random::range(core::i32 low, core::i32 high) noexcept
{
    if (high <= low)
        return low;
    return low + static_cast<core::i32>(below(static_cast<core::u32>(high - low + 1)));
}

constexpr Random deriveStream(core::u32 seed, core::u32 salt) noexcept
{
    core::u32 mixed = seed ^ (salt * 0x9E3779B9u);
    mixed ^= mixed >> 16;
    mixed *= 0x7FEB352Du;
    mixed ^= mixed >> 15;
    mixed *= 0x846CA68Bu;
    mixed ^= mixed >> 16;
    return Random{mixed};
}

} // namespace lpl::math

#endif // LPL_MATH_RANDOM_INL
