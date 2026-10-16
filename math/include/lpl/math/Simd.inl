#ifndef LPL_MATH_SIMD_INL
#define LPL_MATH_SIMD_INL

namespace lpl::math::simd {

#if !defined(LPL_ARCH_X64)

inline SimdFixed4 SimdFixed4::load(const core::i32 *raw)
{
    return {
        {raw[0], raw[1], raw[2], raw[3]}
    };
}

inline SimdFixed4 SimdFixed4::splat(core::i32 raw)
{
    return {
        {raw, raw, raw, raw}
    };
}

inline void SimdFixed4::store(core::i32 *raw) const
{
    raw[0] = data[0];
    raw[1] = data[1];
    raw[2] = data[2];
    raw[3] = data[3];
}

inline SimdFixed4 SimdFixed4::operator+(SimdFixed4 rhs) const
{
    return {
        {data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2], data[3] + rhs.data[3]}
    };
}

inline SimdFixed4 SimdFixed4::operator-(SimdFixed4 rhs) const
{
    return {
        {data[0] - rhs.data[0], data[1] - rhs.data[1], data[2] - rhs.data[2], data[3] - rhs.data[3]}
    };
}

inline SimdFixed4 SimdFixed4::operator*(SimdFixed4 rhs) const
{
    SimdFixed4 s;
    for (int i = 0; i < 4; ++i)
    {
        core::i64 w = static_cast<core::i64>(data[i]) * static_cast<core::i64>(rhs.data[i]);
        w += (1LL << 15);
        s.data[i] = static_cast<core::i32>(w >> 16);
    }
    return s;
}

#endif

} // namespace lpl::math::simd

#endif // LPL_MATH_SIMD_INL
