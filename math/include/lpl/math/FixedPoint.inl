#ifndef LPL_MATH_FIXED_POINT_INL
#define LPL_MATH_FIXED_POINT_INL

namespace lpl::math {

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::fromRaw(I raw)
{
    FixedPoint fp;
    fp._raw = raw;
    return fp;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::fromInt(I integer)
{
    FixedPoint fp;
    fp._raw = integer << F;
    return fp;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::fromFloat(float f)
{
    const float scaled = f * static_cast<float>(kOne);
    if (scaled != scaled)
        return FixedPoint{};
    if (scaled >= static_cast<float>(std::numeric_limits<I>::max()))
        return max();
    if (scaled <= static_cast<float>(std::numeric_limits<I>::min()))
        return min();
    return fromRaw(static_cast<I>(scaled));
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::fromDouble(double d)
{
    const double scaled = d * static_cast<double>(kOne);
    if (scaled != scaled)
        return FixedPoint{};
    if (scaled >= static_cast<double>(std::numeric_limits<I>::max()))
        return max();
    if (scaled <= static_cast<double>(std::numeric_limits<I>::min()))
        return min();
    return fromRaw(static_cast<I>(scaled));
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr I FixedPoint<I, F>::raw() const
{
    return _raw;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr I FixedPoint<I, F>::toInt() const
{
    return _raw >> F;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr float FixedPoint<I, F>::toFloat() const
{
    return static_cast<float>(_raw) / static_cast<float>(kOne);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr double FixedPoint<I, F>::toDouble() const
{
    return static_cast<double>(_raw) / static_cast<double>(kOne);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::operator+(FixedPoint rhs) const
{
    return fromRaw(_raw + rhs._raw);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::operator-(FixedPoint rhs) const
{
    return fromRaw(_raw - rhs._raw);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::operator*(FixedPoint rhs) const
{
    auto wide = static_cast<wide_type>(_raw) * static_cast<wide_type>(rhs._raw);
    wide += (static_cast<wide_type>(1) << (F - 1));
    return fromRaw(static_cast<I>(wide >> F));
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::operator/(FixedPoint rhs) const
{
    constexpr I kNarrowRaw = std::numeric_limits<I>::max() >> F;
    if (_raw >= -kNarrowRaw && _raw <= kNarrowRaw)
        return fromRaw(static_cast<I>(static_cast<I>(_raw << F) / rhs._raw));
    return fromRaw(static_cast<I>((static_cast<wide_type>(_raw) << F) / static_cast<wide_type>(rhs._raw)));
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::operator-() const
{
    return fromRaw(-_raw);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> &FixedPoint<I, F>::operator+=(FixedPoint rhs)
{
    _raw += rhs._raw;
    return *this;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> &FixedPoint<I, F>::operator-=(FixedPoint rhs)
{
    _raw -= rhs._raw;
    return *this;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> &FixedPoint<I, F>::operator*=(FixedPoint rhs)
{
    *this = *this * rhs;
    return *this;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> &FixedPoint<I, F>::operator/=(FixedPoint rhs)
{
    *this = *this / rhs;
    return *this;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::zero()
{
    return fromRaw(0);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::one()
{
    return fromRaw(kOne);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::half()
{
    return fromRaw(kOne >> 1);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::pi()
{
    return fromRaw(static_cast<I>(3.14159265358979323846 * static_cast<double>(kOne)));
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::epsilon()
{
    return fromRaw(1);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::max()
{
    return fromRaw(std::numeric_limits<I>::max());
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::min()
{
    return fromRaw(std::numeric_limits<I>::min());
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::abs() const
{
    return fromRaw(_raw < 0 ? -_raw : _raw);
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::reciprocal() const
{
    if constexpr (2u * F == 8u * sizeof(I))
    {
        if (_raw > 2)
        {
            using unsigned_type = std::make_unsigned_t<I>;
            constexpr unsigned_type kWordMax = std::numeric_limits<unsigned_type>::max();
            const auto divisor = static_cast<unsigned_type>(_raw);
            const unsigned_type below = kWordMax / divisor;
            const unsigned_type carry = kWordMax % divisor == divisor - 1u ? 1u : 0u;
            return fromRaw(static_cast<I>(below + carry));
        }
    }
    return one() / *this;
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::saturatingAdd(FixedPoint rhs) const
{
    auto wide = static_cast<wide_type>(_raw) + static_cast<wide_type>(rhs._raw);
    if (wide > std::numeric_limits<I>::max())
        return max();
    if (wide < std::numeric_limits<I>::min())
        return min();
    return fromRaw(static_cast<I>(wide));
}

template <typename I, core::u32 F>
requires std::is_signed_v<I>
constexpr FixedPoint<I, F> FixedPoint<I, F>::saturatingSub(FixedPoint rhs) const
{
    auto wide = static_cast<wide_type>(_raw) - static_cast<wide_type>(rhs._raw);
    if (wide > std::numeric_limits<I>::max())
        return max();
    if (wide < std::numeric_limits<I>::min())
        return min();
    return fromRaw(static_cast<I>(wide));
}

} // namespace lpl::math

#endif // LPL_MATH_FIXED_POINT_INL
