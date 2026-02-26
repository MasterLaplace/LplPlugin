/**
 * @file FixedPoint.inl
 * @brief Inline implementation of FixedPoint arithmetic operations.
 * @see   FixedPoint.hpp
 */

#ifndef LPL_MATH_FIXED_POINT_INL
    #define LPL_MATH_FIXED_POINT_INL

namespace lpl::math {

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::fromRaw(I raw)
{
    FixedPoint fp;
    fp._raw = raw;
    return fp;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::fromInt(I integer)
{
    FixedPoint fp;
    fp._raw = integer << F;
    return fp;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::fromFloat(float f)
{
    FixedPoint fp;
    fp._raw = static_cast<I>(f * static_cast<float>(kOne));
    return fp;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::fromDouble(double d)
{
    FixedPoint fp;
    fp._raw = static_cast<I>(d * static_cast<double>(kOne));
    return fp;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr I FixedPoint<I,F>::raw() const { return _raw; }

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr I FixedPoint<I,F>::toInt() const { return _raw >> F; }

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr float FixedPoint<I,F>::toFloat() const
{
    return static_cast<float>(_raw) / static_cast<float>(kOne);
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr double FixedPoint<I,F>::toDouble() const
{
    return static_cast<double>(_raw) / static_cast<double>(kOne);
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::operator+(FixedPoint rhs) const
{
    return fromRaw(_raw + rhs._raw);
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::operator-(FixedPoint rhs) const
{
    return fromRaw(_raw - rhs._raw);
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::operator*(FixedPoint rhs) const
{
    auto wide = static_cast<wide_type>(_raw) * static_cast<wide_type>(rhs._raw);
    wide += (static_cast<wide_type>(1) << (F - 1));
    return fromRaw(static_cast<I>(wide >> F));
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::operator/(FixedPoint rhs) const
{
    auto wide = static_cast<wide_type>(_raw) << F;
    return fromRaw(static_cast<I>(wide / static_cast<wide_type>(rhs._raw)));
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::operator-() const
{
    return fromRaw(-_raw);
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> &FixedPoint<I,F>::operator+=(FixedPoint rhs)
{
    _raw += rhs._raw;
    return *this;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> &FixedPoint<I,F>::operator-=(FixedPoint rhs)
{
    _raw -= rhs._raw;
    return *this;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> &FixedPoint<I,F>::operator*=(FixedPoint rhs)
{
    *this = *this * rhs;
    return *this;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> &FixedPoint<I,F>::operator/=(FixedPoint rhs)
{
    *this = *this / rhs;
    return *this;
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::zero()    { return fromRaw(0); }

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::one()     { return fromRaw(kOne); }

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::half()    { return fromRaw(kOne >> 1); }

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::pi()
{
    return fromRaw(static_cast<I>(3.14159265358979323846 * static_cast<double>(kOne)));
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::epsilon() { return fromRaw(1); }

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::max()
{
    return fromRaw(std::numeric_limits<I>::max());
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::min()
{
    return fromRaw(std::numeric_limits<I>::min());
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::abs() const
{
    return fromRaw(_raw < 0 ? -_raw : _raw);
}

template <typename I, core::u32 F>
    requires std::is_signed_v<I>
constexpr FixedPoint<I,F> FixedPoint<I,F>::saturatingAdd(FixedPoint rhs) const
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
constexpr FixedPoint<I,F> FixedPoint<I,F>::saturatingSub(FixedPoint rhs) const
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
