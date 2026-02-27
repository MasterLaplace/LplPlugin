/**
 * @file FixedPoint.hpp
 * @brief Deterministic fixed-point arithmetic type (Q16.16 default).
 *
 * Replaces IEEE 754 floating-point for all simulation math to guarantee
 * bit-exact reproducibility across compilers, CPUs, and operating systems.
 * This is the mathematical backbone of the deterministic Lockstep / Rollback
 * netcode.
 *
 * The default instantiation Fixed32 uses a 32-bit signed integer with
 * 16 fractional bits, yielding a range of roughly [-32768, +32768) and
 * a precision of ~0.000015.
 *
 * @tparam IntType  Underlying signed integer type (i32 or i64).
 * @tparam FracBits Number of fractional bits.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_FIXED_POINT_HPP
    #define LPL_MATH_FIXED_POINT_HPP

    #include <lpl/core/Platform.hpp>
    #include <lpl/core/Types.hpp>

    #include <compare>
    #include <limits>
    #include <type_traits>

namespace lpl::math {

template <typename IntType, core::u32 FracBits>
    requires std::is_signed_v<IntType>
class FixedPoint final {
public:
    using raw_type = IntType;

    static constexpr core::u32 kFracBits = FracBits;
    static constexpr IntType   kOne      = static_cast<IntType>(1) << FracBits;

    constexpr FixedPoint() = default;

    /// @brief Construct from a raw integer value (direct bit pattern).
    explicit constexpr FixedPoint(IntType rawValue) : _raw{rawValue} {}

    static constexpr FixedPoint fromRaw(IntType raw);
    static constexpr FixedPoint fromInt(IntType integer);
    static constexpr FixedPoint fromFloat(float f);
    static constexpr FixedPoint fromDouble(double d);

    [[nodiscard]] constexpr IntType raw()      const;
    [[nodiscard]] constexpr IntType toInt()    const;
    [[nodiscard]] constexpr float   toFloat()  const;
    [[nodiscard]] constexpr double  toDouble() const;

    constexpr FixedPoint  operator+ (FixedPoint rhs) const;
    constexpr FixedPoint  operator- (FixedPoint rhs) const;
    constexpr FixedPoint  operator* (FixedPoint rhs) const;
    constexpr FixedPoint  operator/ (FixedPoint rhs) const;
    constexpr FixedPoint  operator- ()               const;

    constexpr FixedPoint &operator+=(FixedPoint rhs);
    constexpr FixedPoint &operator-=(FixedPoint rhs);
    constexpr FixedPoint &operator*=(FixedPoint rhs);
    constexpr FixedPoint &operator/=(FixedPoint rhs);

    constexpr auto operator<=>(const FixedPoint& rhs) const = default;
    constexpr bool operator== (const FixedPoint& rhs) const = default;

    static constexpr FixedPoint zero();
    static constexpr FixedPoint one();
    static constexpr FixedPoint half();
    static constexpr FixedPoint pi();
    static constexpr FixedPoint epsilon();
    static constexpr FixedPoint max();
    static constexpr FixedPoint min();

    [[nodiscard]] constexpr FixedPoint abs() const;
    [[nodiscard]] constexpr FixedPoint saturatingAdd(FixedPoint rhs) const;
    [[nodiscard]] constexpr FixedPoint saturatingSub(FixedPoint rhs) const;

private:
    IntType _raw = 0;

    using wide_type = std::conditional_t<
        sizeof(IntType) <= 4,
        core::i64,
        __int128
    >;
};

using Fixed32 = FixedPoint<core::i32, 16>;
using Fixed64 = FixedPoint<core::i64, 32>;

} // namespace lpl::math

    #include "FixedPoint.inl"

#endif // LPL_MATH_FIXED_POINT_HPP
