/**
 * @file Vec3.hpp
 * @brief 3-component vector template for deterministic simulation math.
 *
 * Parameterised on the scalar type so that the same code operates on
 * Fixed32 (simulation) and float (rendering interpolation).
 *
 * @tparam T Scalar type satisfying lpl::core::Arithmetic.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_VEC3_HPP
    #define LPL_MATH_VEC3_HPP

    #include <lpl/core/Concepts.hpp>
    #include <lpl/core/Platform.hpp>

namespace lpl::math {

template <core::Arithmetic T>
struct Vec3 final {
    T x{};
    T y{};
    T z{};

    constexpr Vec3() = default;
    constexpr Vec3(T x, T y, T z);

    [[nodiscard]] LPL_HD constexpr Vec3 operator+(Vec3 rhs) const;
    [[nodiscard]] LPL_HD constexpr Vec3 operator-(Vec3 rhs) const;
    [[nodiscard]] LPL_HD constexpr Vec3 operator*(T scalar)  const;
    [[nodiscard]] LPL_HD constexpr Vec3 operator/(T scalar)  const;
    [[nodiscard]] LPL_HD constexpr Vec3 operator-()          const;

    constexpr Vec3 &operator+=(Vec3 rhs);
    constexpr Vec3 &operator-=(Vec3 rhs);
    constexpr Vec3 &operator*=(T scalar);

    [[nodiscard]] LPL_HD constexpr T    dot(Vec3 rhs)      const;
    [[nodiscard]] LPL_HD constexpr Vec3 cross(Vec3 rhs)    const;
    [[nodiscard]] LPL_HD constexpr T    lengthSquared()    const;
    [[nodiscard]] constexpr Vec3        normalize()        const;

    static constexpr Vec3 zero();
    static constexpr Vec3 unitX();
    static constexpr Vec3 unitY();
    static constexpr Vec3 unitZ();
};

using Vec3f  = Vec3<float>;
using Vec3d  = Vec3<double>;

} // namespace lpl::math

    #include "Vec3.inl"

#endif // LPL_MATH_VEC3_HPP
