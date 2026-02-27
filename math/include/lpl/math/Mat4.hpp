/**
 * @file Mat4.hpp
 * @brief 4x4 column-major matrix for transforms and projections.
 *
 * @tparam T Scalar type satisfying lpl::core::Arithmetic.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_MAT4_HPP
    #define LPL_MATH_MAT4_HPP

    #include "Vec3.hpp"
    #include "Quat.hpp"

    #include <array>
    #include <cmath>
    #include <type_traits>

namespace lpl::math {

template <core::Arithmetic T>
struct Mat4 final {
    std::array<T, 16> m{};

    constexpr Mat4() = default;

    [[nodiscard]] constexpr T &operator()(core::u32 row, core::u32 col);
    [[nodiscard]] constexpr T  operator()(core::u32 row, core::u32 col) const;

    [[nodiscard]] constexpr Mat4 operator*(Mat4 rhs) const;
    [[nodiscard]] constexpr Vec3<T> transformPoint(Vec3<T> p) const;
    [[nodiscard]] constexpr Vec3<T> transformDirection(Vec3<T> d) const;

    static constexpr Mat4 identity();
    static constexpr Mat4 translate(Vec3<T> offset);
    static constexpr Mat4 scale(Vec3<T> s);
    static constexpr Mat4 fromQuat(Quat<T> q);
    static Mat4 perspective(T fovRad, T aspect, T near, T far);
    static Mat4 lookAt(Vec3<T> eye, Vec3<T> target, Vec3<T> up);
};

using Mat4f = Mat4<float>;

} // namespace lpl::math

#include "Mat4.inl"

#endif // LPL_MATH_MAT4_HPP
