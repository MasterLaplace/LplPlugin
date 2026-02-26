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

// ─── Inline implementations ─────────────────────────────────────────────────

template <core::Arithmetic T>
constexpr T& Mat4<T>::operator()(core::u32 row, core::u32 col)
{
    return m[col * 4 + row];
}

template <core::Arithmetic T>
constexpr T Mat4<T>::operator()(core::u32 row, core::u32 col) const
{
    return m[col * 4 + row];
}

template <core::Arithmetic T>
constexpr Mat4<T> Mat4<T>::operator*(Mat4 rhs) const
{
    Mat4 result;
    for (core::u32 col = 0; col < 4; ++col)
    {
        for (core::u32 row = 0; row < 4; ++row)
        {
            T sum{};
            for (core::u32 k = 0; k < 4; ++k)
            {
                sum = sum + (*this)(row, k) * rhs(k, col);
            }
            result(row, col) = sum;
        }
    }
    return result;
}

template <core::Arithmetic T>
constexpr Vec3<T> Mat4<T>::transformPoint(Vec3<T> p) const
{
    return Vec3<T>(
        (*this)(0, 0) * p.x + (*this)(0, 1) * p.y + (*this)(0, 2) * p.z + (*this)(0, 3),
        (*this)(1, 0) * p.x + (*this)(1, 1) * p.y + (*this)(1, 2) * p.z + (*this)(1, 3),
        (*this)(2, 0) * p.x + (*this)(2, 1) * p.y + (*this)(2, 2) * p.z + (*this)(2, 3));
}

template <core::Arithmetic T>
constexpr Vec3<T> Mat4<T>::transformDirection(Vec3<T> d) const
{
    return Vec3<T>(
        (*this)(0, 0) * d.x + (*this)(0, 1) * d.y + (*this)(0, 2) * d.z,
        (*this)(1, 0) * d.x + (*this)(1, 1) * d.y + (*this)(1, 2) * d.z,
        (*this)(2, 0) * d.x + (*this)(2, 1) * d.y + (*this)(2, 2) * d.z);
}

template <core::Arithmetic T>
constexpr Mat4<T> Mat4<T>::identity()
{
    Mat4 r;
    r.m.fill(T{});
    r(0, 0) = T{1}; r(1, 1) = T{1}; r(2, 2) = T{1}; r(3, 3) = T{1};
    return r;
}

template <core::Arithmetic T>
constexpr Mat4<T> Mat4<T>::translate(Vec3<T> offset)
{
    auto r = identity();
    r(0, 3) = offset.x;
    r(1, 3) = offset.y;
    r(2, 3) = offset.z;
    return r;
}

template <core::Arithmetic T>
constexpr Mat4<T> Mat4<T>::scale(Vec3<T> s)
{
    auto r = identity();
    r(0, 0) = s.x;
    r(1, 1) = s.y;
    r(2, 2) = s.z;
    return r;
}

template <core::Arithmetic T>
constexpr Mat4<T> Mat4<T>::fromQuat(Quat<T> q)
{
    auto r = identity();
    const T xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const T xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const T wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    const T one{1}, two{2};
    (void)one;
    r(0, 0) = one - two * (yy + zz);
    r(0, 1) = two * (xy - wz);
    r(0, 2) = two * (xz + wy);
    r(1, 0) = two * (xy + wz);
    r(1, 1) = one - two * (xx + zz);
    r(1, 2) = two * (yz - wx);
    r(2, 0) = two * (xz - wy);
    r(2, 1) = two * (yz + wx);
    r(2, 2) = one - two * (xx + yy);
    return r;
}

template <core::Arithmetic T>
Mat4<T> Mat4<T>::perspective(T fovRad, T aspect, T nearPlane, T farPlane)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        const T tanHalf = std::tan(fovRad / T(2));
        Mat4 r;
        r.m.fill(T{});
        r(0, 0) = T(1) / (aspect * tanHalf);
        r(1, 1) = T(1) / tanHalf;
        r(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        r(2, 3) = -(T(2) * farPlane * nearPlane) / (farPlane - nearPlane);
        r(3, 2) = -T(1);
        return r;
    }
    else
    {
        return identity();
    }
}

template <core::Arithmetic T>
Mat4<T> Mat4<T>::lookAt(Vec3<T> eye, Vec3<T> target, Vec3<T> up)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        const auto f = (target - eye).normalize();
        const auto s = f.cross(up).normalize();
        const auto u = s.cross(f);

        auto r = identity();
        r(0, 0) = s.x; r(0, 1) = s.y; r(0, 2) = s.z;
        r(1, 0) = u.x; r(1, 1) = u.y; r(1, 2) = u.z;
        r(2, 0) = -f.x; r(2, 1) = -f.y; r(2, 2) = -f.z;
        r(0, 3) = -(s.dot(eye));
        r(1, 3) = -(u.dot(eye));
        r(2, 3) = f.dot(eye);
        return r;
    }
    else
    {
        (void)eye; (void)target; (void)up;
        return identity();
    }
}

} // namespace lpl::math

#endif // LPL_MATH_MAT4_HPP
