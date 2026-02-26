/**
 * @file Vec3.inl
 * @brief Inline implementation of Vec3 operations.
 * @see   Vec3.hpp
 */

#ifndef LPL_MATH_VEC3_INL
    #define LPL_MATH_VEC3_INL

#include <cmath>

namespace lpl::math {

template <core::Arithmetic T>
constexpr Vec3<T>::Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

template <core::Arithmetic T>
LPL_HD constexpr Vec3<T> Vec3<T>::operator+(Vec3 rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }

template <core::Arithmetic T>
LPL_HD constexpr Vec3<T> Vec3<T>::operator-(Vec3 rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }

template <core::Arithmetic T>
LPL_HD constexpr Vec3<T> Vec3<T>::operator*(T s) const { return {x * s, y * s, z * s}; }

template <core::Arithmetic T>
LPL_HD constexpr Vec3<T> Vec3<T>::operator/(T s) const { return {x / s, y / s, z / s}; }

template <core::Arithmetic T>
LPL_HD constexpr Vec3<T> Vec3<T>::operator-() const { return {-x, -y, -z}; }

template <core::Arithmetic T>
constexpr Vec3<T> &Vec3<T>::operator+=(Vec3 rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }

template <core::Arithmetic T>
constexpr Vec3<T> &Vec3<T>::operator-=(Vec3 rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }

template <core::Arithmetic T>
constexpr Vec3<T> &Vec3<T>::operator*=(T s) { x *= s; y *= s; z *= s; return *this; }

template <core::Arithmetic T>
LPL_HD constexpr T Vec3<T>::dot(Vec3 rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }

template <core::Arithmetic T>
LPL_HD constexpr Vec3<T> Vec3<T>::cross(Vec3 rhs) const
{
    return {
        y * rhs.z - z * rhs.y,
        z * rhs.x - x * rhs.z,
        x * rhs.y - y * rhs.x
    };
}

template <core::Arithmetic T>
LPL_HD constexpr T Vec3<T>::lengthSquared() const { return dot(*this); }

template <core::Arithmetic T>
constexpr Vec3<T> Vec3<T>::normalize() const
{
    T lenSq = lengthSquared();
    if constexpr (std::is_floating_point_v<T>) {
        T inv = T(1) / std::sqrt(lenSq);
        return *this * inv;
    } else {
        (void)lenSq;
        return *this;
    }
}

template <core::Arithmetic T>
constexpr Vec3<T> Vec3<T>::zero()  { return {T{}, T{}, T{}}; }

template <core::Arithmetic T>
constexpr Vec3<T> Vec3<T>::unitX() { return {T{1}, T{}, T{}}; }

template <core::Arithmetic T>
constexpr Vec3<T> Vec3<T>::unitY() { return {T{}, T{1}, T{}}; }

template <core::Arithmetic T>
constexpr Vec3<T> Vec3<T>::unitZ() { return {T{}, T{}, T{1}}; }

} // namespace lpl::math

#endif // LPL_MATH_VEC3_INL
