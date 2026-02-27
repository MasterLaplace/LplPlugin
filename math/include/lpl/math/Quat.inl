/**
 * @file Quat.inl
 * @brief Inline implementation of quaternion operations.
 * @see   Quat.hpp
 */

#ifndef LPL_MATH_QUAT_INL
    #define LPL_MATH_QUAT_INL

namespace lpl::math {

template <core::Arithmetic T>
constexpr Quat<T>::Quat(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}

template <core::Arithmetic T>
constexpr Quat<T> Quat<T>::operator*(Quat rhs) const
{
    return {
        w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
        w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
        w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
        w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w
    };
}

template <core::Arithmetic T>
constexpr Vec3<T> Quat<T>::rotate(Vec3<T> v) const
{
    Vec3<T> qVec{x, y, z};
    Vec3<T> t = qVec.cross(v) * (T{2});
    return v + t * w + qVec.cross(t);
}

template <core::Arithmetic T>
constexpr Quat<T> Quat<T>::conjugate() const { return {w, -x, -y, -z}; }

template <core::Arithmetic T>
constexpr Quat<T> Quat<T>::inverse() const
{
    T lenSq = lengthSquared();
    Quat c  = conjugate();
    return {c.w / lenSq, c.x / lenSq, c.y / lenSq, c.z / lenSq};
}

template <core::Arithmetic T>
constexpr T Quat<T>::dot(Quat rhs) const
{
    return w * rhs.w + x * rhs.x + y * rhs.y + z * rhs.z;
}

template <core::Arithmetic T>
constexpr T Quat<T>::lengthSquared() const { return dot(*this); }

template <core::Arithmetic T>
constexpr Quat<T> Quat<T>::normalize() const
{
    if constexpr (std::is_floating_point_v<T>) {
        T inv = T(1) / std::sqrt(lengthSquared());
        return {w * inv, x * inv, y * inv, z * inv};
    } else {
        return *this;
    }
}

template <core::Arithmetic T>
constexpr Quat<T> Quat<T>::identity() { return {T{1}, T{}, T{}, T{}}; }

template <core::Arithmetic T>
constexpr Quat<T> Quat<T>::fromAxisAngle([[maybe_unused]] Vec3<T> axis, [[maybe_unused]] T angleRad)
{
    return identity();
}

template <core::Arithmetic T>
constexpr Quat<T> Quat<T>::slerp([[maybe_unused]] Quat a, [[maybe_unused]] Quat b, [[maybe_unused]] T t)
{
    return identity();
}

template <core::Arithmetic T>
constexpr core::u64 Quat<T>::hash() const
{
    return 0;
}

} // namespace lpl::math

#endif // LPL_MATH_QUAT_INL
