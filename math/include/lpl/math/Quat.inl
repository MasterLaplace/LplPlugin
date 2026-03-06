/**
 * @file Quat.inl
 * @brief Inline implementation of quaternion operations.
 * @see   Quat.hpp
 */

#ifndef LPL_MATH_QUAT_INL
#define LPL_MATH_QUAT_INL

#include <cmath>
#include <cstring>

namespace lpl::math {

template <core::Arithmetic T> constexpr Quat<T>::Quat(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}

template <core::Arithmetic T> constexpr Quat<T> Quat<T>::operator*(Quat rhs) const
{
    return {w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z, w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x, w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w};
}

template <core::Arithmetic T> constexpr Vec3<T> Quat<T>::rotate(Vec3<T> v) const
{
    Vec3<T> qVec{x, y, z};
    Vec3<T> t = qVec.cross(v) * (T{2});
    return v + t * w + qVec.cross(t);
}

template <core::Arithmetic T> constexpr Quat<T> Quat<T>::conjugate() const { return {w, -x, -y, -z}; }

template <core::Arithmetic T> constexpr Quat<T> Quat<T>::inverse() const
{
    T lenSq = lengthSquared();
    Quat c = conjugate();
    return {c.w / lenSq, c.x / lenSq, c.y / lenSq, c.z / lenSq};
}

template <core::Arithmetic T> constexpr T Quat<T>::dot(Quat rhs) const
{
    return w * rhs.w + x * rhs.x + y * rhs.y + z * rhs.z;
}

template <core::Arithmetic T> constexpr T Quat<T>::lengthSquared() const { return dot(*this); }

template <core::Arithmetic T> constexpr Quat<T> Quat<T>::normalize() const
{
    if constexpr (std::is_floating_point_v<T>)
    {
        T inv = T(1) / std::sqrt(lengthSquared());
        return {w * inv, x * inv, y * inv, z * inv};
    }
    else
    {
        return *this;
    }
}

template <core::Arithmetic T> constexpr Quat<T> Quat<T>::identity() { return {T{1}, T{}, T{}, T{}}; }

template <core::Arithmetic T> constexpr Quat<T> Quat<T>::fromAxisAngle(Vec3<T> axis, T angleRad)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        T halfAngle = angleRad * T(0.5);
        T s = std::sin(halfAngle);
        T c = std::cos(halfAngle);
        return {c, axis.x * s, axis.y * s, axis.z * s};
    }
    else
    {
        // For integral/fixed-point types: use raw half-angle
        // (users can provide CORDIC-computed sin/cos externally)
        T halfAngle = angleRad / T{2};
        T s = halfAngle; // identity approx — caller should use Cordic
        T c = T{1};      // identity approx
        (void) s;
        (void) c;
        return identity();
    }
}

template <core::Arithmetic T> constexpr Quat<T> Quat<T>::slerp(Quat a, Quat b, T t)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        T d = a.dot(b);

        // If the dot product is negative, negate one to take the short path
        if (d < T(0))
        {
            b = {-b.w, -b.x, -b.y, -b.z};
            d = -d;
        }

        // If quaternions are very close, fall back to normalized lerp
        constexpr T kThreshold = T(0.9995);
        if (d > kThreshold)
        {
            Quat result{a.w + (b.w - a.w) * t, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
            return result.normalize();
        }

        T theta = std::acos(d);
        T sinTheta = std::sin(theta);
        T factorA = std::sin((T(1) - t) * theta) / sinTheta;
        T factorB = std::sin(t * theta) / sinTheta;

        return {factorA * a.w + factorB * b.w, factorA * a.x + factorB * b.x, factorA * a.y + factorB * b.y,
                factorA * a.z + factorB * b.z};
    }
    else
    {
        // For non-floating types: normalized lerp approximation
        Quat result{a.w + (b.w - a.w) * t, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
        return result.normalize();
    }
}

template <core::Arithmetic T> constexpr core::u64 Quat<T>::hash() const
{
    // FNV-1a hash over the raw bytes of {w, x, y, z}
    constexpr core::u64 kFnvBasis = 14695981039346656037ULL;
    constexpr core::u64 kFnvPrime = 1099511628211ULL;

    core::u64 h = kFnvBasis;
    const T components[4] = {w, x, y, z};

    const auto *bytes = reinterpret_cast<const core::u8 *>(components);
    for (core::usize i = 0; i < sizeof(components); ++i)
    {
        h ^= static_cast<core::u64>(bytes[i]);
        h *= kFnvPrime;
    }
    return h;
}

} // namespace lpl::math

#endif // LPL_MATH_QUAT_INL
