/**
 * @file Quat.hpp
 * @brief Quaternion type for 3D rotation, parameterised on scalar type.
 *
 * Uses Hamilton convention (w, x, y, z) where w is the real part.
 * All operations are deterministic when instantiated with Fixed32.
 *
 * @tparam T Scalar type satisfying lpl::core::Arithmetic.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_QUAT_HPP
#    define LPL_MATH_QUAT_HPP

#    include "Vec3.hpp"

namespace lpl::math {

template <core::Arithmetic T> struct Quat final {
    T w{};
    T x{};
    T y{};
    T z{};

    constexpr Quat() = default;
    constexpr Quat(T w, T x, T y, T z);

    [[nodiscard]] constexpr Quat operator*(Quat rhs) const;
    [[nodiscard]] constexpr Vec3<T> rotate(Vec3<T> v) const;
    [[nodiscard]] constexpr Quat conjugate() const;
    [[nodiscard]] constexpr Quat inverse() const;
    [[nodiscard]] constexpr T dot(Quat rhs) const;
    [[nodiscard]] constexpr T lengthSquared() const;
    [[nodiscard]] constexpr Quat normalize() const;

    static constexpr Quat identity();

    /**
     * @brief The rotation of @p angleRad about @p axis, for a floating-point @p T; the identity for any other.
     *
     * A fixed-point caller takes the sine and cosine of the half angle from Cordic and builds the
     * quaternion itself.
     *
     * @param axis     Axis of the rotation.
     * @param angleRad Angle of the rotation, in radians.
     * @return The rotation.
     */
    static constexpr Quat fromAxisAngle(Vec3<T> axis, T angleRad);

    /**
     * @brief Spherical interpolation from @p a to @p b, for a floating-point @p T.
     *
     * When the dot product is negative, one end is negated to take the short path; when the two are
     * very close (a dot product above 0.9995), it falls back to a normalized lerp. For any other @p T
     * it is the lerp alone: normalize() leaves a non-floating quaternion as it is.
     *
     * @param a Start, at @p t = 0.
     * @param b End, at @p t = 1.
     * @param t Interpolation parameter.
     * @return The interpolated rotation.
     */
    static constexpr Quat slerp(Quat a, Quat b, T t);

    /**
     * @brief FNV-1a hash over the raw bytes of {w, x, y, z}.
     */
    [[nodiscard]] constexpr core::u64 hash() const;
};

using Quatf = Quat<float>;

} // namespace lpl::math

#    include "Quat.inl"

#endif // LPL_MATH_QUAT_HPP
