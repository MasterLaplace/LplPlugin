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
#    define LPL_MATH_MAT4_HPP

#    include "Cordic.hpp"
#    include "Quat.hpp"
#    include "Vec3.hpp"

#    include <array>
#    include <lpl/std/cmath.hpp>
#    include <type_traits>

namespace lpl::math {

template <core::Arithmetic T> struct Mat4 final {
    std::array<T, 16u> m{};

    constexpr Mat4() = default;

    [[nodiscard]] constexpr T &operator()(core::u32 row, core::u32 col);
    [[nodiscard]] constexpr T operator()(core::u32 row, core::u32 col) const;

    [[nodiscard]] constexpr Mat4 operator*(Mat4 rhs) const;
    [[nodiscard]] constexpr Vec3<T> transformPoint(Vec3<T> p) const;
    [[nodiscard]] constexpr Vec3<T> transformDirection(Vec3<T> d) const;

    static constexpr Mat4 identity();
    static constexpr Mat4 translate(Vec3<T> offset);
    static constexpr Mat4 scale(Vec3<T> s);
    static constexpr Mat4 fromQuat(Quat<T> q);

    /**
     * @brief A perspective projection, for a floating-point @p T; the identity for any other.
     *
     * tan(fov/2) comes from CORDIC, not from libm. The determinism contract forbids libm
     * transcendentals in engine code linked into the kernel, and tan is the one that has no way around
     * it: sqrt folds to the SSE instruction and is IEEE-exact on both targets, but tan always becomes a
     * call — resolved to glibc on the host and to the kernel's own Taylor-series tanf in ring 0. Those
     * two disagree in the low bits, so every projection matrix, and every image folded through one,
     * would differ between the oracle and the kernel.
     *
     * CORDIC is shifts and adds over Fixed32, identical everywhere. The divide and the widening after
     * it are SSE operations, which are IEEE-defined and therefore also identical on both targets.
     *
     * @param fovRad Vertical field of view, in radians.
     * @param aspect Width over height.
     * @param near   Distance to the near plane.
     * @param far    Distance to the far plane.
     * @return The projection matrix.
     */
    static Mat4 perspective(T fovRad, T aspect, T near, T far);

    static Mat4 lookAt(Vec3<T> eye, Vec3<T> target, Vec3<T> up);
};

using Mat4f = Mat4<float>;

} // namespace lpl::math

#    include "Mat4.inl"

#endif // LPL_MATH_MAT4_HPP
