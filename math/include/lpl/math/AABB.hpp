/**
 * @file AABB.hpp
 * @brief Axis-Aligned Bounding Box for broadphase collision and spatial queries.
 *
 * @tparam T Scalar type satisfying lpl::core::Arithmetic.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_AABB_HPP
    #define LPL_MATH_AABB_HPP

    #include "Vec3.hpp"
    #include <type_traits>

namespace lpl::math {

template <core::Arithmetic T>
struct AABB final {
    Vec3<T> min{};
    Vec3<T> max{};

    constexpr AABB() = default;
    constexpr AABB(Vec3<T> min, Vec3<T> max);

    [[nodiscard]] LPL_HD constexpr bool contains(Vec3<T> point)   const;
    [[nodiscard]] LPL_HD constexpr bool intersects(AABB other)    const;
    [[nodiscard]] constexpr AABB        merge(AABB other)         const;
    [[nodiscard]] constexpr AABB        expand(T margin)          const;
    [[nodiscard]] constexpr Vec3<T>     center()                  const;
    [[nodiscard]] constexpr Vec3<T>     halfExtents()             const;
    [[nodiscard]] constexpr T           volume()                  const;
};

} // namespace lpl::math

#include "AABB.inl"

#endif // LPL_MATH_AABB_HPP
