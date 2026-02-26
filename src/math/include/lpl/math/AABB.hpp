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

// ─── Inline implementations ─────────────────────────────────────────────────

template <core::Arithmetic T>
constexpr AABB<T>::AABB(Vec3<T> mn, Vec3<T> mx) : min(mn), max(mx) {}

template <core::Arithmetic T>
LPL_HD constexpr bool AABB<T>::contains(Vec3<T> point) const
{
    return point.x >= min.x && point.x <= max.x
        && point.y >= min.y && point.y <= max.y
        && point.z >= min.z && point.z <= max.z;
}

template <core::Arithmetic T>
LPL_HD constexpr bool AABB<T>::intersects(AABB other) const
{
    return min.x <= other.max.x && max.x >= other.min.x
        && min.y <= other.max.y && max.y >= other.min.y
        && min.z <= other.max.z && max.z >= other.min.z;
}

template <core::Arithmetic T>
constexpr AABB<T> AABB<T>::merge(AABB other) const
{
    auto lo = [](T a, T b) { return (a < b) ? a : b; };
    auto hi = [](T a, T b) { return (a > b) ? a : b; };
    return AABB{
        Vec3<T>(lo(min.x, other.min.x), lo(min.y, other.min.y), lo(min.z, other.min.z)),
        Vec3<T>(hi(max.x, other.max.x), hi(max.y, other.max.y), hi(max.z, other.max.z))};
}

template <core::Arithmetic T>
constexpr AABB<T> AABB<T>::expand(T margin) const
{
    Vec3<T> m(margin, margin, margin);
    return AABB{min - m, max + m};
}

template <core::Arithmetic T>
constexpr Vec3<T> AABB<T>::center() const
{
    auto half = [](T a, T b) -> T
    {
        if constexpr (std::is_floating_point_v<T>)
            return (a + b) * T(0.5);
        else
            return T::fromRaw((a.raw() + b.raw()) / 2);
    };
    return Vec3<T>(half(min.x, max.x), half(min.y, max.y), half(min.z, max.z));
}

template <core::Arithmetic T>
constexpr Vec3<T> AABB<T>::halfExtents() const
{
    auto half = [](T a, T b) -> T
    {
        if constexpr (std::is_floating_point_v<T>)
            return (b - a) * T(0.5);
        else
            return T::fromRaw((b.raw() - a.raw()) / 2);
    };
    return Vec3<T>(half(min.x, max.x), half(min.y, max.y), half(min.z, max.z));
}

template <core::Arithmetic T>
constexpr T AABB<T>::volume() const
{
    auto ext = max - min;
    return ext.x * ext.y * ext.z;
}

} // namespace lpl::math

#endif // LPL_MATH_AABB_HPP
