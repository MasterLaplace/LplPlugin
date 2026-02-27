/**
 * @file AABB.inl
 * @brief Inline implementations of AABB template methods.
 *
 * @note This file is automatically included at the end of AABB.hpp.
 *       Do not include it directly.
 */

namespace lpl::math {

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
