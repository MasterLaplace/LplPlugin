// /////////////////////////////////////////////////////////////////////////////
/// @file AntiTunneling.cpp
/// @brief CCD ray-cast and swept AABB implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/physics/AntiTunneling.hpp>

#include <algorithm>

namespace lpl::physics {

std::optional<RayHit> AntiTunneling::rayVsAABB(
    const math::Vec3<math::Fixed32>& origin,
    const math::Vec3<math::Fixed32>& direction,
    const math::AABB<math::Fixed32>& aabb,
    math::Fixed32 tMax) noexcept
{
    auto invDir = [](math::Fixed32 d) -> math::Fixed32 {
        if (d == math::Fixed32{0})
        {
            return math::Fixed32::max();
        }
        return math::Fixed32{1} / d;
    };

    const auto invX = invDir(direction.x);
    const auto invY = invDir(direction.y);
    const auto invZ = invDir(direction.z);

    const auto t1x = (aabb.min.x - origin.x) * invX;
    const auto t2x = (aabb.max.x - origin.x) * invX;
    const auto t1y = (aabb.min.y - origin.y) * invY;
    const auto t2y = (aabb.max.y - origin.y) * invY;
    const auto t1z = (aabb.min.z - origin.z) * invZ;
    const auto t2z = (aabb.max.z - origin.z) * invZ;

    const auto tMinX = std::min(t1x, t2x);
    const auto tMaxX = std::max(t1x, t2x);
    const auto tMinY = std::min(t1y, t2y);
    const auto tMaxY = std::max(t1y, t2y);
    const auto tMinZ = std::min(t1z, t2z);
    const auto tMaxZ = std::max(t1z, t2z);

    const auto tEnter = std::max({tMinX, tMinY, tMinZ});
    const auto tExit  = std::min({tMaxX, tMaxY, tMaxZ});

    if (tEnter > tExit || tExit < math::Fixed32{0} || tEnter > tMax)
    {
        return std::nullopt;
    }

    RayHit hit{};
    hit.t = tEnter;

    if (tEnter == tMinX)
    {
        hit.normal = {(direction.x > math::Fixed32{0}) ? math::Fixed32{-1} : math::Fixed32{1},
                      math::Fixed32{0}, math::Fixed32{0}};
    }
    else if (tEnter == tMinY)
    {
        hit.normal = {math::Fixed32{0},
                      (direction.y > math::Fixed32{0}) ? math::Fixed32{-1} : math::Fixed32{1},
                      math::Fixed32{0}};
    }
    else
    {
        hit.normal = {math::Fixed32{0}, math::Fixed32{0},
                      (direction.z > math::Fixed32{0}) ? math::Fixed32{-1} : math::Fixed32{1}};
    }

    return hit;
}

std::optional<math::Fixed32> AntiTunneling::sweptAABB(
    const math::AABB<math::Fixed32>& movingAABB,
    const math::Vec3<math::Fixed32>& displacement,
    const math::AABB<math::Fixed32>& staticAABB) noexcept
{
    math::AABB<math::Fixed32> expanded{
        {staticAABB.min.x - movingAABB.halfExtents().x,
         staticAABB.min.y - movingAABB.halfExtents().y,
         staticAABB.min.z - movingAABB.halfExtents().z},
        {staticAABB.max.x + movingAABB.halfExtents().x,
         staticAABB.max.y + movingAABB.halfExtents().y,
         staticAABB.max.z + movingAABB.halfExtents().z}
    };

    auto result = rayVsAABB(movingAABB.center(), displacement, expanded, math::Fixed32{1});
    if (!result.has_value())
    {
        return std::nullopt;
    }

    return result->t;
}

} // namespace lpl::physics
