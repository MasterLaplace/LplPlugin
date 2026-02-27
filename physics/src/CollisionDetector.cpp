/**
 * @file CollisionDetector.cpp
 * @brief Narrow-phase collision detection implementations.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/physics/CollisionDetector.hpp>
#include <stdexcept>
#include <lpl/core/Assert.hpp>
#include <algorithm>

namespace lpl::physics {

CollisionResult CollisionDetector::testAABBvsAABB(
    const math::AABB<math::Fixed32>& a,
    const math::AABB<math::Fixed32>& b) noexcept
{
    CollisionResult result{};

    if (!a.intersects(b))
    {
        return result;
    }

    result.colliding = true;

    const auto overlapX = std::min(a.max.x, b.max.x) - std::max(a.min.x, b.min.x);
    const auto overlapY = std::min(a.max.y, b.max.y) - std::max(a.min.y, b.min.y);
    const auto overlapZ = std::min(a.max.z, b.max.z) - std::max(a.min.z, b.min.z);

    if (overlapX <= overlapY && overlapX <= overlapZ)
    {
        result.contact.normal = math::Vec3<math::Fixed32>{
            (a.center().x < b.center().x) ? math::Fixed32{-1} : math::Fixed32{1},
            math::Fixed32{0}, math::Fixed32{0}};
        result.contact.penetrationDepth = overlapX;
    }
    else if (overlapY <= overlapZ)
    {
        result.contact.normal = math::Vec3<math::Fixed32>{
            math::Fixed32{0},
            (a.center().y < b.center().y) ? math::Fixed32{-1} : math::Fixed32{1},
            math::Fixed32{0}};
        result.contact.penetrationDepth = overlapY;
    }
    else
    {
        result.contact.normal = math::Vec3<math::Fixed32>{
            math::Fixed32{0}, math::Fixed32{0},
            (a.center().z < b.center().z) ? math::Fixed32{-1} : math::Fixed32{1}};
        result.contact.penetrationDepth = overlapZ;
    }

    const auto ca = a.center();
    const auto cb = b.center();
    result.contact.position = math::Vec3<math::Fixed32>{
        (ca.x + cb.x) / math::Fixed32{2},
        (ca.y + cb.y) / math::Fixed32{2},
        (ca.z + cb.z) / math::Fixed32{2}};

    return result;
}

CollisionResult CollisionDetector::testSphereVsSphere(
    const math::Vec3<math::Fixed32>& centerA, math::Fixed32 radiusA,
    const math::Vec3<math::Fixed32>& centerB, math::Fixed32 radiusB) noexcept
{
    CollisionResult result{};

    const auto diff = centerB - centerA;
    const auto distSq = diff.lengthSquared();
    const auto radSum = radiusA + radiusB;
    const auto radSumSq = radSum * radSum;

    if (distSq > radSumSq)
    {
        return result;
    }

    result.colliding = true;
    result.contact.normal = diff.normalize();
    result.contact.penetrationDepth = radSum;
    result.contact.position = centerA + result.contact.normal * radiusA;

    return result;
}

CollisionResult CollisionDetector::testGJK(
    const void* /*shapeA*/,
    const void* /*shapeB*/) noexcept
{
    LPL_ASSERT(false && "unimplemented");
    return {};
}

CollisionResult CollisionDetector::testSAT(
    const void* /*obbA*/,
    const void* /*obbB*/) noexcept
{
    LPL_ASSERT(false && "unimplemented");
    return {};
}

} // namespace lpl::physics
