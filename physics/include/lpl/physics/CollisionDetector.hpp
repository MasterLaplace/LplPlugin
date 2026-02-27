/**
 * @file CollisionDetector.hpp
 * @brief Narrow-phase collision detection: AABB, GJK, SAT.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_COLLISIONDETECTOR_HPP
    #define LPL_PHYSICS_COLLISIONDETECTOR_HPP

#include <lpl/math/AABB.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>

#include <optional>

namespace lpl::physics {

/**
 * @struct ContactPoint
 * @brief Single contact between two colliding bodies.
 */
struct ContactPoint
{
    math::Vec3<math::Fixed32> position;
    math::Vec3<math::Fixed32> normal;
    math::Fixed32             penetrationDepth;
};

/**
 * @struct CollisionResult
 * @brief Result of a narrow-phase test between two bodies.
 */
struct CollisionResult
{
    bool                          colliding{false};
    ContactPoint                  contact{};
};

/**
 * @class CollisionDetector
 * @brief Stateless narrow-phase collision query functions.
 *
 * All functions are deterministic (Fixed32 math only).
 */
class CollisionDetector
{
public:
    /**
     * @brief AABB vs AABB intersection test.
     * @param a First bounding box.
     * @param b Second bounding box.
     * @return Collision result with contact if overlapping.
     */
    [[nodiscard]] static CollisionResult testAABBvsAABB(
        const math::AABB<math::Fixed32>& a,
        const math::AABB<math::Fixed32>& b) noexcept;

    /**
     * @brief Sphere vs sphere intersection test.
     * @param centerA Centre of sphere A.
     * @param radiusA Radius of sphere A.
     * @param centerB Centre of sphere B.
     * @param radiusB Radius of sphere B.
     * @return Collision result.
     */
    [[nodiscard]] static CollisionResult testSphereVsSphere(
        const math::Vec3<math::Fixed32>& centerA, math::Fixed32 radiusA,
        const math::Vec3<math::Fixed32>& centerB, math::Fixed32 radiusB) noexcept;

    /**
     * @brief GJK convex vs convex support-mapping test (stub).
     * @return Collision result.
     */
    [[nodiscard]] static CollisionResult testGJK(
        const void* shapeA,
        const void* shapeB) noexcept;

    /**
     * @brief SAT (Separating Axis Theorem) test for oriented boxes (stub).
     * @return Collision result.
     */
    [[nodiscard]] static CollisionResult testSAT(
        const void* obbA,
        const void* obbB) noexcept;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_COLLISIONDETECTOR_HPP
