/**
 * @file AntiTunneling.hpp
 * @brief Continuous collision detection (CCD) to prevent fast-moving
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_ANTITUNNELING_HPP
    #define LPL_PHYSICS_ANTITUNNELING_HPP

#include <lpl/math/AABB.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>

#include <optional>

namespace lpl::physics {

/**
 * @struct RayHit
 * @brief Result of a ray cast against an AABB.
 */
struct RayHit
{
    math::Fixed32             t;
    math::Vec3<math::Fixed32> normal;
};

/**
 * @class AntiTunneling
 * @brief Provides swept-AABB and ray-AABB tests for CCD.
 */
class AntiTunneling
{
public:
    /**
     * @brief Casts a ray against an AABB (slab test).
     * @param origin    Ray origin.
     * @param direction Ray direction (not necessarily normalised).
     * @param aabb      Target bounding box.
     * @param tMax      Maximum ray parameter.
     * @return Hit result if intersection exists within [0, tMax].
     */
    [[nodiscard]] static std::optional<RayHit> rayVsAABB(
        const math::Vec3<math::Fixed32>& origin,
        const math::Vec3<math::Fixed32>& direction,
        const math::AABB<math::Fixed32>& aabb,
        math::Fixed32 tMax) noexcept;

    /**
     * @brief Swept AABB vs AABB test.
     * @param movingAABB    AABB of the moving body at t=0.
     * @param displacement  Velocity × dt displacement vector.
     * @param staticAABB    AABB of the static obstacle.
     * @return Time of first contact in [0,1], or empty if no contact.
     */
    [[nodiscard]] static std::optional<math::Fixed32> sweptAABB(
        const math::AABB<math::Fixed32>& movingAABB,
        const math::Vec3<math::Fixed32>& displacement,
        const math::AABB<math::Fixed32>& staticAABB) noexcept;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_ANTITUNNELING_HPP
