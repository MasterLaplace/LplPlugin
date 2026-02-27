/**
 * @file ISpatialIndex.hpp
 * @brief Abstract spatial index interface for broad-phase queries.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_ISPATIALINDEX_HPP
    #define LPL_PHYSICS_ISPATIALINDEX_HPP

#include <lpl/math/AABB.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>

#include <functional>

namespace lpl::physics {

/**
 * @class ISpatialIndex
 * @brief Strategy interface for spatial acceleration structures.
 *
 * Concrete implementations: @c Octree, @c SpatialGrid.
 */
class ISpatialIndex
{
public:
    virtual ~ISpatialIndex() = default;

    /**
     * @brief Inserts an object with the given AABB.
     * @param objectId User-defined identifier.
     * @param aabb     Axis-aligned bounding box.
     */
    virtual void insert(core::u32 objectId,
                        const math::AABB<math::Fixed32>& aabb) = 0;

    /** @brief Updates the AABB of an already-inserted object. */
    virtual void update(core::u32 objectId,
                        const math::AABB<math::Fixed32>& aabb) = 0;

    /** @brief Removes an object from the index. */
    virtual void remove(core::u32 objectId) = 0;

    /**
     * @brief Queries all objects whose AABB overlaps the given region.
     * @param region   Query AABB.
     * @param callback Called for every overlapping object ID.
     */
    virtual void query(const math::AABB<math::Fixed32>& region,
                       const std::function<void(core::u32)>& callback) const = 0;

    /** @brief Rebuilds the internal structure (e.g. re-sort Morton keys). */
    virtual void rebuild() = 0;

    /** @brief Returns the total number of tracked objects. */
    [[nodiscard]] virtual core::u32 count() const noexcept = 0;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_ISPATIALINDEX_HPP
