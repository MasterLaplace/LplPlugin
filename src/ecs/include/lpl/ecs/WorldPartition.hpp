// /////////////////////////////////////////////////////////////////////////////
/// @file WorldPartition.hpp
/// @brief Spatial world partitioning using Morton-code indexed cells.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/ecs/Entity.hpp>
#include <lpl/math/Morton.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/container/FlatAtomicHashMap.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>
#include <lpl/core/Expected.hpp>

#include <memory>
#include <vector>

namespace lpl::ecs {

// /////////////////////////////////////////////////////////////////////////////
/// @class WorldPartition
/// @brief Divides the world into Morton-order cells for broad-phase spatial
///        queries and interest management.
///
/// Each cell is identified by its Morton code. Entities are assigned to a
/// cell based on their position (snapped to cell-grid coordinates).
// /////////////////////////////////////////////////////////////////////////////
class WorldPartition final : public core::NonCopyable<WorldPartition>
{
public:
    /// @brief Constructs a world partition with the given cell size.
    /// @param cellSize Side length of each cubic cell (in world units).
    explicit WorldPartition(math::Fixed32 cellSize);

    ~WorldPartition();

    /// @brief Inserts or moves an entity to the cell corresponding to @p pos.
    /// @param id  Entity to insert/update.
    /// @param pos World-space position.
    /// @return OK on success.
    [[nodiscard]] core::Expected<void> insertOrUpdate(EntityId id,
                                                      const math::Vec3<math::Fixed32>& pos);

    /// @brief Removes an entity from its current cell.
    /// @param id Entity to remove.
    /// @return OK on success.
    [[nodiscard]] core::Expected<void> remove(EntityId id);

    /// @brief Queries all entities within @p radius of @p center.
    /// @param center Query center in world-space.
    /// @param radius Query radius.
    /// @param[out] results Populated with entity IDs in range.
    void queryRadius(const math::Vec3<math::Fixed32>& center,
                     math::Fixed32 radius,
                     std::vector<EntityId>& results) const;

    /// @brief Returns the Morton code for a world-space position.
    [[nodiscard]] core::u64 mortonForPosition(const math::Vec3<math::Fixed32>& pos) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::ecs
