/**
 * @file WorldPartition.hpp
 * @brief Spatial world partitioning using Morton-code indexed cells.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_WORLDPARTITION_HPP
    #define LPL_ECS_WORLDPARTITION_HPP

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

/**
 * @class WorldPartition
 * @brief Divides the world into Morton-order cells for broad-phase spatial
 *        queries and interest management.
 *
 * Each cell is identified by its Morton code. Entities are assigned to a
 * cell based on their position (snapped to cell-grid coordinates).
 */
class WorldPartition final : public core::NonCopyable<WorldPartition>
{
public:
    /**
     * @brief Constructs a world partition with the given cell size.
     * @param cellSize Side length of each cubic cell (in world units).
     */
    explicit WorldPartition(math::Fixed32 cellSize);

    ~WorldPartition();

    /**
     * @brief Inserts or moves an entity to the cell corresponding to @p pos.
     * @param id  Entity to insert/update.
     * @param pos World-space position.
     * @return OK on success.
     */
    [[nodiscard]] core::Expected<void> insertOrUpdate(EntityId id,
                                                      const math::Vec3<math::Fixed32>& pos);

    /**
     * @brief Removes an entity from its current cell.
     * @param id Entity to remove.
     * @return OK on success.
     */
    [[nodiscard]] core::Expected<void> remove(EntityId id);

    /**
     * @brief Queries all entities within @p radius of @p center.
     * @param center Query center in world-space.
     * @param radius Query radius.
     * @param[out] results Populated with entity IDs in range.
     */
    void queryRadius(const math::Vec3<math::Fixed32>& center,
                     math::Fixed32 radius,
                     std::vector<EntityId>& results) const;

    /**
     * @brief Runs one physics tick, auto-selecting CPU or GPU backend.
     *
     * Ported from legacy WorldPartition::tick(). If a GPU backend is
     * registered and entity count exceeds @c kGpuThreshold, dispatches to
     * GPU; otherwise falls back to CPU.
     *
     * @param dt Fixed delta-time.
     */
    void step(core::f32 dt);

    /**
     * @brief Garbage-collects empty Morton cells.
     *
     * Should be called periodically (e.g., every N ticks) to release
     * cells that no longer contain any entities.
     *
     * @return Number of cells removed.
     */
    core::u32 gcEmptyCells();

    /** @brief Returns the Morton code for a world-space position. */
    [[nodiscard]] core::u64 mortonForPosition(const math::Vec3<math::Fixed32>& pos) const noexcept;

    /** @brief Returns the number of active cells. */
    [[nodiscard]] core::u32 cellCount() const noexcept;

    /** @brief GPU dispatch threshold (entity count). */
    static constexpr core::u32 kGpuThreshold = 4096;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::ecs

#endif // LPL_ECS_WORLDPARTITION_HPP
