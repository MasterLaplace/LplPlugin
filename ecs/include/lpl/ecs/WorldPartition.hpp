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
#    define LPL_ECS_WORLDPARTITION_HPP

#    include <lpl/container/FlatAtomicHashMap.hpp>
#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Entity.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Morton.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/std/functional.hpp>
#    include <lpl/std/memory.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::gpu {
class IComputeBackend;
}

namespace lpl::ecs {

/**
 * @class WorldPartition
 * @brief Divides the world into Morton-order cells for broad-phase spatial
 *        queries and interest management.
 *
 * Each cell is identified by its Morton code. Entities are assigned to a
 * cell based on their position (snapped to cell-grid coordinates).
 */
class WorldPartition final : public core::NonCopyable<WorldPartition> {
public:
    /** @brief Default spatial-cell capacity (legacy WORLD_CAPACITY = 1 << 16). */
    static constexpr core::u32 kDefaultCellCapacity = 1u << 16u;

    /**
     * @brief Constructs a world partition with the given cell size.
     *
     * @param cellSize Side length of each cubic cell (in world units).
     * @param cellCapacity Maximum number of spatial cells. This is budgeted up
     *        front, not grown on demand: the backing map allocates one atomic
     *        entry per slot plus a cell pool, so the default 65536 costs a few
     *        MiB. Hosts can afford it; memory-tight targets (the freestanding
     *        kernel runs on a 4 MiB heap) must pass a smaller figure sized to
     *        the world they actually simulate.
     */
    explicit WorldPartition(math::Fixed32 cellSize, core::u32 cellCapacity = kDefaultCellCapacity);

    ~WorldPartition();

    /**
     * @brief Inserts or moves an entity to the cell corresponding to @p pos.
     * @param id  Entity to insert/update.
     * @param pos World-space position.
     * @return OK on success.
     */
    [[nodiscard]] core::Expected<void> insertOrUpdate(EntityId id, const math::Vec3<math::Fixed32> &pos);

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
    void queryRadius(const math::Vec3<math::Fixed32> &center, math::Fixed32 radius,
                     pmr::vector<EntityId> &results) const;

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
     * @brief Re-assigns all tracked entities to the correct spatial cell
     *        based on a position-provider callback.
     *
     * Must be called post-physics to fix entities that moved across cell
     * boundaries. The callback receives an entity raw ID and must return
     * its current world-space position.
     *
     * @param positionOf Callback: (core::u32 rawId) → Vec3<Fixed32> position.
     * @return Number of entities that migrated to a different cell.
     */
    core::u32 migrateEntities(const pmr::function<math::Vec3<math::Fixed32>(core::u32)> &positionOf);

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
    [[nodiscard]] core::u64 mortonForPosition(const math::Vec3<math::Fixed32> &pos) const noexcept;

    /** @brief Returns the number of active cells. */
    [[nodiscard]] core::u32 cellCount() const noexcept;

public:
    /** @brief GPU dispatch threshold (entity count). */
    static constexpr core::u32 kGpuThreshold = 4096;

    /**
     * @brief Registers an optional GPU backend for physics dispatch.
     *
     * When the backend is set and entity count exceeds @c kGpuThreshold,
     * @c step() automatically offloads physics computation to the GPU.
     * Pass @c nullptr to revert to CPU-only.
     *
     * @param backend GPU backend (non-owning pointer; lifetime managed by caller).
     */
    void setGpuBackend(gpu::IComputeBackend *backend) noexcept;

private:
    struct Impl;
    pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::ecs

#endif // LPL_ECS_WORLDPARTITION_HPP
