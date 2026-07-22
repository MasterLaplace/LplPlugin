/**
 * @file Octree.hpp
 * @brief Morton-sorted linear octree for broad-phase collision.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_OCTREE_HPP
#    define LPL_PHYSICS_OCTREE_HPP

#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/physics/ISpatialIndex.hpp>

#    include <lpl/std/memory.hpp>

namespace lpl::physics {

/**
 * @class Octree
 * @brief Flat, Morton-sorted radix octree. Objects are sorted by their
 *        Morton code each rebuild pass, enabling cache-friendly traversal
 *        and O(n log n) broad-phase.
 */
class Octree final : public ISpatialIndex, public core::NonCopyable<Octree> {
public:
    /**
     * @brief Constructs an octree covering the given world-space bounds.
     * @param worldBounds Total world AABB.
     */
    explicit Octree(const math::AABB<math::Fixed32> &worldBounds);
    ~Octree() override;

    void insert(core::u32 objectId, const math::AABB<math::Fixed32> &aabb) override;

    void update(core::u32 objectId, const math::AABB<math::Fixed32> &aabb) override;

    void remove(core::u32 objectId) override;

    void query(const math::AABB<math::Fixed32> &region,
               const lpl::pmr::function<void(core::u32)> &callback) const override;

    void rebuild() override;

    /**
     * @brief Drops every object but keeps all buffers and their capacity.
     *
     * For the common broad-phase pattern of refilling the index from scratch
     * each step: clear + re-insert reuses the memory the previous step already
     * grew, so a warm tick performs no allocation at all. Constructing a fresh
     * Octree per step instead pays for the whole structure every time.
     *
     * clear() on all three keeps the capacity; the id table is blanked in place
     * rather than shrunk, so a refill of the same id range reallocates nothing.
     * tempEntries is radix-sort scratch, never read before being written: left
     * at its current size so the sort does not have to re-grow it every step.
     */
    void clear() noexcept;

    [[nodiscard]] core::u32 count() const noexcept override;

private:
    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_OCTREE_HPP
