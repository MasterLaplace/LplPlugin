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
    /**
     * @brief Constructs an octree covering the given world-space bounds.
     *
     * @param worldBounds  Total world AABB.
     * @param leafCapacity Objects a node holds before it splits.
     *
     * The capacity is a parameter because the right value depends on what the
     * tree is FOR, and the default (32) is tuned for a broad-phase: pairs within
     * a leaf are checked directly, so a bigger leaf trades tree work for cheap
     * pair tests. A frustum cull wants the opposite — every object in a surviving
     * leaf is tested individually, so a large leaf means the hierarchy prunes
     * nothing. Measured on the world viewer: at 32, nineteen chunks made ONE node
     * and the traversal degenerated to the linear scan it was meant to replace.
     */
    explicit Octree(const math::AABB<math::Fixed32> &worldBounds, core::u32 leafCapacity = 32u);
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

    /**
     * @brief Moves the volume the tree subdivides.
     *
     * A broad-phase has one world and never needs this. A CULLER does: the resident
     * set of a streamed world slides with the camera, and an index whose bounds
     * stayed at the origin would spend its whole key range on space that holds
     * nothing. Because the Morton key is now normalised to these bounds, changing
     * them invalidates every key — so this only takes effect on the next @ref
     * rebuild, and callers set it while the tree is empty.
     */
    void setWorldBounds(const math::AABB<math::Fixed32> &worldBounds) noexcept;

    [[nodiscard]] const math::AABB<math::Fixed32> &worldBounds() const noexcept;

    /**
     * @brief Visits objects whose enclosing NODE passes a caller's test.
     *
     * The generalisation @ref query was missing. A box query answers "what is in
     * this region", which serves a broad-phase and cannot serve a frustum: a view
     * volume is not an AABB, and querying its bounding box hands back most of the
     * world whenever the camera is pitched. What a culler actually needs is to
     * reject a NODE and skip its whole subtree — which this tree can do, because it
     * has real node bounds; it simply had no way to expose them.
     *
     * @param nodeVisible  Called with a node's bounds; false prunes the subtree.
     * @param callback     Called with the object id of each survivor.
     * @param outNodesVisited   Optional: nodes tested.
     * @param outNodesPruned    Optional: subtrees rejected whole.
     *
     * The test is applied to node bounds AND to each object's own bounds, so a
     * caller gets exactly the objects that pass, not the contents of nodes that do.
     */
    void queryVisible(const lpl::pmr::function<bool(const math::AABB<math::Fixed32> &)> &nodeVisible,
                      const lpl::pmr::function<void(core::u32)> &callback, core::u32 *outNodesVisited = nullptr,
                      core::u32 *outNodesPruned = nullptr) const;

    [[nodiscard]] core::u32 count() const noexcept override;

private:
    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_OCTREE_HPP
