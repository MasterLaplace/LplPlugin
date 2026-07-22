/**
 * @file Partition.hpp
 * @brief Chunk-based ECS storage with SoA double-buffering.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_PARTITION_HPP
#    define LPL_ECS_PARTITION_HPP

#    include <lpl/core/Constants.hpp>
#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Component.hpp>
#    include <lpl/memory/IAllocator.hpp>
#    include <lpl/ecs/Entity.hpp>

#    include <lpl/std/memory.hpp>
#    include <lpl/std/vector.hpp>

#    include <cstddef>
#    include <span>

namespace lpl::ecs {

/**
 * @class Chunk
 * @brief Fixed-capacity block storing entities of a single archetype in
 *        SoA layout with double-buffered component arrays.
 *
 * Each component occupies a contiguous typed array sized to
 * @c kChunkCapacity. Two buffers (front/back) enable lock-free read
 * during write.
 */
class Chunk final : public core::NonCopyable<Chunk> {
public:
    static constexpr core::u32 kChunkCapacity = 256;

    /**
     * @brief Constructs a chunk for the given archetype and component layouts.
     * @param archetype The archetype this chunk stores.
     * @param layouts   Span of component layouts.
     * @param allocator Where the component buffers come from. nullptr keeps the
     *        default heap allocator; a World passes its persistent arena so ECS
     *        storage is bump-allocated out of one bounded reservation instead of
     *        hitting the heap (which the freestanding REAL_TIME mode forbids on
     *        a tick's path).
     */
    Chunk(const Archetype &archetype, std::span<const ComponentLayout> layouts,
          memory::IAllocator *allocator = nullptr);

    ~Chunk();

    /** @brief Number of live entities in this chunk. */
    [[nodiscard]] core::u32 count() const noexcept;

    /** @brief Tests whether the chunk is full. */
    [[nodiscard]] bool isFull() const noexcept;

    /**
     * @brief Adds an entity to the chunk.
     * @param id Entity to add.
     * @return Local index within the chunk, or error if full.
     */
    [[nodiscard]] core::Expected<core::u32> add(EntityId id);

    /**
     * @brief Removes an entity by local index (swap-and-pop).
     * @param localIndex Index to remove.
     * @return The EntityId that was swapped into @p localIndex (or null).
     */
    [[nodiscard]] core::Expected<EntityId> remove(core::u32 localIndex);

    /**
     * @brief Gets a read-only pointer to the front buffer of a component.
     * @param id Component to access.
     * @return Pointer to the front array, or nullptr if not present.
     */
    [[nodiscard]] const void *readComponent(ComponentId id) const noexcept;

    /**
     * @brief Gets a writable pointer to the back buffer of a component.
     * @param id Component to access.
     * @return Pointer to the back array, or nullptr if not present.
     */
    [[nodiscard]] void *writeComponent(ComponentId id) noexcept;

    /** @brief Swaps front and back buffers (publish step at tick boundary). */
    void swapBuffers() noexcept;

    /** @brief Returns the archetype of this chunk. */
    [[nodiscard]] const Archetype &archetype() const noexcept;

    /** @brief Returns a span of entity IDs in this chunk (count() elements). */
    [[nodiscard]] std::span<const EntityId> entities() const noexcept;

    /**
     * @brief Finds the local index of an entity by its ID in O(1).
     *
     * Uses the internal sparse set: no linear scan of the dense array.
     *
     * @param id Entity to look up.
     * @return Local index within [0, count()), or std::nullopt if not found.
     */
    [[nodiscard]] std::optional<core::u32> findLocalIndex(EntityId id) const noexcept;

private:
    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

/**
 * @class Partition
 * @brief Top-level container managing chunks for a single archetype.
 *
 * Grows automatically by allocating new Chunks when existing ones are full.
 */
class Partition final : public core::NonCopyable<Partition> {
public:
    /**
     * @brief Constructs a partition for the given archetype.
     * @param archetype  Archetype of entities stored here.
     * @param layouts    Component layouts for the archetype.
     * @param allocator  Optional arena for chunk storage. nullptr keeps the default heap allocator.
     */
    Partition(Archetype archetype, lpl::pmr::vector<ComponentLayout> layouts,
              memory::IAllocator *allocator = nullptr);

    ~Partition();

    /**
     * @brief Allocates a slot for a new entity.
     * @param id Entity to insert.
     * @return EntityRef pointing to the new slot.
     */
    [[nodiscard]] core::Expected<EntityRef> insert(EntityId id);

    /**
     * @brief Removes an entity by its reference.
     * @param ref Reference obtained from @ref insert or a lookup.
     * @return Swapped EntityId on success, or error.
     */
    [[nodiscard]] core::Expected<EntityId> erase(const EntityRef &ref);

    /** @brief Returns the total number of live entities across all chunks. */
    [[nodiscard]] core::u32 entityCount() const noexcept;

    /** @brief Returns a read-only view of all chunks. */
    [[nodiscard]] std::span<const lpl::pmr::unique_ptr<Chunk>> chunks() const noexcept;

    /** @brief Swaps front/back buffers on every chunk. */
    void swapAllBuffers() noexcept;

    /** @brief Returns the archetype. */
    [[nodiscard]] const Archetype &archetype() const noexcept;

private:
    Archetype _archetype;
    lpl::pmr::vector<ComponentLayout> _layouts;
    lpl::pmr::vector<lpl::pmr::unique_ptr<Chunk>> _chunks;
    memory::IAllocator *_allocator = nullptr; ///< Optional arena for chunk storage.
};

} // namespace lpl::ecs

#endif // LPL_ECS_PARTITION_HPP
