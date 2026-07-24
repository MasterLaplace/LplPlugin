/**
 * @file Registry.hpp
 * @brief Central entity registry — creates, destroys, and locates entities.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_REGISTRY_HPP
#    define LPL_ECS_REGISTRY_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Entity.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/memory/IAllocator.hpp>

#    include <lpl/std/memory.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ecs {

/**
 * @class Registry
 * @brief Owns the entity free-list, generation table, and partition map.
 *
 * Entities are created with a unique slot + generation. On destruction the
 * slot is recycled and the generation is bumped, invalidating stale
 * EntityIds.
 */
class Registry final : public core::NonCopyable<Registry> {
public:
    /**
     * @brief Default-constructs an empty registry.
     * @brief Default-constructs an empty registry.
     */
    Registry();
    ~Registry();

    /**
     * @brief Route this registry's chunk storage through @p allocator.
     *
     * Must be called before any entity is created: chunks already built keep
     * the allocator they were made with. A World sets this to its persistent
     * arena so ECS storage is one bounded reservation instead of many heap
     * allocations. nullptr restores the default heap allocator.
     */
    void setAllocator(memory::IAllocator *allocator) noexcept;

    // --------------------------------------------------------------------- //
    //  Entity lifecycle                                                      //
    // --------------------------------------------------------------------- //

    /**
     * @brief Creates a new entity with the given archetype.
     * @param archetype Archetype determining which components the entity has.
     * @return Entity identifier on success, or error on pool exhaustion.
     */
    [[nodiscard]] core::Expected<EntityId> createEntity(const Archetype &archetype);

    /**
     * @brief Creates an entity under a caller-supplied id (network reconciliation).
     *
     * A networked client does not mint its own ids for the entities the server
     * owns: their identity IS the server's EntityId, carried on the wire and
     * folded into World::stateHash (§6.4). The client therefore adopts that exact
     * id so a later snapshot for the same entity resolves to it — updated in
     * place, not duplicated — and so client and server digest the same bytes.
     * This is the modern form of the legacy EntityRegistry::registerEntity(publicId).
     *
     * The id's slot is reserved directly; a subsequent local createEntity never
     * mints it (allocateSlot skips a slot already alive). Idempotent: if the slot
     * already holds this exact id, the call succeeds and changes nothing.
     *
     * Not thread-safe against a concurrent createEntity: reconciliation runs in
     * its scheduler phase, single-threaded, like the legacy StateReconciliation.
     *
     * @param id        The identity to adopt (typically from a state snapshot).
     * @param archetype Components the entity has.
     * @return @p id on success; an error if the slot is out of range or already
     *         holds a different live entity.
     */
    [[nodiscard]] core::Expected<EntityId> createEntityWithId(EntityId id, const Archetype &archetype);

    /**
     * @brief Destroys an entity, recycling its slot.
     * @param id Entity to destroy.
     * @return OK on success, or error if the entity is already dead.
     */
    [[nodiscard]] core::Expected<void> destroyEntity(EntityId id);

    /** @brief Tests whether an entity is alive (generation matches). */
    [[nodiscard]] bool isAlive(EntityId id) const noexcept;

    // --------------------------------------------------------------------- //
    //  Lookup                                                                //
    // --------------------------------------------------------------------- //

    /**
     * @brief Resolves an EntityId to its in-partition reference.
     * @return EntityRef on success, or error if dead/unknown.
     */
    [[nodiscard]] core::Expected<EntityRef> resolve(EntityId id) const;

    /** @brief Returns the total number of live entities. */
    [[nodiscard]] core::u32 liveCount() const noexcept;

    // --------------------------------------------------------------------- //
    //  Partition access                                                      //
    // --------------------------------------------------------------------- //

    /** @brief Returns or creates the partition for the given archetype. */
    [[nodiscard]] Partition &getOrCreatePartition(const Archetype &archetype);

    /** @brief Returns a read-only view of all partitions. */
    [[nodiscard]] std::span<const lpl::pmr::unique_ptr<Partition>> partitions() const noexcept;

    /** @brief Swaps front/back buffers on every partition (end-of-tick). */
    void swapAllBuffers() noexcept;

private:
    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::ecs

#endif // LPL_ECS_REGISTRY_HPP
