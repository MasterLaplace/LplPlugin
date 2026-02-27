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
    #define LPL_ECS_REGISTRY_HPP

#include <lpl/ecs/Entity.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>
#include <vector>

namespace lpl::ecs {

/**
 * @class Registry
 * @brief Owns the entity free-list, generation table, and partition map.
 *
 * Entities are created with a unique slot + generation. On destruction the
 * slot is recycled and the generation is bumped, invalidating stale
 * EntityIds.
 */
class Registry final : public core::NonCopyable<Registry>
{
public:
    /** @brief Default-constructs an empty registry. */
    Registry();
    ~Registry();

    // --------------------------------------------------------------------- //
    //  Entity lifecycle                                                      //
    // --------------------------------------------------------------------- //

    /**
     * @brief Creates a new entity with the given archetype.
     * @param archetype Archetype determining which components the entity has.
     * @return Entity identifier on success, or error on pool exhaustion.
     */
    [[nodiscard]] core::Expected<EntityId> createEntity(const Archetype& archetype);

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
    [[nodiscard]] Partition& getOrCreatePartition(const Archetype& archetype);

    /** @brief Returns a read-only view of all partitions. */
    [[nodiscard]] std::span<const std::unique_ptr<Partition>> partitions() const noexcept;

    /** @brief Swaps front/back buffers on every partition (end-of-tick). */
    void swapAllBuffers() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::ecs

#endif // LPL_ECS_REGISTRY_HPP
