/**
 * @file Entity.hpp
 * @brief Entity identifier: packed generation + slot index.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_ENTITY_HPP
    #define LPL_ECS_ENTITY_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/Constants.hpp>

#include <cstdint>
#include <functional>
#include <limits>

namespace lpl::ecs {

/**
 * @class EntityId
 * @brief Packed 32-bit entity identifier.
 *
 * Layout (MSB → LSB):
 *   [generation : kGenerationBits] [slot : kSlotBits]
 *
 * The generation counter detects stale references after an entity is
 * destroyed and its slot is recycled.
 */
class EntityId final
{
public:
    static constexpr core::u32 kGenerationBits = core::kGenerationBits;
    static constexpr core::u32 kSlotBits       = core::kSlotBits;
    static constexpr core::u32 kSlotMask       = (1u << kSlotBits) - 1u;
    static constexpr core::u32 kGenerationMask = (1u << kGenerationBits) - 1u;

    /** @brief Null sentinel. */
    static constexpr core::u32 kNull = std::numeric_limits<core::u32>::max();

    /** @brief Default-constructs a null entity. */
    constexpr EntityId() noexcept = default;

    /**
     * @brief Constructs from a raw packed value.
     * @param raw Packed generation|slot value.
     */
    constexpr explicit EntityId(core::u32 raw) noexcept
        : _raw{raw}
    {}

    /**
     * @brief Constructs from separate generation + slot.
     * @param generation Generation counter.
     * @param slot       Slot index.
     */
    constexpr EntityId(core::u32 generation, core::u32 slot) noexcept
        : _raw{((generation & kGenerationMask) << kSlotBits) | (slot & kSlotMask)}
    {}

    /** @brief Returns the slot index. */
    [[nodiscard]] constexpr core::u32 slot() const noexcept
    {
        return _raw & kSlotMask;
    }

    /** @brief Returns the generation counter. */
    [[nodiscard]] constexpr core::u32 generation() const noexcept
    {
        return (_raw >> kSlotBits) & kGenerationMask;
    }

    /** @brief Returns the raw packed value. */
    [[nodiscard]] constexpr core::u32 raw() const noexcept { return _raw; }

    /** @brief Tests whether the entity is valid (non-null). */
    [[nodiscard]] constexpr bool isValid() const noexcept
    {
        return _raw != kNull;
    }

    /** @brief Equality comparison. */
    [[nodiscard]] constexpr bool operator==(EntityId other) const noexcept
    {
        return _raw == other._raw;
    }

    /** @brief Ordering comparison. */
    [[nodiscard]] constexpr auto operator<=>(EntityId other) const noexcept
    {
        return _raw <=> other._raw;
    }

private:
    core::u32 _raw{kNull};
};

/**
 * @struct EntityRef
 * @brief Lightweight proxy referencing an entity within a partition.
 *
 * Allows component access via the owning partition context.
 */
struct EntityRef
{
    EntityId  id{};
    core::u32 chunkIndex{0};
    core::u32 localIndex{0};
};

} // namespace lpl::ecs

// -------------------------------------------------------------------------- //
//  std::hash specialisation                                                  //
// -------------------------------------------------------------------------- //
template <>
struct std::hash<lpl::ecs::EntityId>
{
    [[nodiscard]] std::size_t operator()(lpl::ecs::EntityId id) const noexcept
    {
        return std::hash<lpl::core::u32>{}(id.raw());
    }
};

#endif // LPL_ECS_ENTITY_HPP
