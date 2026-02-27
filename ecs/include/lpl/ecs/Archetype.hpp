/**
 * @file Archetype.hpp
 * @brief Archetype definition — a unique set of component types.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_ARCHETYPE_HPP
    #define LPL_ECS_ARCHETYPE_HPP

#include <lpl/ecs/Component.hpp>
#include <lpl/core/Types.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <span>

namespace lpl::ecs {

/**
 * @class Archetype
 * @brief Describes a unique combination of component types.
 *
 * Internally a fixed-size bitset where bit N corresponds to
 * @c ComponentId(N).  Two entities sharing the same Archetype are stored
 * in the same Partition chunk(s).
 */
class Archetype final
{
public:
    static constexpr core::usize kMaxComponents =
        static_cast<core::usize>(ComponentId::Count);

    using Mask = std::bitset<kMaxComponents>;

    /** @brief Default-constructs an empty archetype. */
    constexpr Archetype() noexcept = default;

    /**
     * @brief Constructs from a list of component IDs.
     * @param ids Span of component IDs that define this archetype.
     */
    explicit Archetype(std::span<const ComponentId> ids) noexcept;

    /** @brief Adds a component to the archetype. */
    void add(ComponentId id) noexcept;

    /** @brief Removes a component from the archetype. */
    void remove(ComponentId id) noexcept;

    /** @brief Tests whether the archetype contains a given component. */
    [[nodiscard]] bool has(ComponentId id) const noexcept;

    /** @brief Tests whether this archetype is a superset of @p other. */
    [[nodiscard]] bool contains(const Archetype& other) const noexcept;

    /** @brief Returns the raw bitmask. */
    [[nodiscard]] const Mask& mask() const noexcept;

    /** @brief Returns the number of component types in this archetype. */
    [[nodiscard]] core::usize count() const noexcept;

    /** @brief Equality. */
    [[nodiscard]] bool operator==(const Archetype& other) const noexcept;

private:
    Mask _mask{};
};

} // namespace lpl::ecs

#include "Archetype.inl"

#endif // LPL_ECS_ARCHETYPE_HPP
