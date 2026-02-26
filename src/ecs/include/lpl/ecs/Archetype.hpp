// /////////////////////////////////////////////////////////////////////////////
/// @file Archetype.hpp
/// @brief Archetype definition — a unique set of component types.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/ecs/Component.hpp>
#include <lpl/core/Types.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <span>

namespace lpl::ecs {

// /////////////////////////////////////////////////////////////////////////////
/// @class Archetype
/// @brief Describes a unique combination of component types.
///
/// Internally a fixed-size bitset where bit N corresponds to
/// @c ComponentId(N).  Two entities sharing the same Archetype are stored
/// in the same Partition chunk(s).
// /////////////////////////////////////////////////////////////////////////////
class Archetype final
{
public:
    static constexpr core::usize kMaxComponents =
        static_cast<core::usize>(ComponentId::Count);

    using Mask = std::bitset<kMaxComponents>;

    /// @brief Default-constructs an empty archetype.
    constexpr Archetype() noexcept = default;

    /// @brief Constructs from a list of component IDs.
    /// @param ids Span of component IDs that define this archetype.
    explicit Archetype(std::span<const ComponentId> ids) noexcept
    {
        for (auto id : ids)
        {
            mask_.set(static_cast<core::usize>(id));
        }
    }

    /// @brief Adds a component to the archetype.
    void add(ComponentId id) noexcept
    {
        mask_.set(static_cast<core::usize>(id));
    }

    /// @brief Removes a component from the archetype.
    void remove(ComponentId id) noexcept
    {
        mask_.reset(static_cast<core::usize>(id));
    }

    /// @brief Tests whether the archetype contains a given component.
    [[nodiscard]] bool has(ComponentId id) const noexcept
    {
        return mask_.test(static_cast<core::usize>(id));
    }

    /// @brief Tests whether this archetype is a superset of @p other.
    [[nodiscard]] bool contains(const Archetype& other) const noexcept
    {
        return (mask_ & other.mask_) == other.mask_;
    }

    /// @brief Returns the raw bitmask.
    [[nodiscard]] const Mask& mask() const noexcept { return mask_; }

    /// @brief Returns the number of component types in this archetype.
    [[nodiscard]] core::usize count() const noexcept { return mask_.count(); }

    /// @brief Equality.
    [[nodiscard]] bool operator==(const Archetype& other) const noexcept
    {
        return mask_ == other.mask_;
    }

private:
    Mask mask_{};
};

} // namespace lpl::ecs
