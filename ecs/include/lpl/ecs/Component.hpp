/**
 * @file Component.hpp
 * @brief Component metadata types for the ECS.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_COMPONENT_HPP
#    define LPL_ECS_COMPONENT_HPP

#    include <lpl/core/Types.hpp>

#    include <cstddef>
#    include <cstdint>

namespace lpl::ecs {

/**
 * @enum ComponentId
 * @brief Compile-time enumeration of all known component types.
 *
 * Adding a new component requires appending to this enum and registering
 * its layout in the Partition's component table.
 */
enum class ComponentId : core::u16 {
    Position = 0,
    Velocity = 1,
    Rotation = 2,
    AngularVelocity = 3,
    Mass = 4,
    AABB = 5,
    Health = 6,
    NetworkSync = 7,
    InputSnapshot = 8,
    PlayerTag = 9,
    SleepState = 10,
    BciInput = 11,

    Count
};

/**
 * @enum AccessMode
 * @brief Describes how a System accesses a component (read-only or
 *        read-write).  Used by the SystemScheduler DAG builder.
 */
enum class AccessMode : core::u8 {
    ReadOnly = 0,
    ReadWrite = 1
};

/**
 * @struct ComponentAccess
 * @brief Pair of component ID + access mode used in system descriptors.
 */
struct ComponentAccess {
    ComponentId id;
    AccessMode mode;
};

/**
 * @struct ComponentLayout
 * @brief Describes the size and alignment of a single component type.
 */
struct ComponentLayout {
    ComponentId id;
    core::u32 size;
    core::u32 alignment;
};

/**
 * @brief Returns the default size and alignment for a known component type.
 *
 * This maps ComponentId to concrete data types:
 *   Position/Velocity/AngularVelocity/AABB → Vec3<float> (12, 4)
 *   Rotation         → Quat<float> (16, 4)
 *   Mass             → float (4, 4)
 *   Health           → i32 (4, 4)
 *   NetworkSync      → u32 (4, 4)
 *   InputSnapshot    → u32 (4, 4) — entity-level input index
 *   PlayerTag        → u8 (1, 1)
 *   SleepState       → u8 (1, 1)
 *   BciInput         → 3×float (12, 4) — alpha, beta, concentration
 */
[[nodiscard]] constexpr ComponentLayout defaultLayout(ComponentId id) noexcept
{
    switch (id)
    {
    case ComponentId::Position: return {id, 12, 4};
    case ComponentId::Velocity: return {id, 12, 4};
    case ComponentId::Rotation: return {id, 16, 4};
    case ComponentId::AngularVelocity: return {id, 12, 4};
    case ComponentId::Mass: return {id, 4, 4};
    case ComponentId::AABB: return {id, 12, 4};
    case ComponentId::Health: return {id, 4, 4};
    case ComponentId::NetworkSync: return {id, 4, 4};
    case ComponentId::InputSnapshot: return {id, 4, 4};
    case ComponentId::PlayerTag: return {id, 1, 1};
    case ComponentId::SleepState: return {id, 1, 1};
    case ComponentId::BciInput: return {id, 12, 4};
    default: return {id, 4, 4};
    }
}

} // namespace lpl::ecs

#endif // LPL_ECS_COMPONENT_HPP
