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
    #define LPL_ECS_COMPONENT_HPP

#include <lpl/core/Types.hpp>

#include <cstddef>
#include <cstdint>

namespace lpl::ecs {

/**
 * @enum ComponentId
 * @brief Compile-time enumeration of all known component types.
 *
 * Adding a new component requires appending to this enum and registering
 * its layout in the Partition's component table.
 */
enum class ComponentId : core::u16
{
    Position       = 0,
    Velocity       = 1,
    Rotation       = 2,
    AngularVelocity= 3,
    Mass           = 4,
    AABB           = 5,
    Health         = 6,
    NetworkSync    = 7,
    InputSnapshot  = 8,
    PlayerTag      = 9,
    SleepState     = 10,
    BciInput       = 11,

    Count
};

/**
 * @enum AccessMode
 * @brief Describes how a System accesses a component (read-only or
 *        read-write).  Used by the SystemScheduler DAG builder.
 */
enum class AccessMode : core::u8
{
    ReadOnly  = 0,
    ReadWrite = 1
};

/**
 * @struct ComponentAccess
 * @brief Pair of component ID + access mode used in system descriptors.
 */
struct ComponentAccess
{
    ComponentId id;
    AccessMode  mode;
};

/**
 * @struct ComponentLayout
 * @brief Describes the size and alignment of a single component type.
 */
struct ComponentLayout
{
    ComponentId   id;
    core::u32     size;
    core::u32     alignment;
};

} // namespace lpl::ecs

#endif // LPL_ECS_COMPONENT_HPP
