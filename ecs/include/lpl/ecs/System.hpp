/**
 * @file System.hpp
 * @brief System descriptor and scheduling phase definitions.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_SYSTEM_HPP
    #define LPL_ECS_SYSTEM_HPP

#include <lpl/ecs/Component.hpp>
#include <lpl/core/Types.hpp>

#include <span>
#include <string_view>

namespace lpl::ecs {

/**
 * @enum SchedulePhase
 * @brief Logical phases within a single tick, ordered by execution priority.
 */
enum class SchedulePhase : core::u8
{
    Input       = 0,
    PrePhysics  = 1,
    Physics     = 2,
    PostPhysics = 3,
    Network     = 4,
    Render      = 5,

    Count
};

/**
 * @struct SystemDescriptor
 * @brief Declares a system's identity, phase, and component dependencies.
 *
 * The SystemScheduler uses these descriptors to build a DAG and detect
 * data hazards at registration time rather than runtime.
 */
struct SystemDescriptor
{
    std::string_view              name;
    SchedulePhase                 phase;
    std::span<const ComponentAccess> accesses;
};

/**
 * @class ISystem
 * @brief Abstract base for all ECS systems.
 *
 * Implementations override @ref descriptor to declare metadata and
 * @ref execute to perform per-tick logic.
 */
class ISystem
{
public:
    virtual ~ISystem() = default;

    /** @brief Returns the static descriptor for this system. */
    [[nodiscard]] virtual const SystemDescriptor& descriptor() const noexcept = 0;

    /**
     * @brief Executes the system logic for one tick.
     * @param dt Fixed delta-time in seconds (typically 1/144).
     */
    virtual void execute(core::f32 dt) = 0;
};

} // namespace lpl::ecs

#endif // LPL_ECS_SYSTEM_HPP
