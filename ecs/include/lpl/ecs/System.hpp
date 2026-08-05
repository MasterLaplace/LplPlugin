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
#    define LPL_ECS_SYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Component.hpp>

#    include <span>
#    include <string_view>

namespace lpl::ecs {

/**
 * @enum SchedulePhase
 * @brief Logical phases within a single tick, ordered by execution priority.
 */
enum class SchedulePhase : core::u8 {
    Input = 0,
    PrePhysics = 1,
    Physics = 2,
    PostPhysics = 3,
    Network = 4,
    Render = 5,

    Count
};

/**
 * @enum ResourceId
 * @brief World state a system touches that is NOT held per entity.
 *
 * A closed enumeration, exactly like @ref ComponentId and for the same reason:
 * the scheduler can only order what it can name. Everything a system reads or
 * writes used to have to be a component, so the three systems that share a
 * pheromone field declared an EMPTY access set — no edge, no order, and the
 * evaporation pass was free to run before the marks it evaporates. It happened
 * to work because a wave executes in registration order under the inline job
 * system; it would race the day a real thread pool ran a wave in parallel.
 *
 * A resource is singular per world. Two of them are the terrain: where a body
 * may stand (read) and what is standing there to eat (written when eaten).
 */
enum class ResourceId : core::u8 {
    /// Pheromone / scent field: deposited into, evaporated, read as a gradient.
    ScentField = 0,
    /// The heightfield and its walkability — whatever a World generated.
    Terrain = 1,
    /// Standing plants: eaten by grazers, regrown by the vegetation tick.
    Vegetation = 2,
    /// The spatial broad-phase index, when a world enables one.
    SpatialIndex = 3,

    Count
};

/**
 * @struct ResourceAccess
 * @brief Pair of resource ID + access mode used in system descriptors.
 */
struct ResourceAccess {
    ResourceId id;
    AccessMode mode;
};

/**
 * @struct SystemDescriptor
 * @brief Declares a system's identity, phase, and data dependencies.
 *
 * The SystemScheduler uses these descriptors to build a DAG and detect
 * data hazards at registration time rather than runtime. Dependencies come in
 * two flavours — per-entity @ref ComponentAccess and world-level
 * @ref ResourceAccess — and both produce the same ordering edges.
 */
struct SystemDescriptor {
    std::string_view name;
    SchedulePhase phase;
    std::span<const ComponentAccess> accesses;
    /// Empty for a system that only touches components.
    std::span<const ResourceAccess> resources{};
};

/**
 * @class ISystem
 * @brief Abstract base for all ECS systems.
 *
 * Implementations override @ref descriptor to declare metadata and
 * @ref execute to perform per-tick logic.
 */
class ISystem {
public:
    virtual ~ISystem() = default;

    /** @brief Returns the static descriptor for this system. */
    [[nodiscard]] virtual const SystemDescriptor &descriptor() const noexcept = 0;

    /**
     * @brief Executes the system logic for one tick.
     * @param dt Fixed delta-time in seconds (typically 1/144).
     */
    virtual void execute(core::f32 dt) = 0;
};

} // namespace lpl::ecs

#endif // LPL_ECS_SYSTEM_HPP
