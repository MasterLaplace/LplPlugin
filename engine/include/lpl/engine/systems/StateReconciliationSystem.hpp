/**
 * @file StateReconciliationSystem.hpp
 * @brief ECS system that applies authoritative state updates (client-side).
 *
 * Drains StateUpdateEvent queue and reconciles entity positions,
 * creating missing entities as needed.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_STATERECONCILIATIONSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_STATERECONCILIATIONSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/engine/EventQueue.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class StateReconciliationSystem
 * @brief Applies authoritative state and creates missing entities (client).
 */
class StateReconciliationSystem final : public ecs::ISystem {
public:
    /**
     * @param queues   Event queues (reads states).
     * @param world    World partition for entity lookup and insertion.
     * @param registry Registry for entity management.
     */
    StateReconciliationSystem(EventQueues &queues, ecs::WorldPartition &world, ecs::Registry &registry);
    ~StateReconciliationSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_STATERECONCILIATIONSYSTEM_HPP
