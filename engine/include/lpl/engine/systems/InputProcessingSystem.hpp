/**
 * @file InputProcessingSystem.hpp
 * @brief ECS system that deserializes network InputEvents into InputManager.
 *
 * Drains the InputEvent queue and feeds key/axis/neural data into the
 * per-entity InputManager for use by MovementSystem.
 *
 * Server-side only.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_INPUTPROCESSINGSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_INPUTPROCESSINGSYSTEM_HPP

#    include <lpl/ecs/System.hpp>
#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/input/InputManager.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class InputProcessingSystem
 * @brief Deserializes network InputEvents into InputManager (server).
 */
class InputProcessingSystem final : public ecs::ISystem {
public:
    /**
     * @param queues       Event queues (reads inputs).
     * @param inputManager Input manager to populate.
     */
    InputProcessingSystem(EventQueues &queues, input::InputManager &inputManager);
    ~InputProcessingSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_INPUTPROCESSINGSYSTEM_HPP
