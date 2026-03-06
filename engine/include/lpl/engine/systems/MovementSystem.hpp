/**
 * @file MovementSystem.hpp
 * @brief ECS system that computes entity velocity from inputs + neural modulation.
 *
 * Iterates all entities with input state and applies WASD movement
 * with neural concentration modulation (scale [0.70x..1.30x]) and
 * blink-based jumping (rising-edge detection, grounded-gated).
 *
 * Used by both server (authoritative) and client (prediction).
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_MOVEMENTSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_MOVEMENTSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/input/InputManager.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class MovementSystem
 * @brief Computes velocity from WASD inputs + neural modulation.
 */
class MovementSystem final : public ecs::ISystem {
public:
    /**
     * @param inputManager Per-entity input state.
     * @param registry     Entity registry for iteration.
     */
    MovementSystem(input::InputManager &inputManager, ecs::Registry &registry);
    ~MovementSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_MOVEMENTSYSTEM_HPP
