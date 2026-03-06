/**
 * @file WelcomeSystem.hpp
 * @brief ECS system that processes server welcome messages (client-side).
 *
 * Drains WelcomeEvent queue and updates connection state.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_WELCOMESYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_WELCOMESYSTEM_HPP

#    include <lpl/ecs/System.hpp>
#    include <lpl/engine/EventQueue.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class WelcomeSystem
 * @brief Processes WelcomeEvents from the server (client).
 */
class WelcomeSystem final : public ecs::ISystem {
public:
    /**
     * @param queues       Event queues (reads welcomes).
     * @param myEntityId   Output: set when welcome is received.
     * @param connected    Output: set to true on successful connection.
     */
    WelcomeSystem(EventQueues &queues, core::u32 &myEntityId, bool &connected);
    ~WelcomeSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_WELCOMESYSTEM_HPP
