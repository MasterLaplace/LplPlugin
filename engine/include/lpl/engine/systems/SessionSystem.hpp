/**
 * @file SessionSystem.hpp
 * @brief ECS system that handles client connections (server-side).
 *
 * Drains ConnectEvent queue, creates player entities, sends MSG_WELCOME,
 * and registers the client endpoint for future broadcasts.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_SESSIONSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_SESSIONSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/input/InputManager.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class SessionSystem
 * @brief Processes connection events and creates player entities (server).
 */
class SessionSystem final : public ecs::ISystem {
public:
    /**
     * @param sessionManager Session registry.
     * @param queues         Event queues (reads connects).
     * @param transport      Transport to send welcome packets.
     * @param inputManager   Input manager to create per-entity input state.
     * @param world          World partition to spawn entities.
     * @param registry       ECS registry for entity creation.
     */
    SessionSystem(net::session::SessionManager &sessionManager, EventQueues &queues,
                  std::shared_ptr<net::transport::ITransport> transport, input::InputManager &inputManager,
                  ecs::WorldPartition &world, ecs::Registry &registry);
    ~SessionSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_SESSIONSYSTEM_HPP
