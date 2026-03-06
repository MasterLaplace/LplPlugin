/**
 * @file BroadcastSystem.hpp
 * @brief ECS system that serializes and broadcasts world state to clients.
 *
 * Runs in the Network phase (after physics and buffer swap).
 * Reads the stable read-buffer and sends the state to all connected
 * clients via the SessionManager.
 *
 * Server-side only.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_BROADCASTSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_BROADCASTSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class BroadcastSystem
 * @brief Serializes world state and broadcasts to all clients (server).
 */
class BroadcastSystem final : public ecs::ISystem {
public:
    /**
     * @param sessionManager Active sessions to broadcast to.
     * @param transport      Transport for sending.
     * @param world          World partition for entity data.
     * @param registry       Registry for entity iteration.
     */
    BroadcastSystem(net::session::SessionManager &sessionManager, std::shared_ptr<net::transport::ITransport> transport,
                    ecs::WorldPartition &world, ecs::Registry &registry);
    ~BroadcastSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_BROADCASTSYSTEM_HPP
