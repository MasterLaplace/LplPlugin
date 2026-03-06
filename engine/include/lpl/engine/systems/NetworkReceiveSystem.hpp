/**
 * @file NetworkReceiveSystem.hpp
 * @brief ECS system that polls the transport layer and fills typed event queues.
 *
 * Runs at the very beginning of each tick (SchedulePhase::Input).
 * Reads raw packets from the ITransport, parses the protocol header,
 * and dispatches deserialized events into the appropriate TypedQueue.
 *
 * Used by both server and client.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_NETWORKRECEIVESYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_NETWORKRECEIVESYSTEM_HPP

#    include <lpl/ecs/System.hpp>
#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class NetworkReceiveSystem
 * @brief Polls the transport layer and deserializes packets into typed queues.
 */
class NetworkReceiveSystem final : public ecs::ISystem {
public:
    /**
     * @param transport Shared transport (KernelTransport or SocketTransport).
     * @param queues    Event queues to populate.
     */
    NetworkReceiveSystem(std::shared_ptr<net::transport::ITransport> transport, EventQueues &queues);
    ~NetworkReceiveSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_NETWORKRECEIVESYSTEM_HPP
