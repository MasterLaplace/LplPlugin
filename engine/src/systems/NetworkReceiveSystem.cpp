/**
 * @file NetworkReceiveSystem.cpp
 * @brief Polls transport and dispatches packets into typed event queues.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/core/Log.hpp>
#include <lpl/engine/PacketDispatch.hpp>
#include <lpl/engine/systems/NetworkReceiveSystem.hpp>
#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/net/session/SessionManager.hpp>

#include <array>
#include <cstring>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor (static storage)                                               //
// ========================================================================== //

static const ecs::SystemDescriptor kNetworkReceiveDesc{"NetworkReceive", ecs::SchedulePhase::Input, {}};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct NetworkReceiveSystem::Impl {
    std::shared_ptr<net::transport::ITransport> transport;
    EventQueues &queues;

    Impl(std::shared_ptr<net::transport::ITransport> t, EventQueues &q) : transport{std::move(t)}, queues{q} {}
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

NetworkReceiveSystem::NetworkReceiveSystem(std::shared_ptr<net::transport::ITransport> transport, EventQueues &queues)
    : _impl{std::make_unique<Impl>(std::move(transport), queues)}
{
}

NetworkReceiveSystem::~NetworkReceiveSystem() = default;

const ecs::SystemDescriptor &NetworkReceiveSystem::descriptor() const noexcept { return kNetworkReceiveDesc; }

void NetworkReceiveSystem::execute(core::f32 /*dt*/)
{
    // Buffer must fit a full UDP datagram: PacketHeader (16B) + max payload.
    std::array<core::byte, sizeof(net::protocol::PacketHeader) + net::session::SessionManager::kMaxPayloadSize>
        buffer{};
    net::Endpoint fromAddr{};

    for (core::u32 i = 0; i < detail::kMaxPacketsPerTick; ++i)
    {
        auto result = _impl->transport->receive(std::span<core::byte>{buffer.data(), buffer.size()}, &fromAddr);
        if (!result.has_value())
            break;

        const core::u32 bytesRead = result.value();
        if (bytesRead == 0)
            break;

        net::protocol::PacketHeader header{};
        std::span<const core::byte> payload;
        if (!detail::parsePacket(std::span<const core::byte>{buffer.data(), bytesRead}, header, payload))
            continue;

        // A single-world host decodes everything into its one set of queues; the
        // multi-instance server picks the queues per sender instead.
        detail::dispatchPacket(header, payload, fromAddr, _impl->queues);
    }
}

} // namespace lpl::engine::systems
