/**
 * @file NetworkReceiveSystem.cpp
 * @brief Polls transport and dispatches packets into typed event queues.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/engine/systems/NetworkReceiveSystem.hpp>
#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/net/session/SessionManager.hpp>
#include <lpl/core/Log.hpp>

#include <array>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor (static storage)                                               //
// ========================================================================== //

static const ecs::SystemDescriptor kNetworkReceiveDesc{
    "NetworkReceive",
    ecs::SchedulePhase::Input,
    {}
};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct NetworkReceiveSystem::Impl
{
    std::shared_ptr<net::transport::ITransport> transport;
    EventQueues&                                queues;

    Impl(std::shared_ptr<net::transport::ITransport> t, EventQueues& q)
        : transport{std::move(t)}, queues{q} {}
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

NetworkReceiveSystem::NetworkReceiveSystem(
    std::shared_ptr<net::transport::ITransport> transport,
    EventQueues& queues)
    : _impl{std::make_unique<Impl>(std::move(transport), queues)}
{
}

NetworkReceiveSystem::~NetworkReceiveSystem() = default;

const ecs::SystemDescriptor& NetworkReceiveSystem::descriptor() const noexcept
{
    return kNetworkReceiveDesc;
}

void NetworkReceiveSystem::execute(core::f32 /*dt*/)
{
    constexpr core::u32 kMaxPacketsPerTick = 256;
    // Buffer must fit a full UDP datagram: PacketHeader (16B) + max payload (1400B)
    std::array<core::byte, sizeof(net::protocol::PacketHeader) + net::session::SessionManager::kMaxPayloadSize> buffer{};
    struct sockaddr_storage fromAddr{};

    for (core::u32 i = 0; i < kMaxPacketsPerTick; ++i)
    {
        auto result = _impl->transport->receive(
            std::span<core::byte>{buffer.data(), buffer.size()},
            &fromAddr);

        if (!result.has_value())
        {
            break;
        }

        const core::u32 bytesRead = result.value();
        if (bytesRead == 0)
        {
            break;
        }

        if (bytesRead < sizeof(net::protocol::PacketHeader))
        {
            continue;
        }

        net::protocol::PacketHeader header{};
        std::memcpy(&header, buffer.data(), sizeof(header));

        if (header.magic != net::protocol::kProtocolMagic)
        {
            continue;
        }

        const core::byte* payload = buffer.data() + sizeof(header);
        const core::u32 payloadSize = bytesRead - sizeof(header);

        switch (header.type)
        {
            case net::protocol::PacketType::Handshake:
            {
                ConnectEvent ev{};
                if (payloadSize >= 6)
                {
                    std::memcpy(&ev.srcIp, payload, 4);
                    std::memcpy(&ev.srcPort, payload + 4, 2);
                }
                // Copy raw sender address for Session storage
                const core::u32 addrLen = static_cast<core::u32>(sizeof(fromAddr));
                ev.rawAddrLen = std::min(addrLen, kMaxAddrSize);
                std::memcpy(ev.rawAddr.data(), &fromAddr, ev.rawAddrLen);
                _impl->queues.connects.push(ev);
                break;
            }

            case net::protocol::PacketType::HandshakeAck:
            {
                WelcomeEvent ev{};
                if (payloadSize >= 4)
                {
                    std::memcpy(&ev.entityId, payload, 4);
                }
                _impl->queues.welcomes.push(ev);
                break;
            }

            case net::protocol::PacketType::StateSnapshot:
            {
                StateUpdateEvent ev{};
                net::protocol::Bitstream stream{
                    std::span<const core::byte>{payload, payloadSize},
                    payloadSize * 8};

                auto countResult = stream.readU16();
                if (!countResult.has_value())
                    break;

                const core::u16 count = countResult.value();
                for (core::u16 e = 0; e < count; ++e)
                {
                    auto rId   = stream.readU32();
                    auto rPosX = stream.readFloat();
                    auto rPosY = stream.readFloat();
                    auto rPosZ = stream.readFloat();
                    auto rSzX  = stream.readFloat();
                    auto rSzY  = stream.readFloat();
                    auto rSzZ  = stream.readFloat();
                    auto rHp   = stream.readI32();

                    if (!rId.has_value() || !rHp.has_value())
                        break;

                    StateEntity se{};
                    se.id     = rId.value();
                    se.pos    = {rPosX.value(), rPosY.value(), rPosZ.value()};
                    se.size   = {rSzX.value(), rSzY.value(), rSzZ.value()};
                    se.hp     = rHp.value();
                    ev.entities.push_back(se);
                }
                _impl->queues.states.push(std::move(ev));
                break;
            }

            case net::protocol::PacketType::InputPayload:
            {
                InputEvent ev{};
                net::protocol::Bitstream stream{
                    std::span<const core::byte>{payload, payloadSize},
                    payloadSize * 8};

                auto rEntityId = stream.readU32();
                if (!rEntityId.has_value())
                    break;
                ev.entityId = rEntityId.value();

                auto rKeyCount = stream.readU16();
                if (rKeyCount.has_value())
                {
                    const core::u16 keyCount = rKeyCount.value();
                    for (core::u16 k = 0; k < keyCount; ++k)
                    {
                        auto rKey     = stream.readU16();
                        auto rPressed = stream.readBool();
                        if (!rKey.has_value() || !rPressed.has_value())
                            break;

                        KeyInput ki{};
                        ki.key     = rKey.value();
                        ki.pressed = rPressed.value();
                        ev.keys.push_back(ki);
                    }
                }

                auto rAxisCount = stream.readU8();
                if (rAxisCount.has_value())
                {
                    const core::u8 axisCount = rAxisCount.value();
                    for (core::u8 a = 0; a < axisCount; ++a)
                    {
                        auto rAxisId = stream.readU8();
                        auto rValue  = stream.readFloat();
                        if (!rAxisId.has_value() || !rValue.has_value())
                            break;

                        AxisInput ai{};
                        ai.axisId = rAxisId.value();
                        ai.value  = rValue.value();
                        ev.axes.push_back(ai);
                    }
                }

                // Neural data (3 floats + 1 bool)
                auto rAlpha = stream.readFloat();
                auto rBeta  = stream.readFloat();
                auto rConc  = stream.readFloat();
                auto rBlink = stream.readBool();
                if (rAlpha.has_value() && rBeta.has_value() &&
                    rConc.has_value()  && rBlink.has_value())
                {
                    ev.hasNeural = true;
                    ev.neural.alpha         = rAlpha.value();
                    ev.neural.beta          = rBeta.value();
                    ev.neural.concentration = rConc.value();
                    ev.neural.blink         = rBlink.value();
                }

                _impl->queues.inputs.push(std::move(ev));
                break;
            }

            default:
                break;
        }
    }
}

} // namespace lpl::engine::systems
