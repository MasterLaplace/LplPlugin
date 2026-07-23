/**
 * @file PacketDispatch.cpp
 * @brief Shared packet decoding implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-22
 * @copyright MIT License
 */

#include <lpl/engine/PacketDispatch.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/net/protocol/Bitstream.hpp>

#    include <cstring>

namespace lpl::engine::detail {

bool parsePacket(std::span<const core::byte> datagram, net::protocol::PacketHeader &outHeader,
                 std::span<const core::byte> &outPayload)
{
    if (datagram.size() < sizeof(net::protocol::PacketHeader))
        return false;

    std::memcpy(&outHeader, datagram.data(), sizeof(outHeader));
    if (outHeader.magic != net::protocol::kProtocolMagic)
        return false;

    outPayload = datagram.subspan(sizeof(outHeader));
    return true;
}

void dispatchPacket(const net::protocol::PacketHeader &header, std::span<const core::byte> payloadSpan,
                    const net::Endpoint &fromAddr, EventQueues &queues)
{
    const core::byte *payload = payloadSpan.data();
    const auto payloadSize = static_cast<core::u32>(payloadSpan.size());
    (void) payload;
    (void) payloadSize;

    switch (header.type)
    {
    case net::protocol::PacketType::Handshake: {
        // The address the packet actually came from is authoritative; the
        // handshake payload's self-reported address is not consulted.
        ConnectEvent ev{};
        ev.source = fromAddr;
        queues.connects.push(ev);
        break;
    }

    case net::protocol::PacketType::HandshakeAck: {
        WelcomeEvent ev{};
        if (payloadSize >= 4)
        {
            std::memcpy(&ev.entityId, payload, 4);
        }
        queues.welcomes.push(ev);
        break;
    }

    case net::protocol::PacketType::StateSnapshot: {
        StateUpdateEvent ev{};
        net::protocol::Bitstream stream{
            std::span<const core::byte>{payload, payloadSize},
             payloadSize * 8
        };

        auto countResult = stream.readU16();
        if (!countResult.has_value())
            break;

        const core::u16 count = countResult.value();
        for (core::u16 e = 0; e < count; ++e)
        {
            auto rId = stream.readU32();
            auto rPosX = stream.readFloat();
            auto rPosY = stream.readFloat();
            auto rPosZ = stream.readFloat();
            auto rSzX = stream.readFloat();
            auto rSzY = stream.readFloat();
            auto rSzZ = stream.readFloat();
            auto rHp = stream.readI32();

            if (!rId.has_value() || !rHp.has_value())
                break;

            StateEntity se{};
            se.id = rId.value();
            se.pos = {rPosX.value(), rPosY.value(), rPosZ.value()};
            se.size = {rSzX.value(), rSzY.value(), rSzZ.value()};
            se.hp = rHp.value();
            ev.entities.push_back(se);
        }
        queues.states.push(std::move(ev));
        break;
    }

    case net::protocol::PacketType::StateHashReport: {
        net::protocol::Bitstream stream{
            std::span<const core::byte>{payload, payloadSize},
             payloadSize * 8
        };

        // tick and digest are 64-bit; the Bitstream tops out at 32, so each goes
        // out as an explicit high/low pair rather than widening its API.
        auto rTickHigh = stream.readU32();
        auto rTickLow = stream.readU32();
        auto rDigestHigh = stream.readU32();
        auto rDigestLow = stream.readU32();
        if (!rTickHigh.has_value() || !rTickLow.has_value() || !rDigestHigh.has_value() || !rDigestLow.has_value())
            break;

        StateHashReportEvent ev{};
        ev.source = fromAddr;
        ev.tick = (static_cast<core::u64>(rTickHigh.value()) << 32) | rTickLow.value();
        ev.digest = (static_cast<core::u64>(rDigestHigh.value()) << 32) | rDigestLow.value();
        queues.stateHashReports.push(std::move(ev));
        break;
    }

    case net::protocol::PacketType::InputPayload: {
        InputEvent ev{};
        net::protocol::Bitstream stream{
            std::span<const core::byte>{payload, payloadSize},
             payloadSize * 8
        };

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
                auto rKey = stream.readU16();
                auto rPressed = stream.readBool();
                if (!rKey.has_value() || !rPressed.has_value())
                    break;

                KeyInput ki{};
                ki.key = rKey.value();
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
                auto rValue = stream.readFloat();
                if (!rAxisId.has_value() || !rValue.has_value())
                    break;

                AxisInput ai{};
                ai.axisId = rAxisId.value();
                ai.value = rValue.value();
                ev.axes.push_back(ai);
            }
        }

        // Neural data (3 floats + 1 bool)
        auto rAlpha = stream.readFloat();
        auto rBeta = stream.readFloat();
        auto rConc = stream.readFloat();
        auto rBlink = stream.readBool();
        if (rAlpha.has_value() && rBeta.has_value() && rConc.has_value() && rBlink.has_value())
        {
            ev.hasNeural = true;
            ev.neural.alpha = rAlpha.value();
            ev.neural.beta = rBeta.value();
            ev.neural.concentration = rConc.value();
            ev.neural.blink = rBlink.value();
        }

        queues.inputs.push(std::move(ev));
        break;
    }

    default: break;
    }
}

} // namespace lpl::engine::detail

#endif // LPL_HAS_NET
