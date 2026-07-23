/**
 * @file PacketBuilder.cpp
 * @brief Typed protocol packet builders.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/net/protocol/PacketBuilder.hpp>

namespace lpl::net::protocol {

std::vector<core::byte> buildPacket(PacketType type, std::span<const core::byte> payload, core::u32 seq, core::u8 flags)
{
    PacketHeader header{};
    header.magic = kProtocolMagic;
    header.version = kProtocolVersion;
    header.type = type;
    header.flags = flags;
    header.padding = 0;
    header.sequence = seq;
    header.payloadSize = static_cast<core::u32>(payload.size());

    std::vector<core::byte> packet(sizeof(header) + payload.size());
    std::memcpy(packet.data(), &header, sizeof(header));
    if (!payload.empty())
        std::memcpy(packet.data() + sizeof(header), payload.data(), payload.size());

    return packet;
}

core::Expected<core::u32> sendConnect(transport::ITransport &transport, const Endpoint *address, core::u32 localIp,
                                      core::u16 localPort)
{
    core::byte payload[6]{};
    std::memcpy(payload, &localIp, 4);
    std::memcpy(payload + 4, &localPort, 2);

    auto pkt = buildPacket(PacketType::Handshake, {payload, 6});
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

core::Expected<core::u32> sendWelcome(transport::ITransport &transport, const Endpoint *address, core::u32 entityId)
{
    core::byte payload[4]{};
    std::memcpy(payload, &entityId, 4);

    auto pkt = buildPacket(PacketType::HandshakeAck, {payload, 4});
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

core::Expected<core::u32> sendInputs(transport::ITransport &transport, const Endpoint *address,
                                     std::span<const core::byte> rawInputPayload, core::u32 seq)
{
    auto pkt = buildPacket(PacketType::InputPayload, rawInputPayload, seq);
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

core::Expected<core::u32> sendState(transport::ITransport &transport, const Endpoint *address,
                                    std::span<const core::byte> stateData, core::u32 seq)
{
    auto pkt = buildPacket(PacketType::StateSnapshot, stateData, seq, static_cast<core::u8>(PacketFlag::Fragment));
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

core::Expected<core::u32> sendStateHashReport(transport::ITransport &transport, const Endpoint *address, core::u64 tick,
                                              core::u64 digest, core::u32 seq)
{
    // Two 32-bit halves each, high first — the Bitstream has no 64-bit width and
    // this keeps the wire layout explicit. PacketDispatch reads them back in the
    // same order.
    Bitstream stream;
    stream.writeU32(static_cast<core::u32>(tick >> 32));
    stream.writeU32(static_cast<core::u32>(tick & 0xFFFFFFFFULL));
    stream.writeU32(static_cast<core::u32>(digest >> 32));
    stream.writeU32(static_cast<core::u32>(digest & 0xFFFFFFFFULL));

    const auto payload = stream.data();
    auto pkt = buildPacket(PacketType::StateHashReport, payload, seq);
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

} // namespace lpl::net::protocol
