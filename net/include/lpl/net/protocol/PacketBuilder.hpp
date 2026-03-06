/**
 * @file PacketBuilder.hpp
 * @brief Helper functions for constructing typed protocol packets.
 *
 * Mirrors legacy Network::send_connect(), send_welcome(), send_inputs().
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_PACKETBUILDER_HPP
#    define LPL_NET_PROTOCOL_PACKETBUILDER_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/net/protocol/Protocol.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <cstring>
#    include <span>
#    include <vector>

namespace lpl::net::protocol {

/**
 * @brief Builds a raw header + payload buffer for the given packet type.
 * @param type    Packet type.
 * @param payload Payload bytes.
 * @param seq     Sequence number.
 * @param flags   Packet flags.
 * @return The raw packet bytes (header + payload).
 */
[[nodiscard]] inline std::vector<core::byte> buildPacket(PacketType type, std::span<const core::byte> payload,
                                                         core::u32 seq = 0, core::u8 flags = 0)
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

/**
 * @brief Builds and sends a Handshake (connect) packet.
 * @param transport Transport to send through.
 * @param address   Server address (sockaddr_in*).
 * @param localIp   Client's own IP (network byte order).
 * @param localPort Client's own port (network byte order).
 * @return Ok or error.
 */
[[nodiscard]] inline core::Expected<core::u32> sendConnect(transport::ITransport &transport, const void *address,
                                                           core::u32 localIp = 0, core::u16 localPort = 0)
{
    core::byte payload[6]{};
    std::memcpy(payload, &localIp, 4);
    std::memcpy(payload + 4, &localPort, 2);

    auto pkt = buildPacket(PacketType::Handshake, {payload, 6});
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

/**
 * @brief Builds and sends a HandshakeAck (welcome) packet.
 * @param transport Transport to send through.
 * @param address   Client address.
 * @param entityId  Entity ID assigned to the client.
 * @return Ok or error.
 */
[[nodiscard]] inline core::Expected<core::u32> sendWelcome(transport::ITransport &transport, const void *address,
                                                           core::u32 entityId)
{
    core::byte payload[4]{};
    std::memcpy(payload, &entityId, 4);

    auto pkt = buildPacket(PacketType::HandshakeAck, {payload, 4});
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

/**
 * @brief Builds and sends an InputPayload packet.
 * @param transport Transport to send through.
 * @param address   Server address.
 * @param rawInputPayload Pre-serialized input data.
 * @param seq       Sequence number.
 * @return Ok or error.
 */
[[nodiscard]] inline core::Expected<core::u32> sendInputs(transport::ITransport &transport, const void *address,
                                                          std::span<const core::byte> rawInputPayload,
                                                          core::u32 seq = 0)
{
    auto pkt = buildPacket(PacketType::InputPayload, rawInputPayload, seq);
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

/**
 * @brief Builds and sends a StateSnapshot packet.
 * @param transport Transport to send through.
 * @param address   Client address.
 * @param stateData Pre-serialized state data.
 * @param seq       Sequence number.
 * @return Ok or error.
 */
[[nodiscard]] inline core::Expected<core::u32> sendState(transport::ITransport &transport, const void *address,
                                                         std::span<const core::byte> stateData, core::u32 seq = 0)
{
    auto pkt = buildPacket(PacketType::StateSnapshot, stateData, seq, static_cast<core::u8>(PacketFlag::Fragment));
    return transport.send(std::span<const core::byte>{pkt.data(), pkt.size()}, address);
}

} // namespace lpl::net::protocol

#endif // LPL_NET_PROTOCOL_PACKETBUILDER_HPP
