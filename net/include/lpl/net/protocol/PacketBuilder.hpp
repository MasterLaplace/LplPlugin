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
#    include <lpl/net/protocol/Bitstream.hpp>
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
[[nodiscard]] std::vector<core::byte> buildPacket(PacketType type, std::span<const core::byte> payload, core::u32 seq = 0,
                                                  core::u8 flags = 0);

/**
 * @brief Builds and sends a Handshake (connect) packet.
 * @param transport Transport to send through.
 * @param address   Server address, or nullptr for the transport default.
 * @param localIp   Client's own IP (network byte order).
 * @param localPort Client's own port (network byte order).
 * @return Ok or error.
 */
[[nodiscard]] core::Expected<core::u32> sendConnect(transport::ITransport &transport, const Endpoint *address,
                                                    core::u32 localIp = 0, core::u16 localPort = 0);

/**
 * @brief Builds and sends a HandshakeAck (welcome) packet.
 * @param transport Transport to send through.
 * @param address   Client address.
 * @param entityId  Entity ID assigned to the client.
 * @return Ok or error.
 */
[[nodiscard]] core::Expected<core::u32> sendWelcome(transport::ITransport &transport, const Endpoint *address,
                                                    core::u32 entityId);

/**
 * @brief Builds and sends an InputPayload packet.
 * @param transport Transport to send through.
 * @param address   Server address.
 * @param rawInputPayload Pre-serialized input data.
 * @param seq       Sequence number.
 * @return Ok or error.
 */
[[nodiscard]] core::Expected<core::u32> sendInputs(transport::ITransport &transport, const Endpoint *address,
                                                   std::span<const core::byte> rawInputPayload, core::u32 seq = 0);

/**
 * @brief Builds and sends a StateSnapshot packet.
 * @param transport Transport to send through.
 * @param address   Client address.
 * @param stateData Pre-serialized state data.
 * @param seq       Sequence number.
 * @return Ok or error.
 */
[[nodiscard]] core::Expected<core::u32> sendState(transport::ITransport &transport, const Endpoint *address,
                                                  std::span<const core::byte> stateData, core::u32 seq = 0);

/**
 * @brief Builds and sends a StateHashReport packet (client -> server).
 *
 * The client tells the server what its authoritative state hashed to at a tick
 * it has already simulated; the server compares against its own digest for that
 * tick and answers Match / Diverged / TickUnknown. See the book's §6.4.
 *
 * @param transport Transport to send through.
 * @param address   Server address.
 * @param tick      The tick the client hashed.
 * @param digest    engine::World::stateHash() at that tick.
 * @param seq       Sequence number.
 * @return Ok or error.
 */
[[nodiscard]] core::Expected<core::u32> sendStateHashReport(transport::ITransport &transport, const Endpoint *address,
                                                            core::u64 tick, core::u64 digest, core::u32 seq = 0);

} // namespace lpl::net::protocol

#endif // LPL_NET_PROTOCOL_PACKETBUILDER_HPP
