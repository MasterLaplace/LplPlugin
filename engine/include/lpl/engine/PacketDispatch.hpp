/**
 * @file PacketDispatch.hpp
 * @brief Shared decoding of received packets into typed event queues.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_PACKETDISPATCH_HPP
#    define LPL_ENGINE_PACKETDISPATCH_HPP

#    include <lpl/core/Platform.hpp>

#    ifdef LPL_HAS_NET

#        include <lpl/engine/EventQueue.hpp>
#        include <lpl/net/Endpoint.hpp>
#        include <lpl/net/protocol/Protocol.hpp>

#        include <span>

namespace lpl::engine::detail {

/** @brief Packets drained from a transport in a single tick, at most. */
inline constexpr core::u32 kMaxPacketsPerTick = 256;

/**
 * @brief Validate a received datagram and split it into header + payload.
 * @param datagram The bytes as received.
 * @param outHeader Filled on success.
 * @param outPayload Set to the bytes after the header, on success.
 * @return false if the datagram is too short or is not one of ours.
 */
[[nodiscard]] bool parsePacket(std::span<const core::byte> datagram, net::protocol::PacketHeader &outHeader,
                               std::span<const core::byte> &outPayload);

/**
 * @brief Decode one validated packet into @p queues.
 *
 * Shared by the single-world receive system and by the multi-instance server,
 * which differ only in WHICH queues a packet is decoded into — the server picks
 * them per sender so one instance's inputs never leak into another's.
 *
 * @param header Parsed packet header.
 * @param payload Bytes after the header.
 * @param source Address the datagram came from (authoritative for handshakes).
 * @param queues Destination queues.
 */
void dispatchPacket(const net::protocol::PacketHeader &header, std::span<const core::byte> payload,
                    const net::Endpoint &source, EventQueues &queues);

} // namespace lpl::engine::detail

#    endif // LPL_HAS_NET

#endif // LPL_ENGINE_PACKETDISPATCH_HPP
