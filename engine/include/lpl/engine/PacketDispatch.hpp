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
 * - Handshake and Disconnect: the address the packet actually came from is
 *   authoritative; the payload's self-reported address, if any, is not consulted.
 * - EntitySpawn (AOI): entities that just entered the client's interest radius. Full
 *   snapshot, so the client can create them exactly as a StateSnapshot does.
 * - StateDelta (AOI): field-masked delta of entities the client already holds and that
 *   stayed in range (§6.2.5). Each entity carries only the fields that changed against
 *   what the server last sent; absent fields are left to reconciliation, which merges
 *   onto the state the client already holds. A Compressed StateDelta is a far-ring
 *   group (§6.2.6): its positions are quantized, prefixed by [posBits][extent] so the
 *   client can dequantize.
 * - EntityDestroy (AOI): entities that left the client's interest radius — ids only.
 * - StateHashReport: tick and digest are 64-bit; the Bitstream tops out at 32, so each
 *   goes out as an explicit high/low pair rather than widening its API.
 * - SnapshotAck: the client confirms it has applied every snapshot up to a sequence
 *   (§6.2.5).
 * - InputPayload: keys and axes, then the neural data (3 floats + 1 bool).
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
