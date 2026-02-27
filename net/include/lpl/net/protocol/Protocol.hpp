/**
 * @file Protocol.hpp
 * @brief Wire protocol constants, packet types, and header layout.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_PROTOCOL_HPP
    #define LPL_NET_PROTOCOL_PROTOCOL_HPP

#include <lpl/core/Types.hpp>

namespace lpl::net::protocol {

/**
 * @brief Magic bytes identifying LPL packets on the wire.
 */
static constexpr core::u32 kProtocolMagic = 0x4C504C00;

/** @brief Current protocol version. */
static constexpr core::u8 kProtocolVersion = 1;

/**
 * @enum PacketType
 * @brief Exhaustive list of packet types understood by client and server.
 */
enum class PacketType : core::u8
{
    Handshake          = 0x01,
    HandshakeAck       = 0x02,
    Disconnect         = 0x03,
    Ping               = 0x04,
    Pong               = 0x05,
    InputPayload       = 0x10,
    StateSnapshot      = 0x11,
    StateDelta         = 0x12,
    EntitySpawn        = 0x20,
    EntityDestroy      = 0x21,
    ComponentUpdate    = 0x22,
    RollbackRequest    = 0x30,
    RollbackAck        = 0x31,
    BciPayload         = 0x40,
    Custom             = 0xFF
};

/**
 * @struct PacketHeader
 * @brief Fixed-size header prepended to every packet.
 *
 * Layout (16 bytes):
 *   [magic:4][version:1][type:1][flags:1][pad:1][seq:4][payloadSize:4]
 */
struct PacketHeader
{
    core::u32  magic;
    core::u8   version;
    PacketType type;
    core::u8   flags;
    core::u8   padding;
    core::u32  sequence;
    core::u32  payloadSize;
};

static_assert(sizeof(PacketHeader) == 16, "PacketHeader must be 16 bytes");

/**
 * @enum PacketFlag
 * @brief Bit-flags stored in PacketHeader::flags.
 */
enum class PacketFlag : core::u8
{
    None       = 0x00,
    Reliable   = 0x01,
    Compressed = 0x02,
    Encrypted  = 0x04,
    Fragment   = 0x08
};

[[nodiscard]] inline constexpr core::u8 operator|(PacketFlag a, PacketFlag b) noexcept
{
    return static_cast<core::u8>(static_cast<core::u8>(a) | static_cast<core::u8>(b));
}

[[nodiscard]] inline constexpr bool hasFlag(core::u8 flags, PacketFlag f) noexcept
{
    return (flags & static_cast<core::u8>(f)) != 0;
}

} // namespace lpl::net::protocol

#endif // LPL_NET_PROTOCOL_PROTOCOL_HPP
