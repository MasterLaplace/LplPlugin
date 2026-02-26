// /////////////////////////////////////////////////////////////////////////////
/// @file lpl_protocol.h
/// @brief Shared kernel ↔ userspace protocol definitions (C17).
///
/// This header is included by both the Linux kernel module (lpl_kmod.c)
/// and the userspace KernelTransport. It must remain pure C17.
// /////////////////////////////////////////////////////////////////////////////
#ifndef LPL_PROTOCOL_H
#define LPL_PROTOCOL_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

#define LPL_DEVICE_NAME     "lpl0"
#define LPL_DEVICE_PATH     "/dev/lpl0"
#define LPL_MAGIC           0x4C504C00U  /* "LPL\0" */

#define LPL_MAX_PACKET_SIZE 256U
#define LPL_RING_SLOTS      4096U

// ─────────────────────────────────────────────────────────────────────────────
// ioctl commands
// ─────────────────────────────────────────────────────────────────────────────

#define LPL_IOC_MAGIC       'L'

#define LPL_IOCTL_RESET     _IO(LPL_IOC_MAGIC, 0)
#define LPL_IOCTL_GET_STATS _IOR(LPL_IOC_MAGIC, 1, struct lpl_stats)
#define LPL_IOCTL_SET_PRIO  _IOW(LPL_IOC_MAGIC, 2, uint32_t)

// ─────────────────────────────────────────────────────────────────────────────
// Packet types (mirror net/protocol/Protocol.hpp in C)
// ─────────────────────────────────────────────────────────────────────────────

enum lpl_packet_type
{
    LPL_PKT_CONNECT_REQ    = 0x01,
    LPL_PKT_CONNECT_ACK    = 0x02,
    LPL_PKT_DISCONNECT     = 0x03,
    LPL_PKT_HEARTBEAT      = 0x04,
    LPL_PKT_INPUT          = 0x10,
    LPL_PKT_STATE_DELTA    = 0x20,
    LPL_PKT_STATE_FULL     = 0x21,
    LPL_PKT_NEURAL_INPUT   = 0x30,
};

// ─────────────────────────────────────────────────────────────────────────────
// Wire header  — 16 bytes, matches PacketHeader in Protocol.hpp
// ─────────────────────────────────────────────────────────────────────────────

struct lpl_packet_header
{
    uint32_t magic;
    uint16_t version;
    uint8_t  type;
    uint8_t  flags;
    uint32_t sequence;
    uint16_t payload_size;
    uint16_t checksum;
};

_Static_assert(sizeof(struct lpl_packet_header) == 16,
               "lpl_packet_header must be exactly 16 bytes");

// ─────────────────────────────────────────────────────────────────────────────
// Ring-buffer slot for shared-memory IPC (mmap)
// ─────────────────────────────────────────────────────────────────────────────

struct lpl_ring_slot
{
    uint32_t length;
    uint8_t  data[LPL_MAX_PACKET_SIZE];
};

// ─────────────────────────────────────────────────────────────────────────────
// Stats reported via ioctl
// ─────────────────────────────────────────────────────────────────────────────

struct lpl_stats
{
    uint64_t tx_packets;
    uint64_t rx_packets;
    uint64_t tx_bytes;
    uint64_t rx_bytes;
    uint64_t drops;
};

#ifdef __cplusplus
}
#endif

#endif /* LPL_PROTOCOL_H */
