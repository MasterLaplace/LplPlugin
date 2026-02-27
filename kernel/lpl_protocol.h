/**
 * @file lpl_protocol.h
 * @brief Shared kernel ↔ userspace protocol definitions (C17).
 *
 * This header is included by both the Linux kernel module (lpl_kmod.c)
 * and the userspace KernelTransport. It must remain pure C17.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */
#ifndef LPL_PROTOCOL_H
#define LPL_PROTOCOL_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <asm/barrier.h>
#else
#include <stdint.h>
#include <stddef.h>

/* ── Userspace memory barriers (match kernel smp_*) ─────────────────────── */
#define smp_load_acquire(p)      __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define smp_store_release(p, v)  __atomic_store_n(p, v, __ATOMIC_RELEASE)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Device constants ──────────────────────────────────────────────────── */

#define LPL_DEVICE_NAME     "lpl0"
#define LPL_DEVICE_PATH     "/dev/lpl0"
#define LPL_CLASS_NAME      "lpl"
#define LPL_MAGIC           0x4C504C00U  /* "LPL\0" */
#define LPL_PORT            7777U

/* ─── Ring buffer sizing ────────────────────────────────────────────────── */

#define LPL_MAX_PACKET_SIZE 256U
#define LPL_RING_SLOTS      4096U        /* Power-of-2 for mask-based indexing */
#define LPL_RING_MASK       (LPL_RING_SLOTS - 1U)

/* ─── ioctl commands ────────────────────────────────────────────────────── */

#define LPL_IOC_MAGIC       'L'

#define LPL_IOCTL_RESET     _IO(LPL_IOC_MAGIC, 0)
#define LPL_IOCTL_GET_STATS _IOR(LPL_IOC_MAGIC, 1, struct lpl_stats)
#define LPL_IOCTL_SET_PRIO  _IOW(LPL_IOC_MAGIC, 2, uint32_t)
#define LPL_IOCTL_KICK_TX   _IO(LPL_IOC_MAGIC, 3)

/* ─── Component IDs (dynamic packet format) ─────────────────────────────── */

typedef enum {
    LPL_COMP_TRANSFORM = 1,
    LPL_COMP_HEALTH    = 2,
    LPL_COMP_VELOCITY  = 3,
    LPL_COMP_MASS      = 4,
    LPL_COMP_SIZE      = 5
} LplComponentId;

/* ─── Packet types ──────────────────────────────────────────────────────── */

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

/* Legacy aliases for compatibility */
#define MSG_CONNECT  LPL_PKT_CONNECT_REQ
#define MSG_WELCOME  LPL_PKT_CONNECT_ACK
#define MSG_STATE    LPL_PKT_STATE_FULL
#define MSG_INPUTS   LPL_PKT_INPUT

#ifdef __cplusplus
#define LPL_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define LPL_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/* ─── Wire header — 16 bytes ────────────────────────────────────────────── */

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

LPL_STATIC_ASSERT(sizeof(struct lpl_packet_header) == 16,
               "lpl_packet_header must be exactly 16 bytes");

/* ─── Ring buffer structures (lockless, mmap-shared) ────────────────────── */

/**
 * @brief Ring buffer index header with cache-line padding.
 *
 * The `_pad[6]` fields ensure `head` and `tail` reside on separate
 * 32-byte boundaries, preventing false sharing between producer
 * and consumer cores.
 */
typedef struct {
    uint32_t head;
    uint32_t tail;
    uint32_t _pad[6];   /* pad to 32 bytes — prevent false sharing */
} LplRingHeader;

LPL_STATIC_ASSERT(sizeof(LplRingHeader) == 32,
               "LplRingHeader must be exactly 32 bytes");

/**
 * @brief RX packet slot (network → userspace).
 */
typedef struct {
    uint32_t src_ip;
    uint16_t src_port;
    uint16_t length;
    uint8_t  data[LPL_MAX_PACKET_SIZE];
} LplRxPacket;

/**
 * @brief TX packet slot (userspace → network).
 */
typedef struct {
    uint32_t dst_ip;
    uint16_t dst_port;
    uint16_t length;
    uint8_t  data[LPL_MAX_PACKET_SIZE];
} LplTxPacket;

/**
 * @brief RX ring buffer (Netfilter → userspace, SPSC lockless).
 */
typedef struct {
    LplRingHeader idx;
    LplRxPacket   packets[LPL_RING_SLOTS];
} LplRxRing;

/**
 * @brief TX ring buffer (userspace → kthread TX, SPSC lockless).
 */
typedef struct {
    LplRingHeader idx;
    LplTxPacket   packets[LPL_RING_SLOTS];
} LplTxRing;

/**
 * @brief Top-level shared memory layout for mmap.
 *
 * Mapped via `vmalloc_user` in kernel, `mmap` in userspace.
 * Contains both RX and TX ring buffers for bidirectional
 * zero-copy IPC.
 */
typedef struct {
    LplRxRing rx;
    LplTxRing tx;
} LplSharedMemory;

/* ─── Simple ring slot (for non-mmap fallback path) ─────────────────────── */

struct lpl_ring_slot
{
    uint32_t length;
    uint8_t  data[LPL_MAX_PACKET_SIZE];
};

/* ─── Stats reported via ioctl ──────────────────────────────────────────── */

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
