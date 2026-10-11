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
#    include <asm/barrier.h>
#    include <linux/stddef.h>
#    include <linux/types.h>
#else
#    include <stddef.h>
#    include <stdint.h>

/* ── Userspace memory barriers (match kernel smp_*) ─────────────────────── */
#    define smp_load_acquire(p)     __atomic_load_n(p, __ATOMIC_ACQUIRE)
#    define smp_store_release(p, v) __atomic_store_n(p, v, __ATOMIC_RELEASE)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Device constants ──────────────────────────────────────────────────── */

#define LPL_DEVICE_NAME "lpl0"
#define LPL_DEVICE_PATH "/dev/lpl0"
#define LPL_CLASS_NAME  "lpl"
#define LPL_MAGIC       0x4C504C00U /* "LPL\0" */
#define LPL_PORT        7777U

/* ─── Ring buffer sizing ────────────────────────────────────────────────── */

#define LPL_MAX_PACKET_SIZE 256U
#define LPL_RING_SLOTS      4096U /* Power-of-2 for mask-based indexing */
#define LPL_RING_MASK       (LPL_RING_SLOTS - 1U)

/* ─── ioctl commands ────────────────────────────────────────────────────── */

#define LPL_IOC_MAGIC 'L'

#define LPL_IOCTL_RESET     _IO(LPL_IOC_MAGIC, 0)
#define LPL_IOCTL_GET_STATS _IOR(LPL_IOC_MAGIC, 1, struct lpl_stats)
#define LPL_IOCTL_SET_PRIO  _IOW(LPL_IOC_MAGIC, 2, uint32_t)
#define LPL_IOCTL_KICK_TX   _IO(LPL_IOC_MAGIC, 3)

/* ─── Component IDs (dynamic packet format) ─────────────────────────────── */

typedef enum {
    LPL_COMP_TRANSFORM = 1,
    LPL_COMP_HEALTH = 2,
    LPL_COMP_VELOCITY = 3,
    LPL_COMP_MASS = 4,
    LPL_COMP_SIZE = 5
} LplComponentId;

/* ─── Packet types ──────────────────────────────────────────────────────── */

enum lpl_packet_type {
    LPL_PKT_CONNECT_REQ = 0x01,
    LPL_PKT_CONNECT_ACK = 0x02,
    LPL_PKT_DISCONNECT = 0x03,
    LPL_PKT_HEARTBEAT = 0x04,
    LPL_PKT_INPUT = 0x10,
    LPL_PKT_STATE_DELTA = 0x20,
    LPL_PKT_STATE_FULL = 0x21,
    LPL_PKT_NEURAL_INPUT = 0x30,
};

/* Legacy aliases for compatibility */
#define MSG_CONNECT LPL_PKT_CONNECT_REQ
#define MSG_WELCOME LPL_PKT_CONNECT_ACK
#define MSG_STATE   LPL_PKT_STATE_FULL
#define MSG_INPUTS  LPL_PKT_INPUT

#ifdef __cplusplus
#    define LPL_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#    define LPL_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/* ─── Wire header — 16 bytes ────────────────────────────────────────────── */

struct lpl_packet_header {
    uint32_t magic;
    uint16_t version;
    uint8_t type;
    uint8_t flags;
    uint32_t sequence;
    uint16_t payload_size;
    uint16_t checksum;
};

LPL_STATIC_ASSERT(sizeof(struct lpl_packet_header) == 16, "lpl_packet_header must be exactly 16 bytes");

/* ─── Ring buffer structures (lockless, mmap-shared) ────────────────────── */

/** Version of the layout below. The module writes it in LplSharedHeader; KernelTransport refuses another. */
#define LPL_PROTOCOL_VERSION 3U

/**
 * Distance between what the two sides of a ring write: two 64-byte lines, because x86 prefetches lines
 * in 128-byte aligned pairs. Fixed rather than taken from the compiler: the module and the process may
 * be built by different compilers, and they must agree on every offset of the mapping.
 */
#define LPL_RING_INTERFERENCE_SIZE 128U

/** Alignment of a slot, so that a slot never shares a cache line with the next one. */
#define LPL_RING_SLOT_ALIGNMENT 64U

/**
 * @brief What the producer of a ring writes, alone in its interference span.
 */
typedef struct __attribute__((aligned(LPL_RING_INTERFERENCE_SIZE))) {
    uint32_t write_index; /**< Next slot to fill, free-running, published with release. */
} LplRingWriter;

/**
 * @brief What the consumer of a ring writes, alone in its interference span.
 */
typedef struct __attribute__((aligned(LPL_RING_INTERFERENCE_SIZE))) {
    uint32_t read_index; /**< Next slot to read, free-running, published with release. */
} LplRingReader;

/**
 * @brief Whether the module's sender sleeps, alone in its span: the process reads it after every
 *        publish, and the module writes it only when the sender goes to sleep or wakes up.
 */
typedef struct __attribute__((aligned(LPL_RING_INTERFERENCE_SIZE))) {
    uint32_t sleeping; /**< Nonzero while the sender sleeps: the process then kicks it with LPL_IOCTL_KICK_TX. */
} LplRingWake;

/**
 * @brief Start of the mapping, written once by the module when it loads.
 */
typedef struct __attribute__((aligned(LPL_RING_INTERFERENCE_SIZE))) {
    uint32_t magic;   /**< LPL_MAGIC. */
    uint32_t version; /**< LPL_PROTOCOL_VERSION. */
    uint32_t slots;   /**< LPL_RING_SLOTS. */
    uint32_t size;    /**< sizeof(LplSharedMemory). */
} LplSharedHeader;

/**
 * @brief RX packet slot (network → process). Address and port in host byte order.
 */
typedef struct __attribute__((aligned(LPL_RING_SLOT_ALIGNMENT))) {
    uint32_t src_ip;
    uint16_t src_port;
    uint16_t length;
    uint8_t data[LPL_MAX_PACKET_SIZE];
} LplRxPacket;

/**
 * @brief TX packet slot (process → network). Address and port in host byte order: the module converts.
 */
typedef struct __attribute__((aligned(LPL_RING_SLOT_ALIGNMENT))) {
    uint32_t dst_ip;
    uint16_t dst_port;
    uint16_t length;
    uint8_t data[LPL_MAX_PACKET_SIZE];
} LplTxPacket;

/**
 * @brief RX ring: the module's Netfilter hook produces, the process consumes.
 */
typedef struct {
    LplRingWriter writer; /**< Written by the module. */
    LplRingReader reader; /**< Written by the process. */
    LplRxPacket packets[LPL_RING_SLOTS];
} LplRxRing;

/**
 * @brief TX ring: the process produces, the module's sender consumes.
 */
typedef struct {
    LplRingWriter writer; /**< Written by the process. */
    LplRingReader reader; /**< Written by the module. */
    LplRingWake wake;     /**< Written by the module. */
    LplTxPacket packets[LPL_RING_SLOTS];
} LplTxRing;

/**
 * @brief Top-level shared memory layout for mmap.
 *
 * Mapped via `vmalloc_user` in kernel, `mmap` in userspace. Each side keeps its own index and its
 * last read of the other side's in private memory, and reads the shared one again only when its
 * copy says the ring is full or empty: a value read from the mapping is never trusted beyond the
 * slots it can address.
 */
typedef struct {
    LplSharedHeader header;
    LplRxRing rx;
    LplTxRing tx;
} LplSharedMemory;

LPL_STATIC_ASSERT(sizeof(LplRingWriter) == LPL_RING_INTERFERENCE_SIZE &&
                      sizeof(LplRingReader) == LPL_RING_INTERFERENCE_SIZE &&
                      sizeof(LplRingWake) == LPL_RING_INTERFERENCE_SIZE,
                  "each index of a ring fills exactly one interference span");
LPL_STATIC_ASSERT(sizeof(LplRxPacket) % LPL_RING_SLOT_ALIGNMENT == 0 &&
                      sizeof(LplTxPacket) % LPL_RING_SLOT_ALIGNMENT == 0,
                  "a slot never shares a cache line with the next one");
LPL_STATIC_ASSERT(offsetof(LplRxRing, packets) % LPL_RING_INTERFERENCE_SIZE == 0 &&
                      offsetof(LplTxRing, packets) % LPL_RING_INTERFERENCE_SIZE == 0,
                  "the slots never share a span with the indices");

/* ─── Simple ring slot (for non-mmap fallback path) ─────────────────────── */

struct lpl_ring_slot {
    uint32_t length;
    uint8_t data[LPL_MAX_PACKET_SIZE];
};

/* ─── Stats reported via ioctl ──────────────────────────────────────────── */

struct lpl_stats {
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
