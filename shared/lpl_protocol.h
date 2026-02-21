// --- LAPLACE PROTOCOL --- //
// File: lpl_protocol.h
// Description: Structures réseau partagées entre le kernel module et le moteur userspace
// Note: Ce header doit rester compatible C pur pour le kernel Linux
// Auteur: MasterLaplace

#ifndef LPL_PROTOCOL_H
#define LPL_PROTOCOL_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <asm/barrier.h>
#else
#include <stdint.h>

#define smp_load_acquire(p) __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define smp_store_release(p, v) __atomic_store_n(p, v, __ATOMIC_RELEASE)
#endif

// --- Shared Protocol Constants ---
#define RING_SLOTS 4096      // Puissance de 2 pour utiliser un masque binaire rapide
#define MAX_PACKET_SIZE 256  // Taille max d'un payload binaire
#define LPL_DEVICE_NAME "lpl_driver"
#define LPL_CLASS_NAME  "lpl"
#define LPL_PORT        7777u

// --- Shared Protocol Types ---

/**
 * @brief IDs de composants pour le format de paquet dynamique.
 */
typedef enum {
    COMP_TRANSFORM = 1,
    COMP_HEALTH    = 2,
    COMP_VELOCITY  = 3,
    COMP_MASS      = 4,
    COMP_SIZE      = 5
} ComponentID;

/* --- Message Types for Server<->Client UDP Communication ---
 * These are for direct UDP between server and visual clients,
 * NOT for the kernel ring buffer (which uses ComponentID format).
 */
#define MSG_CONNECT  0x10  /* Client->Server: [1B type]                                      */
#define MSG_WELCOME  0x11  /* Server->Client: [1B type][4B entityId]                          */
#define MSG_STATE    0x12  /* Server->Client: [1B type][2B count][{4B id,12B pos,12B size,4B hp}xN] */
#define MSG_INPUTS   0x13  /* Client->Server: [1B type][4B entityId]
                                              [1B keyCount][{1B key, 1B state}xN]
                                              [1B axisCount][{1B axis, 4B value}xN]
                                              [NeuralData(13B)] (optional, if len remains) */

typedef struct {
    uint32_t head;
    uint32_t tail;
    uint32_t _pad[6];
} RingHeader;

/**
 * @brief Paquet réseau à taille variable.
 * Header: [type(1B)][PayloadSize(2B)]
 * Payload: [EntityID(4B)][CompID(1B)][Data...]...
 */
typedef struct {
    uint32_t src_ip;
    uint16_t src_port;
    uint16_t length;
    uint8_t data[MAX_PACKET_SIZE];
} RxPacket;

typedef struct {
    uint32_t dst_ip;
    uint16_t dst_port;
    uint16_t length;
    uint8_t data[MAX_PACKET_SIZE];
} TxPacket;

/**
 * @brief Ring buffer lockless partagé entre kernel et userspace via mmap.
 */
typedef struct {
    RingHeader idx;
    RxPacket packets[RING_SLOTS];
} RxRingBuffer;

typedef struct {
    RingHeader idx;
    TxPacket packets[RING_SLOTS];
} TxRingBuffer;

typedef struct {
    RxRingBuffer rx;
    TxRingBuffer tx;
} LplSharedMemory;

#define LPL_IOC_MAGIC 'k'
#define LPL_IOC_KICK_TX _IO(LPL_IOC_MAGIC, 1)

#endif // LPL_PROTOCOL_H
