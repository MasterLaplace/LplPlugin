// --- LAPLACE PROTOCOL --- //
// File: lpl_protocol.h
// Description: Structures réseau partagées entre le kernel module et le moteur userspace
// Note: Ce header doit rester compatible C pur pour le kernel Linux
// Auteur: MasterLaplace

#ifndef LPL_PROTOCOL_H
#define LPL_PROTOCOL_H

#if defined(MODULE)
// --- Linux Kernel Context ---
#include <linux/types.h>
#include <linux/atomic.h>

typedef atomic_t atomic_uint;

#else
// --- C++ Userspace Context ---
#include <atomic>
#include <cstdint>

typedef std::atomic<unsigned int> atomic_uint;
typedef std::atomic<bool> atomic_bool;

#define lpl_atomic_load(ptr)          (ptr)->load(std::memory_order_relaxed)
#define lpl_atomic_store(ptr, val)    (ptr)->store(val, std::memory_order_relaxed)
#define lpl_atomic_fetch_add(ptr, val) (ptr)->fetch_add(val, std::memory_order_relaxed)
#define lpl_atomic_fetch_sub(ptr, val) (ptr)->fetch_sub(val, std::memory_order_relaxed)
#define lpl_atomic_exchange(ptr, val)  (ptr)->exchange(val, std::memory_order_relaxed)

#endif

// --- Shared Protocol Constants ---
#define RING_SIZE 4096       // Puissance de 2 pour utiliser un masque binaire rapide
#define MAX_PACKET_SIZE 256  // Taille max d'un payload binaire

// --- Ring Buffer Message Types ---
#define RING_MSG_DYNAMIC 0x01

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
#define MSG_INPUT    0x12  /* Client->Server: [1B type][4B entityId][12B direction(Vec3)]     */
#define MSG_STATE    0x13  /* Server->Client: [1B type][2B count][{4B id,12B pos,12B size,4B hp}xN] */

/**
 * @brief Paquet réseau à taille variable.
 * Header: [MsgType(1B)][PayloadSize(2B)]
 * Payload: [EntityID(4B)][CompID(1B)][Data...]...
 */
typedef struct {
    uint8_t msgType;
    uint16_t size; // Taille du payload (data)
    uint8_t data[MAX_PACKET_SIZE];
} DynamicPacket;

/**
 * @brief Ring buffer lockless partagé entre kernel et userspace via mmap.
 */
typedef struct {
    DynamicPacket packets[RING_SIZE];
    atomic_uint head;
    atomic_uint tail;
} NetworkRingBuffer;

#endif // LPL_PROTOCOL_H
