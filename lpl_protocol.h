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
#define MAX_PACKET_SIZE 256  // Taille max d'un paquet binaire

// --- Shared Protocol Types ---

/**
 * @brief IDs de composants pour le format de paquet dynamique.
 */
typedef enum {
    COMP_TRANSFORM = 1,
    COMP_HEALTH    = 2,
    COMP_VELOCITY  = 3,
    COMP_MASS      = 4
} ComponentID;

/**
 * @brief Paquet réseau à taille variable.
 * Format: [EntityID(4B)][CompID(1B)][Data...]...
 */
typedef struct {
    uint8_t data[MAX_PACKET_SIZE];
    uint16_t size;
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
