// --- LAPLACE PLUGIN SYSTEM --- //
// File: plugin.h
// Description: Header du plugin pour la gestion des entités et de la communication réseau
// Auteur: MasterLaplace

#ifndef PLUGIN_H
#define PLUGIN_H

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __cplusplus
#include <atomic>
using std::atomic;
typedef std::atomic<unsigned int> atomic_uint;
typedef std::atomic<bool> atomic_bool;
#define atomic_load(ptr) (ptr)->load(std::memory_order_relaxed)
#define atomic_store(ptr, val) (ptr)->store(val, std::memory_order_relaxed)
#define atomic_fetch_add(ptr, val) \
    (ptr)->fetch_add(val, std::memory_order_relaxed)
#define atomic_fetch_sub(ptr, val) \
    (ptr)->fetch_sub(val, std::memory_order_relaxed)
#define atomic_exchange(ptr, val) \
    (ptr)->exchange(val, std::memory_order_relaxed)
#else
#include <stdatomic.h>
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Macro pour vérifier les erreurs CUDA (Indispensable pour le debug)
#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                       \
            exit(1);                                               \
        }                                                          \
    } while (0)

typedef struct
{
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
} Chronos;

#elif defined(MODULE)
typedef atomic_t atomic_uint;
typedef atomic_t atomic_bool;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
#else
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#endif

#define MAX_ENTITIES 10000
#define MAX_ID 1000000

#define RING_SIZE 4096      // Puissance de 2 pour utiliser un masque binaire rapide
#define MAX_PACKET_SIZE 256 // Taille max d'un paquet binaire

#define INDEX_BITS 14
#define INDEX_MASK 0x3FFF // (1 << 14) - 1

/**
 * @brief IDs de composants pour le format de paquet dynamique.
 *
 * @param COMP_TRANSFORM Composant de transformation (position x, y, z).
 * @param COMP_HEALTH Composant de santé (points de vie).
 */
typedef enum {
    COMP_TRANSFORM = 1,
    COMP_HEALTH = 2
} ComponentID;

typedef struct {
    uint8_t data[MAX_PACKET_SIZE]; // Le "blob" de données brutes
    uint16_t size;
} DynamicPacket;

typedef struct {
    DynamicPacket packets[RING_SIZE];
    atomic_uint head;
    atomic_uint tail;
} NetworkRingBuffer;

#if !defined(MODULE)
/**
 * @brief Core Structure contenant toutes les structures de données principales
 * pour la gestion des entités.
 *
 * @param sparse_lookup Table d'indirection.
 * @param entity_generations Table de génération des entités.
 * @param dirty_stack Pile des entités "dirty" (modifiées cette frame).
 * @param dirty_count Compteur atomique du nombre d'entités "dirty".
 * @param is_dirty Bitset pour savoir si une entité est "dirty" ou pas.
 * @param free_indices Pile des indices libres pour la création d'entités.
 * @param free_count Compteur atomique du nombre d'indices libres.
 * @param write_idx Index du buffer d'écriture actuel.
 */
typedef struct {
    uint32_t sparse_lookup[MAX_ID];
    uint16_t entity_generations[MAX_ENTITIES];
    uint32_t dirty_stack[MAX_ENTITIES];
    atomic_uint dirty_count;
    atomic_bool is_dirty[MAX_ENTITIES];
    uint32_t free_indices[MAX_ENTITIES];
    atomic_uint free_count;
    atomic_uint write_idx;
} Core;

// Public API
#ifdef __cplusplus
extern "C"
{
#endif

    extern void server_init(void);
    extern void server_cleanup(void);
    extern uint32_t create_entity(uint32_t public_id);
    extern void destroy_entity(uint32_t public_id);
    extern bool is_entity_valid(uint32_t smart_id);
    extern void swap_buffers(void);
    extern void consume_packets(NetworkRingBuffer *ring);
    extern void get_render_pointers(float **out_x, float **out_y, float **out_z);
    extern void get_health_pointer(int **out_health);
    extern void get_gpu_pointers(float **dev_x, float **dev_y, float **dev_z);
    extern void run_physics_gpu(float delta_time);

    static inline uint16_t get_entity_id(uint32_t entity)
    {
        return entity & INDEX_MASK;
    }

    static inline uint32_t get_entity_generation(uint32_t entity)
    {
        return entity >> INDEX_BITS;
    }

#ifdef __cplusplus
}
#endif

#endif // !defined(MODULE)
#endif // PLUGIN_H
