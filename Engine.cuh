// --- LAPLACE ENGINE --- //
// File: Engine.cuh
// Description: Header principal du moteur — ECS, CUDA, API publique
// Auteur: MasterLaplace

#ifndef ENGINE_CUH
#define ENGINE_CUH

#include "lpl_protocol.h"

// --- CUDA Support ---
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

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

#endif // __CUDACC__

// --- ECS Constants ---
#ifndef MAX_ENTITIES
#define MAX_ENTITIES 10000
#endif
#ifndef MAX_ID
#define MAX_ID 1000000
#endif

#ifndef INDEX_BITS
#define INDEX_BITS 14
#endif
#ifndef INDEX_MASK
#define INDEX_MASK 0x3FFF // (1 << 14) - 1
#endif

// --- Performance Monitoring ---
typedef struct {
    uint64_t packets_sent;
    uint64_t packets_received;
    uint64_t total_latency_ns;
    uint64_t min_latency_ns;
    uint64_t max_latency_ns;
    uint64_t frame_count;
    uint64_t start_time_ns;
} PerfMetrics;

// --- Core ECS Structure ---
/**
 * @brief Structure principale contenant toutes les données ECS.
 *
 * @param sparse_lookup Table d'indirection public_id → internal_index.
 * @param entity_generations Table de génération pour invalider les handles.
 * @param dirty_stack Pile des entités modifiées cette frame.
 * @param dirty_count Compteur atomique d'entités dirty.
 * @param is_dirty Bitset atomique par entité.
 * @param free_indices Pile des indices libres (recyclage).
 * @param free_count Compteur atomique d'indices libres.
 * @param write_idx Index du buffer d'écriture actuel (0 ou 1).
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

// --- Public API ---
#ifdef __cplusplus

class WorldPartition; // Forward declaration

extern "C"
{
#endif

    // --- Legacy API (flat ECS arrays) ---
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

// --- New Unified GPU API (WorldPartition-based) ---

/**
 * @brief Initialise le contexte CUDA et les événements de timing.
 * Doit être appelé AVANT toute allocation pinned memory.
 */
void engine_init();

/**
 * @brief Libère les ressources CUDA (événements, contexte).
 */
void engine_cleanup();

/**
 * @brief Exécute la physique GPU sur tous les chunks du monde.
 *
 * Pour chaque chunk non-vide :
 * 1. Obtient les device pointers depuis la pinned memory
 * 2. Lance kernel_physics_tick (gravité + Euler semi-implicite)
 * 3. Synchronise le GPU
 * 4. Vérifie les migrations inter-chunk (CPU)
 *
 * @param world Le WorldPartition contenant les chunks.
 * @param dt    Pas de temps (en secondes).
 */
void engine_physics_tick(WorldPartition &world, float dt);

/**
 * @brief Consomme les paquets du ring buffer et les dispatch dans le WorldPartition.
 * Crée automatiquement les entités inconnues.
 */
void engine_consume_packets(NetworkRingBuffer *ring, WorldPartition &world);

#endif // __cplusplus

#endif // ENGINE_CUH
