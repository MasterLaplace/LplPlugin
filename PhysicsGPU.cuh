// --- LAPLACE PHYSICS GPU TOOLKIT --- //
// File: PhysicsGPU.cuh
// Description: Toolkit CUDA — kernels de physique, helpers de lancement, timing GPU
// Auteur: MasterLaplace

#ifndef PHYSICS_GPU_CUH
#define PHYSICS_GPU_CUH

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
#endif // __CUDACC__

struct Vec3; // Forward declaration

// ─── GPU Lifecycle ────────────────────────────────────────────

/**
 * @brief Initialise le contexte CUDA et les événements de timing.
 * Doit être appelé AVANT toute allocation pinned memory.
 */
void gpu_init();

/**
 * @brief Libère les ressources CUDA (événements).
 */
void gpu_cleanup();

// ─── GPU Kernel Launchers ─────────────────────────────────────

/**
 * @brief Lance le kernel de physique sur un buffer SoA (pas de sync).
 *
 * Calcule automatiquement le nombre de blocks/threads et lance
 * kernel_physics_tick. Ne synchronise PAS — permet de battre
 * plusieurs chunks avant un seul gpu_sync().
 *
 * @param d_pos    Device pointer vers les positions (write buffer).
 * @param d_vel    Device pointer vers les vélocités (write buffer).
 * @param d_forces Device pointer vers les forces (write buffer).
 * @param d_masses Device pointer vers les masses.
 * @param count    Nombre d'entités.
 * @param dt       Pas de temps (secondes).
 */
void launch_physics_kernel(Vec3 *d_pos, Vec3 *d_vel, Vec3 *d_forces,
                           float *d_masses, uint32_t count, float dt);

/**
 * @brief Synchronise le GPU (attend que tous les kernels en cours soient terminés).
 */
void gpu_sync();

// ─── GPU Monitoring ───────────────────────────────────────────

/**
 * @brief Démarre un chronomètre GPU (event recording).
 */
void gpu_timer_start();

/**
 * @brief Arrête le chronomètre GPU et retourne le temps écoulé en ms.
 * @return Temps GPU en millisecondes.
 */
float gpu_timer_stop();

#if !defined(__CUDACC__)
inline void gpu_init() {}
inline void gpu_cleanup() {}
inline void launch_physics_kernel(Vec3 *, Vec3 *, Vec3 *, float *, uint32_t, float) {}
inline void gpu_sync() {}
inline void gpu_timer_start() {}
inline float gpu_timer_stop() { return 0.0f; }
#endif

#endif // PHYSICS_GPU_CUH
