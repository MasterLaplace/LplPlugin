/**
 * @file PhysicsKernel.cuh
 * @brief CUDA physics kernel declarations.
 *
 * Provides GPU-accelerated physics (gravity + semi-implicit Euler).
 * When compiled without CUDA (__CUDACC__ not defined), all functions
 * are provided as inline no-ops for seamless CPU fallback.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_GPU_PHYSICSKERNEL_CUH
#    define LPL_GPU_PHYSICSKERNEL_CUH

#    include <cstdint>

namespace lpl::gpu {

/**
 * @brief Initialises CUDA timing events.
 * Must be called before any kernel launch.
 */
void physics_gpu_init();

/**
 * @brief Destroys CUDA timing events and releases resources.
 */
void physics_gpu_cleanup();

/**
 * @brief Launches the physics tick kernel on SoA buffers.
 *
 * Processes @p count entities in parallel:
 *   1. Gravity force: F = (0, -9.81 * mass, 0)
 *   2. Semi-implicit Euler: v += a * dt, p += v * dt
 *
 * Does NOT synchronize — call physics_gpu_sync() after batching
 * multiple chunk launches.
 *
 * @param d_posX/Y/Z  Device pointers to position X/Y/Z arrays.
 * @param d_velX/Y/Z  Device pointers to velocity X/Y/Z arrays.
 * @param d_frcX/Y/Z  Device pointers to force X/Y/Z arrays.
 * @param d_masses    Device pointer to mass array.
 * @param count       Number of entities.
 * @param dt          Time step in seconds.
 */
void launch_physics_kernel(float *d_posX, float *d_posY, float *d_posZ, float *d_velX, float *d_velY, float *d_velZ,
                           float *d_frcX, float *d_frcY, float *d_frcZ, const float *d_masses, uint32_t count,
                           float dt);

/**
 * @brief Synchronises all pending GPU work (cudaDeviceSynchronize).
 */
void physics_gpu_sync();

/**
 * @brief Starts the GPU performance timer.
 */
void physics_gpu_timer_start();

/**
 * @brief Stops the GPU performance timer.
 * @return Elapsed time in milliseconds.
 */
float physics_gpu_timer_stop();

// ─── CPU Fallback (no-ops) ───────────────────────────────────

#    if !defined(__CUDACC__) && !defined(LPL_HAS_CUDA)

inline void physics_gpu_init() {}
inline void physics_gpu_cleanup() {}
inline void launch_physics_kernel(float *, float *, float *, float *, float *, float *, float *, float *, float *,
                                  const float *, uint32_t, float)
{
}
inline void physics_gpu_sync() {}
inline void physics_gpu_timer_start() {}
inline float physics_gpu_timer_stop() { return 0.0f; }

#    endif // !__CUDACC__ && !LPL_HAS_CUDA

} // namespace lpl::gpu

#endif // LPL_GPU_PHYSICSKERNEL_CUH
