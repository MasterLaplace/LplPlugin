/**
 * @file PhysicsKernel.cu
 * @brief CUDA physics kernels — gravity + semi-implicit Euler integration.
 *
 * Ported from legacy PhysicsGPU.cu into the new modular architecture.
 * Works on SoA buffers (positions, velocities, forces, masses) in-place.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/gpu/PhysicsKernel.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>

namespace lpl::gpu {

// ─── CUDA Error Checking ─────────────────────────────────────

#define LPL_CUDA_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));             \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

// ─── GPU Timing Events ───────────────────────────────────────

static cudaEvent_t s_physStart{};
static cudaEvent_t s_physStop{};
static bool s_eventsCreated{false};

// ─── CUDA Kernels ────────────────────────────────────────────

/**
 * @brief Physics tick kernel — gravity + semi-implicit Euler.
 *
 * Each thread processes one entity:
 *   1. Apply gravity: force = (0, -9.81 * mass, 0)
 *   2. Semi-implicit Euler: v += a * dt, then p += v * dt
 *
 * @param positions  SoA positions array (write buffer).
 * @param velocities SoA velocities array (write buffer).
 * @param forces     SoA forces array (write buffer).
 * @param masses     SoA masses array.
 * @param count      Number of entities.
 * @param dt         Time step in seconds.
 */
__global__ void kernel_physics_tick(float *__restrict__ posX, float *__restrict__ posY, float *__restrict__ posZ,
                                    float *__restrict__ velX, float *__restrict__ velY, float *__restrict__ velZ,
                                    float *__restrict__ frcX, float *__restrict__ frcY, float *__restrict__ frcZ,
                                    const float *__restrict__ masses, uint32_t count, float dt)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    // Gravity
    frcX[idx] = 0.0f;
    frcY[idx] = -9.81f * masses[idx];
    frcZ[idx] = 0.0f;

    // Semi-implicit Euler: v += a * dt
    if (masses[idx] > 0.0001f)
    {
        float invMass = 1.0f / masses[idx];
        velX[idx] += frcX[idx] * invMass * dt;
        velY[idx] += frcY[idx] * invMass * dt;
        velZ[idx] += frcZ[idx] * invMass * dt;
    }

    // Position update: p += v * dt
    posX[idx] += velX[idx] * dt;
    posY[idx] += velY[idx] * dt;
    posZ[idx] += velZ[idx] * dt;
}

// ─── GPU Lifecycle ───────────────────────────────────────────

void physics_gpu_init()
{
    if (!s_eventsCreated)
    {
        LPL_CUDA_CHECK(cudaEventCreate(&s_physStart));
        LPL_CUDA_CHECK(cudaEventCreate(&s_physStop));
        s_eventsCreated = true;
    }
}

void physics_gpu_cleanup()
{
    if (s_eventsCreated)
    {
        LPL_CUDA_CHECK(cudaEventDestroy(s_physStart));
        LPL_CUDA_CHECK(cudaEventDestroy(s_physStop));
        s_eventsCreated = false;
    }
}

// ─── Kernel Launcher ─────────────────────────────────────────

void launch_physics_kernel(float *d_posX, float *d_posY, float *d_posZ, float *d_velX, float *d_velY, float *d_velZ,
                           float *d_frcX, float *d_frcY, float *d_frcZ, const float *d_masses, uint32_t count, float dt)
{
    static constexpr int kThreadsPerBlock = 256;
    int blocks = (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
    kernel_physics_tick<<<blocks, kThreadsPerBlock>>>(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, d_frcX, d_frcY,
                                                      d_frcZ, d_masses, count, dt);
}

// ─── Synchronization ─────────────────────────────────────────

void physics_gpu_sync() { LPL_CUDA_CHECK(cudaDeviceSynchronize()); }

// ─── GPU Monitoring ──────────────────────────────────────────

void physics_gpu_timer_start()
{
    if (s_eventsCreated)
        LPL_CUDA_CHECK(cudaEventRecord(s_physStart, 0));
}

float physics_gpu_timer_stop()
{
    if (!s_eventsCreated)
        return 0.0f;

    LPL_CUDA_CHECK(cudaEventRecord(s_physStop, 0));
    LPL_CUDA_CHECK(cudaEventSynchronize(s_physStop));

    float ms = 0.0f;
    LPL_CUDA_CHECK(cudaEventElapsedTime(&ms, s_physStart, s_physStop));
    return ms;
}

} // namespace lpl::gpu
