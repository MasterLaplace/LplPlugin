// --- LAPLACE PHYSICS GPU TOOLKIT --- //
// File: PhysicsGPU.cu
// Description: Kernels CUDA de physique et helpers de lancement
// Auteur: MasterLaplace

#include "PhysicsGPU.cuh"
#include "Math.hpp"

// ─── GPU Timing Events ───────────────────────────────────────

static cudaEvent_t g_physStart, g_physStop;
static bool g_physEventsCreated = false;

// ─── CUDA Kernels ─────────────────────────────────────────────

/**
 * @brief Kernel de physique (in-place sur le write buffer).
 * Gravité + intégration Euler semi-implicite sur les vecteurs SoA.
 */
__global__ void kernel_physics_tick(
    Vec3 *positions, Vec3 *velocities, Vec3 *forces, float *masses,
    uint32_t count, float dt)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    // Gravité
    forces[idx] = Vec3{0.0f, -9.81f * masses[idx], 0.0f};

    // Euler semi-implicite : v += a*dt, puis p += v*dt
    if (masses[idx] > 0.0001f)
    {
        float invMass = 1.0f / masses[idx];
        velocities[idx] += forces[idx] * invMass * dt;
    }

    positions[idx] += velocities[idx] * dt;
}

// ─── GPU Lifecycle ────────────────────────────────────────────

void gpu_init()
{
    if (!g_physEventsCreated)
    {
        CUDA_CHECK(cudaEventCreate(&g_physStart));
        CUDA_CHECK(cudaEventCreate(&g_physStop));
        g_physEventsCreated = true;
    }
}

void gpu_cleanup()
{
    if (g_physEventsCreated)
    {
        CUDA_CHECK(cudaEventDestroy(g_physStart));
        CUDA_CHECK(cudaEventDestroy(g_physStop));
        g_physEventsCreated = false;
    }
}

// ─── Kernel Launchers ─────────────────────────────────────────

void launch_physics_kernel(Vec3 *d_pos, Vec3 *d_vel, Vec3 *d_forces,
                           float *d_masses, uint32_t count, float dt)
{
    static constexpr int THREADS_PER_BLOCK = 256;
    int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_physics_tick<<<blocks, THREADS_PER_BLOCK>>>(d_pos, d_vel, d_forces, d_masses, count, dt);
}

void gpu_sync()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── GPU Monitoring ───────────────────────────────────────────

void gpu_timer_start()
{
    if (g_physEventsCreated)
        CUDA_CHECK(cudaEventRecord(g_physStart, 0));
}

float gpu_timer_stop()
{
    if (!g_physEventsCreated)
        return 0.0f;

    CUDA_CHECK(cudaEventRecord(g_physStop, 0));
    CUDA_CHECK(cudaEventSynchronize(g_physStop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_physStart, g_physStop));
    return ms;
}
