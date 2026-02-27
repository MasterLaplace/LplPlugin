/**
 * @file CudaBackend.cpp
 * @brief CudaBackend implementation — delegates to real CUDA when available.
 *
 * When compiled with the CUDA toolchain (LPL_HAS_CUDA), this backend
 * uses cudaMalloc/cudaFree/cudaMemcpy and the physics kernel from
 * PhysicsKernel.cu. Without CUDA, all functions return NotSupported.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/gpu/CudaBackend.hpp>
#include <lpl/gpu/PhysicsKernel.cuh>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#ifdef LPL_HAS_CUDA
    #include <cuda_runtime.h>
#endif

namespace lpl::gpu {

struct CudaBackend::Impl
{
    bool initialised{false};
};

CudaBackend::CudaBackend()
    : _impl{std::make_unique<Impl>()}
{}

CudaBackend::~CudaBackend()
{
    if (_impl && _impl->initialised)
    {
        shutdown();
    }
}

core::Expected<void> CudaBackend::init()
{
#ifdef LPL_HAS_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        core::Log::warn("CudaBackend::init — no CUDA devices found");
        return core::makeError(core::ErrorCode::NotSupported, "No CUDA devices");
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    core::Log::info("CudaBackend::init — {} (compute {}.{})",
                    prop.name, prop.major, prop.minor);

    physics_gpu_init();
    _impl->initialised = true;
    return {};
#else
    core::Log::info("CudaBackend::init (stub — no CUDA toolchain)");
    return {};
#endif
}

void CudaBackend::shutdown()
{
#ifdef LPL_HAS_CUDA
    if (_impl->initialised)
    {
        physics_gpu_cleanup();
        _impl->initialised = false;
        core::Log::info("CudaBackend::shutdown — CUDA resources released");
    }
#else
    core::Log::info("CudaBackend::shutdown (stub)");
#endif
}

core::Expected<void*> CudaBackend::allocate(core::usize bytes)
{
#ifdef LPL_HAS_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
    {
        return core::makeError(core::ErrorCode::IoError, "cudaMalloc failed");
    }
    return ptr;
#else
    (void)bytes;
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
#endif
}

void CudaBackend::free(void* ptr)
{
#ifdef LPL_HAS_CUDA
    if (ptr)
    {
        cudaFree(ptr);
    }
#else
    (void)ptr;
#endif
}

core::Expected<void> CudaBackend::uploadSync(void* dst, const void* src, core::usize bytes)
{
#ifdef LPL_HAS_CUDA
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        return core::makeError(core::ErrorCode::IoError, "cudaMemcpy H2D failed");
    }
    return {};
#else
    (void)dst; (void)src; (void)bytes;
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
#endif
}

core::Expected<void> CudaBackend::downloadSync(void* dst, const void* src, core::usize bytes)
{
#ifdef LPL_HAS_CUDA
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        return core::makeError(core::ErrorCode::IoError, "cudaMemcpy D2H failed");
    }
    return {};
#else
    (void)dst; (void)src; (void)bytes;
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
#endif
}

core::Expected<void> CudaBackend::dispatch(
    const char* kernelName,
    core::u32 gridDim,
    core::u32 blockDim,
    std::span<const core::byte> args)
{
#ifdef LPL_HAS_CUDA
    (void)gridDim;
    (void)blockDim;

    // Dispatch known kernels by name
    if (std::strcmp(kernelName, "physics_tick") == 0)
    {
        // Args layout: [9 x float*][const float*][uint32_t count][float dt]
        // Total: 9 pointers + 1 pointer + 1 uint32_t + 1 float
        struct PhysicsArgs
        {
            float *posX, *posY, *posZ;
            float *velX, *velY, *velZ;
            float *frcX, *frcY, *frcZ;
            const float* masses;
            uint32_t count;
            float dt;
        };

        if (args.size() < sizeof(PhysicsArgs))
        {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "physics_tick: args buffer too small");
        }

        PhysicsArgs pa{};
        std::memcpy(&pa, args.data(), sizeof(PhysicsArgs));

        physics_gpu_timer_start();
        launch_physics_kernel(
            pa.posX, pa.posY, pa.posZ,
            pa.velX, pa.velY, pa.velZ,
            pa.frcX, pa.frcY, pa.frcZ,
            pa.masses, pa.count, pa.dt);

        return {};
    }

    return core::makeError(core::ErrorCode::InvalidArgument,
                           "Unknown kernel name");
#else
    (void)kernelName; (void)gridDim; (void)blockDim; (void)args;
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
#endif
}

core::Expected<void> CudaBackend::synchronize()
{
#ifdef LPL_HAS_CUDA
    physics_gpu_sync();
    return {};
#else
    return {};
#endif
}

const char* CudaBackend::name() const noexcept
{
    return "CudaBackend";
}

} // namespace lpl::gpu
