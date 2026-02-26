// /////////////////////////////////////////////////////////////////////////////
/// @file IComputeBackend.hpp
/// @brief Abstract GPU compute backend interface.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

#include <cstddef>
#include <span>

namespace lpl::gpu {

// /////////////////////////////////////////////////////////////////////////////
/// @class IComputeBackend
/// @brief Strategy interface for GPU compute dispatch.
///
/// Concrete implementations:
///   - @c CudaBackend          — NVIDIA CUDA.
///   - @c VulkanComputeBackend — Vulkan Compute (stub).
// /////////////////////////////////////////////////////////////////////////////
class IComputeBackend
{
public:
    virtual ~IComputeBackend() = default;

    /// @brief Initializes the backend (create context, etc.).
    [[nodiscard]] virtual core::Expected<void> init() = 0;

    /// @brief Shuts down the backend.
    virtual void shutdown() = 0;

    /// @brief Allocates device memory.
    /// @param bytes Number of bytes.
    /// @return Device pointer on success.
    [[nodiscard]] virtual core::Expected<void*> allocate(core::usize bytes) = 0;

    /// @brief Frees device memory.
    virtual void free(void* ptr) = 0;

    /// @brief Copies data from host to device.
    [[nodiscard]] virtual core::Expected<void> uploadSync(
        void* dst, const void* src, core::usize bytes) = 0;

    /// @brief Copies data from device to host.
    [[nodiscard]] virtual core::Expected<void> downloadSync(
        void* dst, const void* src, core::usize bytes) = 0;

    /// @brief Dispatches a compute kernel identified by name.
    /// @param kernelName  Name of the kernel.
    /// @param gridDim     Number of blocks.
    /// @param blockDim    Number of threads per block.
    /// @param args        Opaque argument buffer.
    [[nodiscard]] virtual core::Expected<void> dispatch(
        const char* kernelName,
        core::u32 gridDim,
        core::u32 blockDim,
        std::span<const core::byte> args) = 0;

    /// @brief Waits for all dispatched work to complete.
    [[nodiscard]] virtual core::Expected<void> synchronize() = 0;

    /// @brief Returns a human-readable name.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::gpu
