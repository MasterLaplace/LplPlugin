// /////////////////////////////////////////////////////////////////////////////
/// @file CudaBackend.hpp
/// @brief NVIDIA CUDA compute backend.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/gpu/IComputeBackend.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::gpu {

// /////////////////////////////////////////////////////////////////////////////
/// @class CudaBackend
/// @brief IComputeBackend implementation using CUDA Runtime API.
///
/// Wraps cudaMalloc / cudaFree / cudaMemcpy / kernel-launch.
/// Only available when compiled with the CUDA toolchain.
// /////////////////////////////////////////////////////////////////////////////
class CudaBackend final : public IComputeBackend,
                          public core::NonCopyable<CudaBackend>
{
public:
    CudaBackend();
    ~CudaBackend() override;

    [[nodiscard]] core::Expected<void>  init() override;
    void                                shutdown() override;

    [[nodiscard]] core::Expected<void*> allocate(core::usize bytes) override;
    void                                free(void* ptr) override;

    [[nodiscard]] core::Expected<void>  uploadSync(
        void* dst, const void* src, core::usize bytes) override;

    [[nodiscard]] core::Expected<void>  downloadSync(
        void* dst, const void* src, core::usize bytes) override;

    [[nodiscard]] core::Expected<void>  dispatch(
        const char* kernelName,
        core::u32 gridDim,
        core::u32 blockDim,
        std::span<const core::byte> args) override;

    [[nodiscard]] core::Expected<void>  synchronize() override;

    [[nodiscard]] const char*           name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::gpu
