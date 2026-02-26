// /////////////////////////////////////////////////////////////////////////////
/// @file VulkanComputeBackend.hpp
/// @brief Vulkan Compute backend (stub for future implementation).
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/gpu/IComputeBackend.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::gpu {

// /////////////////////////////////////////////////////////////////////////////
/// @class VulkanComputeBackend
/// @brief IComputeBackend implementation using Vulkan Compute shaders.
///
/// Provides vendor-agnostic GPU compute as an alternative to CUDA.
// /////////////////////////////////////////////////////////////////////////////
class VulkanComputeBackend final : public IComputeBackend,
                                    public core::NonCopyable<VulkanComputeBackend>
{
public:
    VulkanComputeBackend();
    ~VulkanComputeBackend() override;

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
