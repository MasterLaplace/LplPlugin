/**
 * @file GpuBuffer.hpp
 * @brief RAII wrapper for a device memory allocation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_GPU_GPUBUFFER_HPP
    #define LPL_GPU_GPUBUFFER_HPP

#include <lpl/gpu/IComputeBackend.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>
#include <lpl/core/Expected.hpp>

namespace lpl::gpu {

/**
 * @class GpuBuffer
 * @brief Owns a device allocation via an IComputeBackend reference.
 *
 * Provides typed upload/download helpers. The buffer is freed automatically
 * on destruction.
 */
class GpuBuffer final : public core::NonCopyable<GpuBuffer>
{
public:
    /** @brief Constructs an empty (null) buffer. */
    GpuBuffer() noexcept = default;

    /** @brief Allocates @p bytes on the given backend. */
    GpuBuffer(IComputeBackend& backend, core::usize bytes);

    /** @brief Move-constructs from @p other (transfers ownership). */
    GpuBuffer(GpuBuffer&& other) noexcept;

    /** @brief Move-assigns from @p other. */
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;

    /** @brief Frees the device allocation. */
    ~GpuBuffer();

    /** @brief Returns the device pointer (may be null). */
    [[nodiscard]] void* devicePtr() noexcept;

    /** @brief Returns the device pointer (const). */
    [[nodiscard]] const void* devicePtr() const noexcept;

    /** @brief Returns the size in bytes. */
    [[nodiscard]] core::usize size() const noexcept;

    /** @brief Uploads host data to this buffer. */
    [[nodiscard]] core::Expected<void> upload(const void* hostSrc, core::usize bytes);

    /** @brief Downloads device data to host memory. */
    [[nodiscard]] core::Expected<void> download(void* hostDst, core::usize bytes) const;

private:
    IComputeBackend* _backend{nullptr};
    void*            _ptr{nullptr};
    core::usize      _size{0};
};

} // namespace lpl::gpu

#endif // LPL_GPU_GPUBUFFER_HPP
