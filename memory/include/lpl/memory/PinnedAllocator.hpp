/**
 * @file PinnedAllocator.hpp
 * @brief STL-compatible allocator using CUDA pinned (page-locked) memory.
 *
 * When compiled with nvcc, memory is allocated via cudaHostAlloc with
 * mapped and portable flags, enabling zero-copy GPU access over PCIe.
 * Without CUDA, falls back to standard malloc/free.
 *
 * @tparam T Element type.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MEMORY_PINNED_ALLOCATOR_HPP
    #define LPL_MEMORY_PINNED_ALLOCATOR_HPP

    #include <lpl/core/Platform.hpp>
    #include <lpl/core/Types.hpp>

    #ifdef __CUDACC__
        #include <cuda_runtime.h>
    #endif

    #include <cstdlib>

namespace lpl::memory {

/**
 * @brief STL-conformant allocator producing CUDA-pinned memory.
 * @tparam T Element type managed by the allocator.
 */
template <typename T>
struct PinnedAllocator {
    using value_type = T;

    PinnedAllocator() noexcept = default;

    template <typename U>
    PinnedAllocator([[maybe_unused]] const PinnedAllocator<U> &) noexcept {}

    [[nodiscard]] T *allocate(core::usize n)
    {
        T *ptr = nullptr;
    #ifdef __CUDACC__
        cudaHostAlloc(&ptr, n * sizeof(T),
                      cudaHostAllocMapped | cudaHostAllocPortable);
    #else
        ptr = static_cast<T *>(std::malloc(n * sizeof(T)));
    #endif
        return ptr;
    }

    void deallocate(T *ptr, [[maybe_unused]] core::usize n) noexcept
    {
    #ifdef __CUDACC__
        cudaFreeHost(ptr);
    #else
        std::free(ptr);
    #endif
    }

    template <typename U>
    bool operator==([[maybe_unused]] const PinnedAllocator<U> &) const noexcept { return true; }
};

} // namespace lpl::memory

#endif // LPL_MEMORY_PINNED_ALLOCATOR_HPP
