#pragma once

#include <cstddef>
#include <new>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#include <cstdlib>
#endif

/**
 * @brief Allocateur pinned-memory pour std::vector.
 * Utilise cudaHostAlloc (mapped + portable) quand compilé avec nvcc,
 * sinon tombe sur malloc/free standard.
 */
template <typename T>
struct PinnedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;

    PinnedAllocator() noexcept = default;

    template <typename U>
    PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T *allocate(std::size_t n)
    {
#if defined(__CUDACC__)
        T *ptr = nullptr;
        cudaError_t err = cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable);
        if (err != cudaSuccess)
            throw std::bad_alloc();
        return ptr;
#else
        T *ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
        if (!ptr)
            throw std::bad_alloc();
        return ptr;
#endif
    }

    void deallocate(T *p, std::size_t) noexcept
    {
#if defined(__CUDACC__)
        cudaFreeHost(p);
#else
        std::free(p);
#endif
    }

    template <typename U>
    bool operator==(const PinnedAllocator<U>&) const noexcept { return true; }
};
