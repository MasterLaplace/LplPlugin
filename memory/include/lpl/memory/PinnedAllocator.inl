#ifndef LPL_MEMORY_PINNED_ALLOCATOR_INL
#define LPL_MEMORY_PINNED_ALLOCATOR_INL

namespace lpl::memory {

template <typename T> inline T *PinnedAllocator<T>::allocate(core::usize n)
{
    T *ptr = nullptr;
#ifdef __CUDACC__
    cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable);
#else
    ptr = static_cast<T *>(lpl::pmr::malloc(n * sizeof(T)));
#endif
    return ptr;
}

template <typename T> inline void PinnedAllocator<T>::deallocate(T *ptr, [[maybe_unused]] core::usize n) noexcept
{
#ifdef __CUDACC__
    cudaFreeHost(ptr);
#else
    lpl::pmr::free(ptr);
#endif
}

} // namespace lpl::memory

#endif // LPL_MEMORY_PINNED_ALLOCATOR_INL
