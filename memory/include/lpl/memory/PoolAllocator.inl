/**
 * @file PoolAllocator.inl
 * @brief Template implementation of PoolAllocator.
 * @see   PoolAllocator.hpp
 */

#ifndef LPL_MEMORY_POOL_ALLOCATOR_INL
    #define LPL_MEMORY_POOL_ALLOCATOR_INL

    #include <algorithm>
    #include <cstdlib>
    #include <cstdint>

namespace lpl::memory {

template <typename T>
PoolAllocator<T>::PoolAllocator(core::usize count)
    : _blockSize(std::max(sizeof(T), sizeof(FreeNode)))
    , _count(count)
    , _freeCount(count)
{
    _memory = static_cast<char *>(std::aligned_alloc(alignof(T), _blockSize * count));

    _head = reinterpret_cast<FreeNode *>(_memory);
    auto *current = _head;
    for (core::usize i = 1; i < count; ++i) {
        auto *next = reinterpret_cast<FreeNode *>(_memory + i * _blockSize);
        current->next = next;
        current = next;
    }
    current->next = nullptr;
}

template <typename T>
PoolAllocator<T>::~PoolAllocator()
{
    std::free(_memory);
}

template <typename T>
void *PoolAllocator<T>::allocate([[maybe_unused]] core::usize size, [[maybe_unused]] core::usize alignment)
{
    return acquire();
}

template <typename T>
void PoolAllocator<T>::deallocate(void *ptr)
{
    release(static_cast<T *>(ptr));
}

template <typename T>
bool PoolAllocator<T>::ownsPtr(const void *ptr) const
{
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto base = reinterpret_cast<std::uintptr_t>(_memory);
    return addr >= base && addr < base + _blockSize * _count;
}

template <typename T>
T *PoolAllocator<T>::acquire()
{
    if (!_head)
        return nullptr;

    auto *node = _head;
    _head = _head->next;
    --_freeCount;
    return reinterpret_cast<T *>(node);
}

template <typename T>
void PoolAllocator<T>::release(T *ptr)
{
    LPL_ASSERT(ownsPtr(ptr));

    auto *node = reinterpret_cast<FreeNode *>(ptr);
    node->next = _head;
    _head = node;
    ++_freeCount;
}

} // namespace lpl::memory

#endif // LPL_MEMORY_POOL_ALLOCATOR_INL
