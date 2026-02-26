/**
 * @file PoolAllocator.hpp
 * @brief Fixed-size block allocator with intrusive free-list.
 *
 * All blocks are the same size and chained through an intrusive
 * linked list stored inside the free blocks themselves, wasting
 * zero extra memory.  O(1) allocate, O(1) free, zero fragmentation.
 *
 * @tparam T Type of objects managed by the pool.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MEMORY_POOL_ALLOCATOR_HPP
    #define LPL_MEMORY_POOL_ALLOCATOR_HPP

    #include "IAllocator.hpp"

    #include <lpl/core/Assert.hpp>
    #include <lpl/core/NonCopyable.hpp>

namespace lpl::memory {

/**
 * @brief Statically-typed object pool with intrusive free-list.
 * @tparam T Pooled type (must be at least pointer-sized).
 */
template <typename T>
class PoolAllocator final : public IAllocator, private core::NonCopyable<PoolAllocator<T>> {
public:
    /**
     * @brief Construct a pool pre-allocating storage for @p count objects.
     * @param count Maximum number of simultaneously live objects.
     */
    explicit PoolAllocator(core::usize count);
    ~PoolAllocator() override;

    [[nodiscard]] void *allocate(core::usize size, core::usize alignment) override;
    void deallocate(void *ptr) override;
    [[nodiscard]] bool ownsPtr(const void *ptr) const override;

    /**
     * @brief Acquire a typed pointer from the pool.
     * @return Pointer to uninitialised storage, or nullptr if exhausted.
     */
    [[nodiscard]] T *acquire();

    /**
     * @brief Return a typed pointer to the pool.
     * @param ptr Pointer previously returned by acquire().
     */
    void release(T *ptr);

    [[nodiscard]] core::usize freeCount() const { return _freeCount; }

private:
    struct FreeNode { FreeNode *next; };

    char       *_memory    = nullptr;
    FreeNode   *_head      = nullptr;
    core::usize _blockSize = 0;
    core::usize _count     = 0;
    core::usize _freeCount = 0;
};

} // namespace lpl::memory

    #include "PoolAllocator.inl"

#endif // LPL_MEMORY_POOL_ALLOCATOR_HPP
