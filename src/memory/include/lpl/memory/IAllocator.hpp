/**
 * @file IAllocator.hpp
 * @brief Abstract interface for all engine custom allocators.
 *
 * Every allocator in the engine conforms to this interface so that
 * subsystems can be parameterised on their memory strategy via
 * dependency injection.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MEMORY_IALLOCATOR_HPP
    #define LPL_MEMORY_IALLOCATOR_HPP

    #include <lpl/core/Types.hpp>

namespace lpl::memory {

/**
 * @brief Abstract allocator interface.
 *
 * Implementations must guarantee that allocate() returns suitably
 * aligned memory or nullptr on failure.  The hot path (allocate /
 * deallocate) must never call into the OS heap.
 */
class IAllocator {
public:
    virtual ~IAllocator() = default;

    /**
     * @brief Allocate a block of memory.
     * @param size      Requested size in bytes.
     * @param alignment Required alignment (power of two).
     * @return Pointer to the allocated block, or nullptr on failure.
     */
    [[nodiscard]] virtual void *allocate(core::usize size, core::usize alignment = alignof(std::max_align_t)) = 0;

    /**
     * @brief Return a previously allocated block.
     * @param ptr Pointer returned by allocate().
     */
    virtual void deallocate(void *ptr) = 0;

    /**
     * @brief Bulk-free all outstanding allocations (if supported).
     *
     * Default implementation is a no-op.  Arena and Stack allocators
     * override this to reset their internal pointer.
     */
    virtual void reset() {}

    /**
     * @brief Query whether a pointer belongs to this allocator.
     * @param ptr Pointer to check.
     * @return True if the block was allocated from this allocator.
     */
    [[nodiscard]] virtual bool ownsPtr(const void *ptr) const = 0;
};

} // namespace lpl::memory

#endif // LPL_MEMORY_IALLOCATOR_HPP
