/**
 * @file StackAllocator.hpp
 * @brief LIFO allocator with marker-based rollback.
 *
 * Similar to ArenaAllocator but supports ordered deallocation via
 * markers.  Ideal for recursive algorithms and scene-graph traversals
 * where temporaries are released in reverse order.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MEMORY_STACK_ALLOCATOR_HPP
    #define LPL_MEMORY_STACK_ALLOCATOR_HPP

    #include "IAllocator.hpp"

    #include <lpl/core/NonCopyable.hpp>

namespace lpl::memory {

/**
 * @brief LIFO allocator with O(1) allocate and O(1) free-to-marker.
 */
class StackAllocator final : public IAllocator, private core::NonCopyable<StackAllocator> {
public:
    using Marker = core::usize;

    explicit StackAllocator(core::usize capacity);
    ~StackAllocator() override;

    [[nodiscard]] void *allocate(core::usize size, core::usize alignment = alignof(std::max_align_t)) override;
    void deallocate(void *ptr) override;
    void reset() override;
    [[nodiscard]] bool ownsPtr(const void *ptr) const override;

    /**
     * @brief Capture the current allocation offset.
     * @return Marker representing the current stack top.
     */
    [[nodiscard]] Marker getMarker() const { return _offset; }

    /**
     * @brief Free all allocations made after the given marker.
     * @param marker Previously captured marker.
     */
    void freeToMarker(Marker marker);

private:
    char       *_memory   = nullptr;
    core::usize _capacity = 0;
    core::usize _offset   = 0;
};

} // namespace lpl::memory

#endif // LPL_MEMORY_STACK_ALLOCATOR_HPP
