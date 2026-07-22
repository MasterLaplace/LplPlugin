/**
 * @file ArenaAllocator.hpp
 * @brief Linear bump-pointer allocator with O(1) mass-reset.
 *
 * Pre-allocates a contiguous slab at construction.  Allocation advances
 * a pointer; deallocation of individual blocks is not supported.
 * Calling reset() reclaims all memory in O(1).
 *
 * Ideal for per-frame scratch memory (physics contacts, network events).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MEMORY_ARENA_ALLOCATOR_HPP
#    define LPL_MEMORY_ARENA_ALLOCATOR_HPP

#    include "IAllocator.hpp"

#    include <lpl/core/NonCopyable.hpp>

namespace lpl::memory {

/**
 * @brief Bump-pointer arena.  O(1) allocate, O(1) reset, no free.
 */
class ArenaAllocator final : public IAllocator, private core::NonCopyable<ArenaAllocator> {
public:
    /**
     * @brief Construct an arena with a fixed capacity.
     * @param capacity Total bytes available.
     */
    explicit ArenaAllocator(core::usize capacity);

    /**
     * @brief Construct an arena over memory somebody else owns.
     *
     * Used when the backing block comes from the platform seam
     * (platform::IMemoryBackend): malloc on a host, a kernel reservation in ring
     * 0. The bump logic — and therefore the byte accounting the determinism gate
     * folds — is identical either way; only the block's origin differs.
     *
     * @param memory Block base; must outlive the arena and stay untouched.
     * @param capacity Block size in bytes.
     */
    ArenaAllocator(void *memory, core::usize capacity) noexcept;

    ~ArenaAllocator() override;

    [[nodiscard]] void *allocate(core::usize size, core::usize alignment = alignof(std::max_align_t)) override;
    void deallocate(void *ptr) override;
    void reset() override;
    [[nodiscard]] bool ownsPtr(const void *ptr) const override;

    [[nodiscard]] core::usize used() const { return _offset; }
    [[nodiscard]] core::usize capacity() const { return _capacity; }

private:
    char *_memory = nullptr;
    core::usize _capacity = 0;
    core::usize _offset = 0;
    bool _ownsMemory = true; ///< False when the block came from the platform.
};

} // namespace lpl::memory

#endif // LPL_MEMORY_ARENA_ALLOCATOR_HPP
