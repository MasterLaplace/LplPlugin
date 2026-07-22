/**
 * @file IMemoryBackend.hpp
 * @brief Abstract source of large memory reservations (arena backing).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-21
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_IMEMORYBACKEND_HPP
#    define LPL_PLATFORM_IMEMORYBACKEND_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::platform {

/**
 * @class IMemoryBackend
 * @brief Strategy interface for reserving the large blocks the engine bump-
 *        allocates out of (arenas, pools).
 *
 * This is the seam the engine's allocators sit ON, not a replacement for them:
 * lpl::memory::ArenaAllocator stays the single, portable, deterministic bump
 * allocator on every target — its byte accounting is a determinism gate (the P1
 * arena smoke folds arena.used() and requires it to match the Linux oracle
 * bit-for-bit, which two different arena implementations could not guarantee).
 * What differs per platform is only WHERE the arena's backing block comes from:
 * malloc on a host, a pre-mapped region in the kernel.
 *
 * Reserving up front and bump-allocating from the block is also the fastest and
 * most predictable option: O(1) allocation, O(1) reset, no fragmentation, and
 * no allocator call at all during a tick — which is what makes the freestanding
 * REAL_TIME path safe, since kmalloc refuses to serve a hot loop.
 */
class IMemoryBackend {
public:
    virtual ~IMemoryBackend() = default;

    /**
     * @brief Reserve a contiguous block, once, at start-up.
     * @param sizeBytes Block size.
     * @param alignment Required base alignment (power of two).
     * @return Block base, or nullptr if the reservation failed.
     */
    [[nodiscard]] virtual void *reserve(core::usize sizeBytes, core::usize alignment) = 0;

    /**
     * @brief Release a block obtained from @ref reserve.
     * @param block Block base (nullptr is a no-op).
     * @param sizeBytes The size it was reserved with.
     */
    virtual void release(void *block, core::usize sizeBytes) = 0;

    /**
     * @brief Mark the start of a real-time critical section (one authoritative
     *        tick). A backend that can enforce it makes allocation fail inside,
     *        so an allocation-free path is proven rather than assumed.
     *        Default: nothing (hosts do not enforce it).
     */
    virtual void beginRealTimeSection() {}

    /**
     * @brief Mark the end of a real-time critical section.
     */
    virtual void endRealTimeSection() {}

    /**
     * @brief Allocation attempts refused inside real-time sections; 0 if unenforced.
     *
     * A backend refuses only what it cannot bound: growth through the page
     * allocator, a first-fit walk, or an outright failure. Requests it can serve
     * in O(1) with a bounded worst case are honoured, and reported by
     * @ref realTimeBoundedCount — a tick that only trips that counter still
     * holds its deadline.
     *
     * @return Count of allocation attempts refused inside real-time sections.
     */
    [[nodiscard]] virtual core::u32 realTimeViolationCount() const noexcept { return 0; }

    /**
     * @brief Bounded-path allocations served inside real-time sections; a budget, not an error.
     */
    [[nodiscard]] virtual core::u32 realTimeBoundedCount() const noexcept { return 0; }

    /**
     * @brief Returns a human-readable name.
     * @return Human-readable name.
     */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::platform

#endif // LPL_PLATFORM_IMEMORYBACKEND_HPP
