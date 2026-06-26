/**
 * @file cstdlib.hpp
 * @brief Portable heap-management umbrella. Hosted: lpl::pmr aliases std::.
 *        Kernel: routes to the kernel heap allocator (kmalloc / kfree).
 *
 * This is the dependency-injection seam for raw byte allocation used by the
 * engine's custom allocators (ArenaAllocator slab, PinnedAllocator fallback).
 * On the kernel target there is no hosted C runtime: malloc/free map onto the
 * kernel heap facade declared in <kernel/memory/heap.h> (kmalloc/kfree).
 *
 * Alignment contract (kernel arm): kmalloc guarantees 8-byte alignment, which
 * equals alignof(max_align_t) on i686. aligned_alloc therefore supports any
 * alignment up to that bound; over-aligned requests (alignment > 8) are NOT
 * serviced here and must go through the C++ aligned operator new path instead.
 *
 * Call sites must use lpl::pmr::malloc / free / aligned_alloc, never std::, so
 * the routing stays a single edit.
 */
#pragma once

#ifndef LPL_STD_CSTDLIB_HPP
#    define LPL_STD_CSTDLIB_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
extern "C" {
void *kmalloc(__SIZE_TYPE__ size);
void kfree(void *ptr);
}

namespace lpl::pmr {

[[nodiscard]] inline void *malloc(__SIZE_TYPE__ size) noexcept { return ::kmalloc(size); }

inline void free(void *ptr) noexcept { ::kfree(ptr); }

/* alignment is honoured implicitly: kmalloc returns 8-byte-aligned storage,
   which covers alignof(max_align_t) on i686. The parameter is accepted for
   signature parity with the hosted std::aligned_alloc. */
[[nodiscard]] inline void *aligned_alloc([[maybe_unused]] __SIZE_TYPE__ alignment, __SIZE_TYPE__ size) noexcept
{
    return ::kmalloc(size);
}

} // namespace lpl::pmr
#    else
#        include <cstdlib>
namespace lpl::pmr {
using ::std::aligned_alloc;
using ::std::free;
using ::std::malloc;
} // namespace lpl::pmr
#    endif

#endif // LPL_STD_CSTDLIB_HPP
