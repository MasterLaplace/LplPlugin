/**
 * @file vector.hpp
 * @brief Portable vector alias: hosted std::vector, or kernel_std::vector on the
 *        freestanding kernel target. Use lpl::pmr::vector at call sites instead
 *        of std::vector so the same source compiles for both targets.
 *
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_STD_VECTOR_HPP
#    define LPL_STD_VECTOR_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
#        include <kernel_std/allocator.hpp>
#        include <kernel_std/vector.hpp>
namespace lpl::pmr {
// Allocator-aware alias: a custom allocator (e.g. memory::PinnedAllocator) can
// be supplied as the second argument; it defaults to the kernel heap allocator.
template <typename T, typename Allocator = ::kstd::KernelAllocator<T>>
using vector = ::kstd::vector<T, Allocator>;
}
#    else
#        include <memory>
#        include <vector>
namespace lpl::pmr {
template <typename T, typename Allocator = ::std::allocator<T>> using vector = ::std::vector<T, Allocator>;
}
#    endif

#endif // LPL_STD_VECTOR_HPP
