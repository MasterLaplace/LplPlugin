/**
 * @file cstring.hpp
 * @brief Portable raw-memory operations umbrella. Hosted: lpl::pmr aliases the
 *        std:: functions. Kernel: lowers to the compiler builtins (which the
 *        freestanding toolchain always provides and which fold to the kernel
 *        libc's memcpy/memset), since <cstring> is not in the freestanding
 *        libstdc++ subset.
 *
 * These are non-authoritative byte copies (component-buffer shuffles, SoA
 * double-buffer publishes); they carry no determinism contract of their own —
 * the bytes they move are already fixed by the deterministic sim state.
 *
 * Call sites must use lpl::pmr::memcpy / memset, never std::, so the routing
 * stays a single edit.
 */
#pragma once

#ifndef LPL_STD_CSTRING_HPP
#    define LPL_STD_CSTRING_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
namespace lpl::pmr {

inline void *memcpy(void *dest, const void *src, __SIZE_TYPE__ count) noexcept
{
    return __builtin_memcpy(dest, src, count);
}

inline void *memset(void *dest, int value, __SIZE_TYPE__ count) noexcept
{
    return __builtin_memset(dest, value, count);
}

} // namespace lpl::pmr
#    else
#        include <cstring>
namespace lpl::pmr {
using ::std::memcpy;
using ::std::memset;
} // namespace lpl::pmr
#    endif

#endif // LPL_STD_CSTRING_HPP
