/**
 * @file Platform.hpp
 * @brief Compile-time platform detection, compiler intrinsics, and
 *        portability macros.
 *
 * Detects the target operating system, CPU architecture, and compiler at
 * preprocessing time. Provides branch-prediction hints, forced inlining,
 * cache-line constants, and the LPL_HD macro for CUDA host+device functions.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_PLATFORM_HPP
#    define LPL_CORE_PLATFORM_HPP

// ---- Operating System ----------------------------------------------------

#    if defined(_WIN32) || defined(_WIN64)
#        define LPL_OS_WINDOWS 1
#    elif defined(__linux__)
#        define LPL_OS_LINUX 1
#    elif defined(__APPLE__)
#        define LPL_OS_MACOS 1
#    elif defined(__ANDROID__)
#        define LPL_OS_ANDROID 1
#    else
#        define LPL_OS_UNKNOWN 1
#    endif

// ---- CPU Architecture ----------------------------------------------------

#    if defined(__x86_64__) || defined(_M_X64)
#        define LPL_ARCH_X64 1
#    elif defined(__aarch64__) || defined(_M_ARM64)
#        define LPL_ARCH_ARM64 1
#    elif defined(__i386__) || defined(_M_IX86)
#        define LPL_ARCH_X86 1
#    else
#        define LPL_ARCH_UNKNOWN 1
#    endif

// ---- Compiler ------------------------------------------------------------

#    if defined(__clang__)
#        define LPL_COMPILER_CLANG 1
#    elif defined(__GNUC__)
#        define LPL_COMPILER_GCC 1
#    elif defined(_MSC_VER)
#        define LPL_COMPILER_MSVC 1
#    else
#        define LPL_COMPILER_UNKNOWN 1
#    endif

// ---- Target runtime ------------------------------------------------------
//
// LPL_TARGET_KERNEL distinguishes the freestanding kernel build (libengine.a,
// linked into LplKernel) from the hosted Linux/dev build. The libengine build
// passes -DLPL_TARGET_KERNEL=1; everything else defaults to the hosted target.
// It selects, via the lpl/std umbrella, whether portable containers/utilities
// resolve to kstd or the hosted std.

#    ifndef LPL_TARGET_KERNEL
#        define LPL_TARGET_KERNEL 0
#    endif

// ---- Intrinsics ----------------------------------------------------------

#    if defined(LPL_COMPILER_GCC) || defined(LPL_COMPILER_CLANG)
#        define LPL_LIKELY(x)   __builtin_expect(!!(x), 1)
#        define LPL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#        define LPL_FORCEINLINE inline __attribute__((always_inline))
#        define LPL_NOINLINE    __attribute__((noinline))
#        define LPL_RESTRICT    __restrict__
#    elif defined(LPL_COMPILER_MSVC)
#        define LPL_LIKELY(x)   (x)
#        define LPL_UNLIKELY(x) (x)
#        define LPL_FORCEINLINE __forceinline
#        define LPL_NOINLINE    __declspec(noinline)
#        define LPL_RESTRICT    __restrict
#    else
#        define LPL_LIKELY(x)   (x)
#        define LPL_UNLIKELY(x) (x)
#        define LPL_FORCEINLINE inline
#        define LPL_NOINLINE
#        define LPL_RESTRICT
#    endif

// ---- CUDA Host+Device ----------------------------------------------------

#    ifdef __CUDACC__
#        define LPL_HD __host__ __device__
#    else
#        define LPL_HD
#    endif

// ---- Cache Line ----------------------------------------------------------

#    include <cstddef>

inline constexpr std::size_t kCacheLineSize = 64;

// ---- CPU Pause Hint ------------------------------------------------------

#    if defined(LPL_ARCH_X64) || defined(LPL_ARCH_X86)
// Use the compiler intrinsic rather than <immintrin.h>: that header drags in
// mm_malloc.h, whose _mm_malloc/_mm_free reference malloc/free unconditionally
// (only <errno.h> is __STDC_HOSTED__-guarded), which the freestanding kernel
// libc does not provide. __builtin_ia32_pause() emits the same PAUSE with no
// header dependency and works on both the hosted and kernel targets.
#        define LPL_CPU_PAUSE() __builtin_ia32_pause()
#    elif defined(LPL_ARCH_ARM64)
#        define LPL_CPU_PAUSE() __asm__ volatile("yield")
#    else
#        define LPL_CPU_PAUSE() ((void) 0)
#    endif

#endif // LPL_CORE_PLATFORM_HPP
