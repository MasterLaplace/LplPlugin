/**
 * @file Assert.hpp
 * @brief Debug assertions and contract-checking macros with source location.
 *
 * Provides LPL_ASSERT (debug-only), LPL_VERIFY (always evaluated), and
 * LPL_UNREACHABLE (marks provably dead code paths).  In hosted debug builds
 * the macros log the failing expression together with the file, line, and
 * function before aborting; the kernel build halts without the location.
 * In release builds LPL_ASSERT is a no-op.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_ASSERT_HPP
#    define LPL_CORE_ASSERT_HPP

#    include "Platform.hpp"

#    if LPL_TARGET_KERNEL
#        include <kstd/support.hpp>

namespace lpl::core::detail {

/**
 * @brief Halts the machine on a failed contract, through the kstd halt primitive.
 *
 * The freestanding arm has no hosted C runtime: no stderr, fprintf or abort, and lpl/std/cstdio.hpp
 * stops a kernel build with an error. A failed contract is non-recoverable, so the expression goes
 * straight to ::kstd::fatal. The source location is dropped, since formatting it needs snprintf,
 * which the kernel target does not yet expose; the failing expression is enough to triage.
 *
 * @param expr The failing expression, as text.
 */
[[noreturn]] inline void assertFail(const char *expr) { ::kstd::fatal(expr); }

} // namespace lpl::core::detail
#    else
#        include <cstdio>
#        include <cstdlib>
#        include <source_location>

namespace lpl::core::detail {

/**
 * @brief Prints a failed contract to stderr, with its file, line and function, then aborts.
 *
 * @param expr The failing expression, as text.
 * @param loc  Where the contract failed (filled in by the compiler).
 */
[[noreturn]] inline void assertFail(const char *expr, std::source_location loc = std::source_location::current());

} // namespace lpl::core::detail
#    endif

#    ifdef LPL_DEBUG
#        define LPL_ASSERT(cond)                                                                                       \
            do                                                                                                         \
            {                                                                                                          \
                if (LPL_UNLIKELY(!(cond)))                                                                             \
                    ::lpl::core::detail::assertFail(#cond);                                                            \
            } while (false)
#    else
#        define LPL_ASSERT(cond) ((void) 0)
#    endif

#    define LPL_VERIFY(cond)                                                                                           \
        do                                                                                                             \
        {                                                                                                              \
            if (LPL_UNLIKELY(!(cond)))                                                                                 \
                ::lpl::core::detail::assertFail(#cond);                                                                \
        } while (false)

#    define LPL_UNREACHABLE()                                                                                          \
        do                                                                                                             \
        {                                                                                                              \
            ::lpl::core::detail::assertFail("UNREACHABLE");                                                            \
            __builtin_unreachable();                                                                                   \
        } while (false)

/**
 * @brief Marks a declared-but-unwritten path.
 *
 * Distinct from LPL_UNREACHABLE, which asserts a state the program can never reach: this one asserts a
 * state it will reach, on purpose, until someone implements it. Failing loudly at the call site is what
 * keeps a scaffolded API honest — a stub that silently returns a default looks like a working feature
 * in every test that touches it.
 *
 * @param what A string literal naming what is missing.
 */
#    define LPL_NOT_IMPLEMENTED(what)                                                                                  \
        do                                                                                                             \
        {                                                                                                              \
            ::lpl::core::detail::assertFail("NOT IMPLEMENTED: " what);                                                 \
            __builtin_unreachable();                                                                                   \
        } while (false)

#    include "Assert.inl"

#endif // LPL_CORE_ASSERT_HPP
