/**
 * @file Assert.hpp
 * @brief Debug assertions and contract-checking macros with source location.
 *
 * Provides LPL_ASSERT (debug-only), LPL_VERIFY (always evaluated), and
 * LPL_UNREACHABLE (marks provably dead code paths).  In debug builds the
 * macros log the failing expression together with the file, line, and
 * function before aborting.  In release builds LPL_ASSERT is a no-op.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_ASSERT_HPP
    #define LPL_CORE_ASSERT_HPP

    #include "Platform.hpp"

    #include <cstdio>
    #include <cstdlib>
    #include <source_location>

namespace lpl::core::detail {

[[noreturn]] inline void assertFail(
    const char *expr,
    std::source_location loc = std::source_location::current()
) {
    std::fprintf(
        stderr,
        "[LPL ASSERT] %s:%u in %s — \"%s\" failed\n",
        loc.file_name(), loc.line(), loc.function_name(), expr
    );
    std::abort();
}

} // namespace lpl::core::detail

    #ifdef LPL_DEBUG
        #define LPL_ASSERT(cond)                                          \
            do {                                                           \
                if (LPL_UNLIKELY(!(cond)))                                 \
                    ::lpl::core::detail::assertFail(#cond);                \
            } while (false)
    #else
        #define LPL_ASSERT(cond) ((void)0)
    #endif

    #define LPL_VERIFY(cond)                                              \
        do {                                                               \
            if (LPL_UNLIKELY(!(cond)))                                     \
                ::lpl::core::detail::assertFail(#cond);                    \
        } while (false)

    #define LPL_UNREACHABLE()                                             \
        do {                                                               \
            ::lpl::core::detail::assertFail("UNREACHABLE");                \
            __builtin_unreachable();                                       \
        } while (false)

#endif // LPL_CORE_ASSERT_HPP
