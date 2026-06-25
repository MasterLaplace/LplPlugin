/**
 * @file cstdio.hpp
 * @brief Portable formatted-output umbrella. Hosted: lpl::pmr aliases std::.
 *        Kernel: routes to the kernel diagnostic sink (serial/console).
 *
 * Logging is explicitly NON-authoritative (never part of deterministic sim
 * state), so unlike lpl/std/cmath this seam carries no bit-identical contract.
 *
 * Kernel arm status (P0): the kernel libc currently exposes only `printf`
 * (no `vfprintf`/`vprintf`), so the formatted variadic path is not yet wired.
 * The seam is declared here; the kernel implementation lands together with the
 * console/serial logging HAL. Call sites must use lpl::pmr::fprintf, never
 * std::fprintf, so the routing is a single edit when the sink is ready.
 */
#pragma once

#ifndef LPL_STD_CSTDIO_HPP
#    define LPL_STD_CSTDIO_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
// Intentionally not yet provided on the kernel target. Any translation unit
// that needs formatted output in the freestanding build is gated out of the
// libengine object list until the kernel console logging sink exists.
#        error "lpl/std/cstdio.hpp: kernel formatted-output sink not yet implemented (see P2 console HAL)"
#    else
#        include <cstdio>
namespace lpl::pmr {
using ::std::fprintf;
using ::std::printf;
using ::std::snprintf;
} // namespace lpl::pmr
#    endif

#endif // LPL_STD_CSTDIO_HPP
