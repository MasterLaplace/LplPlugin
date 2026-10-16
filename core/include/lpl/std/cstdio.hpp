/**
 * @file cstdio.hpp
 * @brief Portable formatted-output umbrella. Hosted: lpl::pmr aliases std::.
 *        Kernel: stops the build with an error.
 *
 * Logging is explicitly NON-authoritative (never part of deterministic sim
 * state), so unlike lpl/std/cmath this seam carries no bit-identical contract.
 *
 * Kernel arm: the kernel libc exposes only `printf` (no `vfprintf`/`vprintf`),
 * so the formatted variadic path is not wired, and the kernel logs through
 * core::Log and the logger LplKernel installs instead. No source includes this
 * header today. Call sites must use lpl::pmr::fprintf, never std::fprintf, so
 * the routing is a single edit if the kernel arm is ever written.
 */
#pragma once

#ifndef LPL_STD_CSTDIO_HPP
#    define LPL_STD_CSTDIO_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
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
