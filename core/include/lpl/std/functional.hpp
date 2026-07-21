/**
 * @file functional.hpp
 * @brief Portable callable alias. On the hosted target lpl::pmr::function is
 *        std::function; on the freestanding kernel target it is the heap-free,
 *        deterministic kstd::inplace_function (fixed inline buffer). Use
 *        lpl::pmr::function at call sites (e.g. undo/redo commands, callbacks).
 *
 * Note: the kernel binding has a fixed capacity (kstd::inplace_function default).
 * A callable that does not fit is a compile error on the kernel target only —
 * keep captured state small (the engine's command/callback closures already are).
 *
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_STD_FUNCTIONAL_HPP
#    define LPL_STD_FUNCTIONAL_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
#        include <kstd/inplace_function.hpp>
namespace lpl::pmr {
template <typename Signature> using function = ::kstd::inplace_function<Signature>;
}
#    else
#        include <functional>
namespace lpl::pmr {
template <typename Signature> using function = ::std::function<Signature>;
}
#    endif

#endif // LPL_STD_FUNCTIONAL_HPP
