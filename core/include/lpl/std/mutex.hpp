/**
 * @file mutex.hpp
 * @brief Portable mutex aliases. Hosted: std::mutex / lock_guard / unique_lock.
 *        Kernel: kernel_std single-threaded no-op locks (the kernel is
 *        single-threaded first per the convergence plan). Use lpl::pmr::mutex,
 *        lpl::pmr::lock_guard, lpl::pmr::unique_lock at call sites.
 *
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_STD_MUTEX_HPP
#    define LPL_STD_MUTEX_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
#        include <kernel_std/mutex.hpp>
namespace lpl::pmr {
using mutex = ::kstd::mutex;
using recursive_mutex = ::kstd::recursive_mutex;
template <typename Mutex>
using lock_guard = ::kstd::lock_guard<Mutex>;
template <typename Mutex>
using unique_lock = ::kstd::unique_lock<Mutex>;
}
#    else
#        include <mutex>
namespace lpl::pmr {
using mutex = ::std::mutex;
using recursive_mutex = ::std::recursive_mutex;
template <typename Mutex>
using lock_guard = ::std::lock_guard<Mutex>;
template <typename Mutex>
using unique_lock = ::std::unique_lock<Mutex>;
}
#    endif

#endif // LPL_STD_MUTEX_HPP
