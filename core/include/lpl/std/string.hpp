/**
 * @file string.hpp
 * @brief Portable string alias: hosted std::string, or kernel_std::string on the
 *        freestanding kernel target. Use lpl::pmr::string at call sites.
 *
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_STD_STRING_HPP
#    define LPL_STD_STRING_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
#        include <kernel_std/string.hpp>
namespace lpl::pmr {
using string = ::kstd::string;
}
#    else
#        include <string>
namespace lpl::pmr {
using string = ::std::string;
}
#    endif

#endif // LPL_STD_STRING_HPP
