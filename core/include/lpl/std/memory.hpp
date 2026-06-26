/**
 * @file memory.hpp
 * @brief Portable smart-pointer umbrella. unique_ptr is in the freestanding
 *        libstdc++ subset on both targets, but std::make_unique is NOT, so the
 *        kernel arm supplies it over the C++ runtime's operator new (defined in
 *        kernel/cxx/cxx_runtime.cpp). Use lpl::pmr::unique_ptr / make_unique at
 *        call sites instead of std:: so the same source compiles for both.
 *
 * Only the single-object form of make_unique is provided; the engine does not
 * use the array form (make_unique<T[]>), so it is intentionally omitted.
 *
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_STD_MEMORY_HPP
#    define LPL_STD_MEMORY_HPP

#    include <lpl/core/Platform.hpp>

#    include <memory>

namespace lpl::pmr {

using ::std::unique_ptr;

#    if LPL_TARGET_KERNEL
template <typename T, typename... Args>
[[nodiscard]] unique_ptr<T> make_unique(Args &&...args)
{
    return unique_ptr<T>(new T(static_cast<Args &&>(args)...));
}
#    else
using ::std::make_unique;
#    endif

} // namespace lpl::pmr

#endif // LPL_STD_MEMORY_HPP
