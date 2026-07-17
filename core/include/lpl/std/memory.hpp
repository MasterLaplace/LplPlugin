/**
 * @file memory.hpp
 * @brief Portable smart-pointer umbrella. unique_ptr is in the freestanding
 *        libstdc++ subset on both targets, but std::make_unique is NOT, so the
 *        kernel arm supplies it over the C++ runtime's operator new (defined in
 *        kernel/cxx/cxx_runtime.cpp). Use lpl::pmr::unique_ptr / make_unique at
 *        call sites instead of std:: so the same source compiles for both.
 *
 * Both the single-object and the unbounded-array forms are provided: the array
 * form backs container::FlatAtomicHashMap, which the kernel build pulls in via
 * ecs::WorldPartition.
 *
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_STD_MEMORY_HPP
#    define LPL_STD_MEMORY_HPP

#    include <lpl/core/Platform.hpp>

#    include <cstddef>
#    include <memory>
#    include <type_traits>

namespace lpl::pmr {

using ::std::unique_ptr;

#    if LPL_TARGET_KERNEL
template <typename T, typename... Args>
requires(!::std::is_array_v<T>)
[[nodiscard]] unique_ptr<T> make_unique(Args &&...args)
{
    return unique_ptr<T>(new T(static_cast<Args &&>(args)...));
}

/// Array form: value-initialises the elements, matching std::make_unique<T[]>.
template <typename T>
requires ::std::is_unbounded_array_v<T>
[[nodiscard]] unique_ptr<T> make_unique(::std::size_t count)
{
    return unique_ptr<T>(new ::std::remove_extent_t<T>[count]());
}
#    else
using ::std::make_unique;
#    endif

} // namespace lpl::pmr

#endif // LPL_STD_MEMORY_HPP
