/**
 * @file unordered_map.hpp
 * @brief Portable unordered_map alias: hosted std::unordered_map, or
 *        kstd::unordered_map on the freestanding kernel target.
 *        Use lpl::pmr::unordered_map at call sites.
 *
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_STD_UNORDERED_MAP_HPP
#    define LPL_STD_UNORDERED_MAP_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
#        include <kstd/unordered_map.hpp>
namespace lpl::pmr {
template <typename Key, typename Value, typename Hash = ::std::hash<Key>, typename KeyEqual = ::std::equal_to<Key>>
using unordered_map = ::kstd::unordered_map<Key, Value, Hash, KeyEqual>;
}
#    else
#        include <unordered_map>
namespace lpl::pmr {
template <typename Key, typename Value, typename Hash = ::std::hash<Key>, typename KeyEqual = ::std::equal_to<Key>>
using unordered_map = ::std::unordered_map<Key, Value, Hash, KeyEqual>;
}
#    endif

#endif // LPL_STD_UNORDERED_MAP_HPP
