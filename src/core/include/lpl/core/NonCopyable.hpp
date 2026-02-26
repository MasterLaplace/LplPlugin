/**
 * @file NonCopyable.hpp
 * @brief CRTP base class that deletes copy operations.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_NON_COPYABLE_HPP
    #define LPL_CORE_NON_COPYABLE_HPP

namespace lpl::core {

/**
 * @brief Inherit (privately) to disable copy construction and assignment.
 * @tparam Derived The CRTP derived class.
 */
template <typename Derived>
class NonCopyable {
protected:
    NonCopyable()  = default;
    ~NonCopyable() = default;

    NonCopyable(const NonCopyable &)            = delete;
    NonCopyable &operator=(const NonCopyable &)  = delete;

    NonCopyable(NonCopyable &&)                 = default;
    NonCopyable &operator=(NonCopyable &&)       = default;
};

} // namespace lpl::core

#endif // LPL_CORE_NON_COPYABLE_HPP
