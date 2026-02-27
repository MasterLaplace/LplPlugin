/**
 * @file Expected.hpp
 * @brief Monadic error-handling type built on std::expected.
 *
 * Provides Expected<T> as an alias for std::expected<T, Error> and a
 * convenience LPL_TRY macro for early-return propagation, mirroring
 * Rust's ? operator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_EXPECTED_HPP
    #define LPL_CORE_EXPECTED_HPP

    #include "Error.hpp"

    #include <expected>

namespace lpl::core {

/**
 * @brief Alias for an expected value or a structured Error.
 * @tparam T The success-path value type.
 */
template <typename T>
using Expected = std::expected<T, Error>;

/**
 * @brief Alias for operations that succeed with no value.
 */
using ExpectedVoid = Expected<void>;

} // namespace lpl::core

/**
 * @brief Propagate an error from an Expected expression.
 *
 * Evaluates @p expr once.  If the result holds an error, the enclosing
 * function immediately returns that error wrapped in an unexpected.
 * Otherwise the macro yields the contained value.
 *
 * @param expr An expression of type lpl::core::Expected<U>.
 */
#define LPL_TRY(expr)                                                     \
    ({                                                                     \
        auto &&_lpl_result = (expr);                                       \
        if (!_lpl_result.has_value()) [[unlikely]]                         \
            return std::unexpected(std::move(_lpl_result.error()));         \
        std::move(_lpl_result.value());                                    \
    })

/**
 * @brief Propagate an error from an ExpectedVoid expression.
 * @param expr An expression of type lpl::core::ExpectedVoid.
 */
#define LPL_TRY_VOID(expr)                                                \
    do {                                                                    \
        auto &&_lpl_result = (expr);                                       \
        if (!_lpl_result.has_value()) [[unlikely]]                         \
            return std::unexpected(std::move(_lpl_result.error()));         \
    } while (false)

#endif // LPL_CORE_EXPECTED_HPP
