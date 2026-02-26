/**
 * @file Types.hpp
 * @brief Primitive type aliases and fundamental result types for the engine.
 *
 * Provides fixed-width integer aliases, floating-point aliases, and a
 * minimal Result enumeration used across all modules.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_TYPES_HPP
    #define LPL_CORE_TYPES_HPP

    #include <cstddef>
    #include <cstdint>

namespace lpl::core {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
using f64 = double;

using usize = std::size_t;
using isize = std::ptrdiff_t;

using byte = std::byte;

/**
 * @brief Lightweight status enumeration for operations that cannot fail
 *        gracefully but must report success or failure.
 */
enum class Result : u8 {
    kSuccess = 0,
    kFailure = 1
};

} // namespace lpl::core

#endif // LPL_CORE_TYPES_HPP
