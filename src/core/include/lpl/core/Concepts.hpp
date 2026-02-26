/**
 * @file Concepts.hpp
 * @brief C++20 concepts constraining generic interfaces across the engine.
 *
 * These concepts are used as template constraints throughout the codebase
 * to enforce compile-time contracts on type capabilities.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_CONCEPTS_HPP
    #define LPL_CORE_CONCEPTS_HPP

    #include "Types.hpp"

    #include <concepts>
    #include <type_traits>

namespace lpl::core {

/**
 * @brief A type that supports basic arithmetic operations.
 */
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T> || requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
};

/**
 * @brief A type that can be serialized into and deserialized from a byte
 *        stream.
 */
template <typename T>
concept Serializable = requires(const T &val) {
    { val.serializedSize() } -> std::convertible_to<usize>;
};

/**
 * @brief A type that exposes a deterministic hash via a `hash()` member.
 */
template <typename T>
concept Hashable = requires(const T &val) {
    { val.hash() } -> std::convertible_to<u64>;
};

/**
 * @brief A type that is trivially copyable and standard-layout, making it
 *        safe for raw memory operations (memcpy, DMA, GPU upload).
 */
template <typename T>
concept Blittable = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

/**
 * @brief A type that fulfils the BasicLockable named requirement.
 */
template <typename T>
concept Lockable = requires(T &m) {
    { m.lock() };
    { m.unlock() };
};

/**
 * @brief A type usable as a custom STL-compatible allocator.
 */
template <typename A>
concept AllocatorLike = requires(A a, typename A::value_type *p, usize n) {
    { a.allocate(n) }     -> std::same_as<typename A::value_type *>;
    { a.deallocate(p, n) };
};

} // namespace lpl::core

#endif // LPL_CORE_CONCEPTS_HPP
