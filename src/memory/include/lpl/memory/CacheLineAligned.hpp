/**
 * @file CacheLineAligned.hpp
 * @brief Wrapper enforcing cache-line alignment to prevent false sharing.
 *
 * @tparam T Wrapped value type.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MEMORY_CACHE_LINE_ALIGNED_HPP
    #define LPL_MEMORY_CACHE_LINE_ALIGNED_HPP

    #include <lpl/core/Platform.hpp>

namespace lpl::memory {

/**
 * @brief Aligns the contained value to a full cache line (64 bytes)
 *        to guarantee that two adjacent CacheLineAligned instances
 *        never share a cache line.
 *
 * @tparam T The value type to wrap.
 */
template <typename T>
struct alignas(kCacheLineSize) CacheLineAligned {
    T value;

    CacheLineAligned() = default;
    explicit CacheLineAligned(T v) : value(v) {}

    operator T &()             { return value; }
    operator const T &() const { return value; }
};

} // namespace lpl::memory

#endif // LPL_MEMORY_CACHE_LINE_ALIGNED_HPP
