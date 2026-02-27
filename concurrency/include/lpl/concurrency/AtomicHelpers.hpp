/**
 * @file AtomicHelpers.hpp
 * @brief Typed wrappers over std::atomic with explicit memory orders.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_CONCURRENCY_ATOMICHELPERS_HPP
    #define LPL_CONCURRENCY_ATOMICHELPERS_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/Concepts.hpp>

#include <atomic>

namespace lpl::concurrency {

/**
 * @brief Typed atomic load with explicit memory order.
 * @tparam T Trivially-copyable type.
 * @param atom Atomic to read from.
 * @param order Memory ordering (default: acquire).
 * @return Current value of @p atom.
 */
template <typename T>
    requires core::Blittable<T>
[[nodiscard]] inline T atomicLoad(const std::atomic<T>& atom,
                                  std::memory_order order = std::memory_order_acquire) noexcept
{
    return atom.load(order);
}

/**
 * @brief Typed atomic store with explicit memory order.
 * @tparam T Trivially-copyable type.
 * @param atom Atomic to write to.
 * @param value Value to store.
 * @param order Memory ordering (default: release).
 */
template <typename T>
    requires core::Blittable<T>
inline void atomicStore(std::atomic<T>& atom,
                        T value,
                        std::memory_order order = std::memory_order_release) noexcept
{
    atom.store(value, order);
}

/**
 * @brief Compare-and-swap wrapper returning success/failure.
 * @tparam T Trivially-copyable type.
 * @param atom Target atomic.
 * @param expected Expected value (updated on failure).
 * @param desired Desired new value.
 * @param successOrder Order on success (default: acq_rel).
 * @param failureOrder Order on failure (default: acquire).
 * @return @c true if CAS succeeded.
 */
template <typename T>
    requires core::Blittable<T>
[[nodiscard]] inline bool atomicCas(
    std::atomic<T>& atom,
    T& expected,
    T desired,
    std::memory_order successOrder = std::memory_order_acq_rel,
    std::memory_order failureOrder = std::memory_order_acquire) noexcept
{
    return atom.compare_exchange_weak(expected, desired, successOrder, failureOrder);
}

/**
 * @brief Atomic fetch-and-add.
 * @tparam T Integral type.
 * @param atom Target atomic.
 * @param delta Value to add.
 * @param order Memory ordering (default: acq_rel).
 * @return Previous value of @p atom.
 */
template <typename T>
    requires std::integral<T>
[[nodiscard]] inline T atomicFetchAdd(
    std::atomic<T>& atom,
    T delta,
    std::memory_order order = std::memory_order_acq_rel) noexcept
{
    return atom.fetch_add(delta, order);
}

/**
 * @brief Atomic fetch-and-subtract.
 * @tparam T Integral type.
 * @param atom Target atomic.
 * @param delta Value to subtract.
 * @param order Memory ordering (default: acq_rel).
 * @return Previous value of @p atom.
 */
template <typename T>
    requires std::integral<T>
[[nodiscard]] inline T atomicFetchSub(
    std::atomic<T>& atom,
    T delta,
    std::memory_order order = std::memory_order_acq_rel) noexcept
{
    return atom.fetch_sub(delta, order);
}

/**
 * @brief Full memory fence (sequentially consistent barrier).
 */
inline void atomicFence(std::memory_order order = std::memory_order_seq_cst) noexcept
{
    std::atomic_thread_fence(order);
}

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_ATOMICHELPERS_HPP
