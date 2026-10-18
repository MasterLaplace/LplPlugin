/**
 * @file ChaseLevDeque.hpp
 * @brief Lock-free work-stealing deque.
 *
 * Implements the Chase-Lev work-stealing deque algorithm. It allows
 * lock-free push and pop from the bottom (owner thread) and lock-free
 * steal from the top (thief threads).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_CONCURRENCY_CHASELEVDEQUE_HPP
#    define LPL_CONCURRENCY_CHASELEVDEQUE_HPP

#    include <lpl/core/Assert.hpp>
#    include <lpl/core/Types.hpp>

#    include <atomic>
#    include <memory>

namespace lpl::concurrency {

/**
 * @class ChaseLevDeque
 * @brief Lock-free Chase-Lev work-stealing deque for pointers.
 *
 * Capacity must be a power of two.
 *
 * @tparam T Pointer type to store (e.g., Job*).
 */
template <typename T> class ChaseLevDeque {
    static_assert(std::is_pointer_v<T>, "ChaseLevDeque requires pointers to avoid data races on complex types");

public:
    /**
     * @brief Allocates a deque of @p capacity slots, all empty.
     * @param capacity Number of slots: a power of two.
     */
    explicit ChaseLevDeque(core::u32 capacity = 4096);

    /**
     * @brief Pushes an item to the bottom of the deque, unless the deque is full.
     *
     * A thief may have taken an item since the owner last looked, so a deque reported full can have a slot
     * free by the time the call returns; it is never reported free when it is not.
     *
     * @note Only the owner thread may call this.
     * @return True when the item was pushed; false when the deque was full and the item was left out.
     */
    [[nodiscard]] bool push(T item);

    /**
     * @brief Pops an item from the bottom of the deque.
     *
     * When one item is left, the owner races the thieves for it with a compare-and-swap on the top
     * index, and a steal that wins leaves the owner nothing.
     *
     * @note Only the owner thread may call this.
     * @return The popped item, or nullptr if empty or a steal took the last item.
     */
    T pop();

    /**
     * @brief Steals an item from the top of the deque.
     * @note Any thread may call this.
     * @return The stolen item, or nullptr if empty or contention failed.
     */
    T steal();

private:
    alignas(64) std::atomic<core::i64> _top{0};
    alignas(64) std::atomic<core::i64> _bottom{0};
    core::i64 _mask;
    std::unique_ptr<std::atomic<T>[]> _buffer;
};

} // namespace lpl::concurrency

#    include "ChaseLevDeque.inl"

#endif // LPL_CONCURRENCY_CHASELEVDEQUE_HPP
