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
    explicit ChaseLevDeque(core::u32 capacity = 4096) : _mask(capacity - 1), _buffer(new std::atomic<T>[capacity])
    {
        LPL_ASSERT((capacity & (capacity - 1)) == 0 && "Capacity must be a power of two");
        for (core::u32 i = 0; i < capacity; ++i)
        {
            _buffer[i].store(nullptr, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Pushes an item to the bottom of the deque.
     * @note Only the owner thread may call this.
     */
    void push(T item)
    {
        core::i64 b = _bottom.load(std::memory_order_relaxed);
        _buffer[b & _mask].store(item, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        _bottom.store(b + 1, std::memory_order_relaxed);
    }

    /**
     * @brief Pops an item from the bottom of the deque.
     * @note Only the owner thread may call this.
     * @return The popped item, or nullptr if empty.
     */
    T pop()
    {
        core::i64 b = _bottom.load(std::memory_order_relaxed) - 1;
        _bottom.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        core::i64 t = _top.load(std::memory_order_relaxed);

        if (t <= b)
        {
            // Non-empty deque
            T item = _buffer[b & _mask].load(std::memory_order_relaxed);
            if (t == b)
            {
                // Single last element in the deque
                if (!_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
                {
                    // Failed race against steal
                    item = nullptr;
                }
                _bottom.store(b + 1, std::memory_order_relaxed);
            }
            return item;
        }
        else
        {
            // Empty deque
            _bottom.store(b + 1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    /**
     * @brief Steals an item from the top of the deque.
     * @note Any thread may call this.
     * @return The stolen item, or nullptr if empty or contention failed.
     */
    T steal()
    {
        core::i64 t = _top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        core::i64 b = _bottom.load(std::memory_order_acquire);

        if (t < b)
        {
            T item = _buffer[t & _mask].load(std::memory_order_relaxed);
            if (!_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
            {
                return nullptr;
            }
            return item;
        }
        return nullptr;
    }

private:
    alignas(64) std::atomic<core::i64> _top{0};
    alignas(64) std::atomic<core::i64> _bottom{0};
    core::i64 _mask;
    std::unique_ptr<std::atomic<T>[]> _buffer;
};

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_CHASELEVDEQUE_HPP
