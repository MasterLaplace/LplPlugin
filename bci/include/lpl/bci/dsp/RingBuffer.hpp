/**
 * @file RingBuffer.hpp
 * @brief Lock-free SPSC ring buffer backed by boost::lockfree::spsc_queue.
 * @author MasterLaplace
 *
 * Provides a type-safe, wait-free Single-Producer Single-Consumer (SPSC)
 * ring buffer suitable for inter-thread communication in the BCI
 * acquisition pipeline. The capacity must be a power of two.
 *
 * @see https://www.boost.org/doc/libs/release/doc/html/lockfree.html
 */

#pragma once

#include <bit>
#include <boost/lockfree/spsc_queue.hpp>
#include <cstddef>
#include <utility>

namespace bci::dsp {

/**
 * @brief Wait-free SPSC ring buffer for inter-thread data transfer.
 *
 * @tparam T        Element type (must be trivially copyable)
 * @tparam Capacity Maximum number of elements (must be a power of 2)
 *
 * Thread safety: exactly one producer thread calling push(), and exactly
 * one consumer thread calling pop()/drain(). Both operations are wait-free.
 *
 * @code
 *   RingBuffer<float, 4096> buffer;
 *
 *   // Producer thread:
 *   buffer.push(42.0f);
 *
 *   // Consumer thread:
 *   float val;
 *   if (buffer.pop(val)) { ... }
 * @endcode
 */
template <typename T, std::size_t Capacity = 4096>
    requires (std::has_single_bit(Capacity))
class RingBuffer {
public:
    RingBuffer();

    /**
     * @brief Enqueues an element (producer side).
     *
     * @param item The element to enqueue
     * @return true if enqueued successfully, false if the buffer is full
     */
    bool push(const T &item) noexcept;

    /**
     * @brief Dequeues an element (consumer side).
     *
     * @param item Receives the dequeued element on success
     * @return true if an element was dequeued, false if the buffer is empty
     */
    bool pop(T &item) noexcept;

    /**
     * @brief Attempts to dequeue without blocking.
     *
     * @param item Receives the dequeued element on success
     * @return true if an element was available and dequeued
     */
    bool tryPop(T &item) noexcept;

    /**
     * @brief Dequeues all available elements, invoking callback for each.
     *
     * @tparam Func Callable with signature void(const T&)
     * @param callback Invoked once per dequeued element
     * @return Number of elements drained
     */
    template <typename Func>
    std::size_t drain(Func &&callback) noexcept;

    /**
     * @brief Returns an approximate count of elements in the buffer.
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief Returns true if the buffer appears empty.
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Returns the fixed capacity of the buffer.
     */
    [[nodiscard]] static constexpr std::size_t capacity() noexcept;

private:
    boost::lockfree::spsc_queue<T> _queue;
};

} // namespace bci::dsp

#include "RingBuffer.inl"

namespace lpl::bci {

} // namespace lpl::bci
