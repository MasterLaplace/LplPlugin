/**
 * @file RingBuffer.hpp
 * @brief Single-producer single-consumer lock-free ring buffer.
 *
 * Capacity must be a power of two so that modular arithmetic reduces to
 * a single bitwise AND.  The buffer is suitable for inter-thread
 * communication between the network and simulation threads.
 *
 * A trivially-copyable @p T (POD, e.g. packets on the network↔simulation
 * path) gets the fast path: push/pop are wait-free and never allocate. A
 * non-trivial @p T (e.g. a sample owning a std::vector) is also supported via
 * move semantics — use @ref push(T&&) so element transfer moves rather than
 * copies; such element assignment is lock-free but may allocate/free.
 *
 * @tparam T        Element type (default-constructible; copy- or move-assignable).
 * @tparam Capacity Number of slots (compile-time, power of two).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CONTAINER_RING_BUFFER_HPP
#    define LPL_CONTAINER_RING_BUFFER_HPP

#    include <lpl/core/Types.hpp>

#    include <array>
#    include <atomic>
#    include <bit>
#    include <span>
#    include <type_traits>
#    include <utility>

namespace lpl::container {

/**
 * @brief SPSC lock-free circular buffer.
 *
 * Uses std::memory_order_acquire / release for the head and tail
 * indices, providing inter-thread visibility guarantees without
 * full sequential consistency overhead.
 */
template <typename T, core::usize Capacity>
requires(std::has_single_bit(Capacity))
class RingBuffer final {
public:
    /**
     * @brief Push one element into the buffer by copy.
     * @param item Element to enqueue.
     * @return True on success, false if buffer is full.
     */
    bool push(const T &item);

    /**
     * @brief Push one element into the buffer by move.
     *
     * Preferred for elements that own heap storage (e.g. a std::vector): the
     * element is moved into the slot instead of copied. For a trivially-copyable
     * @p T this is identical to the copy overload.
     *
     * @param item Element to enqueue (left in a moved-from state on success).
     * @return True on success, false if buffer is full.
     */
    bool push(T &&item);

    /**
     * @brief Pop one element from the buffer.
     * @param[out] item Destination for the dequeued element.
     * @return True on success, false if buffer is empty.
     */
    bool pop(T &item);

    /**
     * @brief Drain all available elements into a span.
     * @param[out] out Destination span.
     * @return Number of elements actually drained.
     */
    core::usize drain(std::span<T> out);

    [[nodiscard]] bool isFull() const;
    [[nodiscard]] bool isEmpty() const;
    [[nodiscard]] core::usize size() const;

private:
    static constexpr core::usize kMask = Capacity - 1;

    std::array<T, Capacity> _buffer{};
    std::atomic<core::usize> _head{0};
    std::atomic<core::usize> _tail{0};
};

} // namespace lpl::container

#    include "RingBuffer.inl"

#endif // LPL_CONTAINER_RING_BUFFER_HPP
