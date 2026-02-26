/**
 * @file RingBuffer.hpp
 * @brief Single-producer single-consumer lock-free ring buffer.
 *
 * Capacity must be a power of two so that modular arithmetic reduces to
 * a single bitwise AND.  The buffer is suitable for inter-thread
 * communication between the network and simulation threads.
 *
 * @tparam T        Element type (must be trivially copyable).
 * @tparam Capacity Number of slots (compile-time, power of two).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CONTAINER_RING_BUFFER_HPP
    #define LPL_CONTAINER_RING_BUFFER_HPP

    #include <lpl/core/Types.hpp>

    #include <atomic>
    #include <array>
    #include <span>
    #include <type_traits>

namespace lpl::container {

/**
 * @brief SPSC lock-free circular buffer.
 *
 * Uses std::memory_order_acquire / release for the head and tail
 * indices, providing inter-thread visibility guarantees without
 * full sequential consistency overhead.
 */
template <typename T, core::usize Capacity>
    requires (std::has_single_bit(Capacity) && std::is_trivially_copyable_v<T>)
class RingBuffer final {
public:
    /**
     * @brief Push one element into the buffer.
     * @param item Element to enqueue.
     * @return True on success, false if buffer is full.
     */
    bool push(const T &item);

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

    [[nodiscard]] bool        isFull()  const;
    [[nodiscard]] bool        isEmpty() const;
    [[nodiscard]] core::usize size()    const;

private:
    static constexpr core::usize kMask = Capacity - 1;

    std::array<T, Capacity>   _buffer{};
    std::atomic<core::usize>  _head{0};
    std::atomic<core::usize>  _tail{0};
};

} // namespace lpl::container

    #include "RingBuffer.inl"

#endif // LPL_CONTAINER_RING_BUFFER_HPP
