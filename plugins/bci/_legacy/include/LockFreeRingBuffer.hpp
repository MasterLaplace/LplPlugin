// File: LockFreeRingBuffer.hpp
// Description: Production-grade lock-free SPSC ring buffer using boost::lockfree::spsc_queue.
// Provides a standardized, industry-proven alternative to the hand-rolled BciRingBuffer.
//
// Compilation conditionnelle :
//   - LPL_USE_BOOST : utilise boost::lockfree::spsc_queue (recommandé en production)
//   - Par défaut    : wrapper fin autour d'un ring buffer atomique maison
//
// Références :
//   - Boost.Lockfree : https://www.boost.org/doc/libs/release/doc/html/lockfree.html
//   - SPSC guarantees : single-producer single-consumer, wait-free on most architectures
//
// Auteur: MasterLaplace

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifdef LPL_USE_BOOST
// ═══════════════════════════════════════════════════════════════════════════════
//  BOOST IMPLEMENTATION — production-grade, cache-padded, wait-free
// ═══════════════════════════════════════════════════════════════════════════════
#    include <boost/lockfree/spsc_queue.hpp>

/// Lock-free SPSC ring buffer backed by boost::lockfree::spsc_queue.
///
/// Template parameters:
///   T        — Element type (must be trivially copyable)
///   Capacity — Maximum number of elements in the buffer (must be power of 2)
///
/// Thread-safety: exactly one producer thread and one consumer thread.
/// Both push() and pop() are wait-free.
///
/// @code
///   LockFreeRingBuffer<float, 4096> buffer;
///   // Producer thread:
///   buffer.push(42.0f);
///   // Consumer thread:
///   float val;
///   if (buffer.pop(val)) { /* use val */ }
/// @endcode
template <typename T, size_t Capacity = 4096>
class LockFreeRingBuffer {
public:
    LockFreeRingBuffer() : _queue(Capacity) {}

    /// Push an element into the buffer (producer side).
    /// @return true if the element was enqueued, false if the buffer is full.
    bool push(const T &item) noexcept { return _queue.push(item); }

    /// Pop an element from the buffer (consumer side).
    /// @return true if an element was dequeued, false if the buffer is empty.
    bool pop(T &item) noexcept { return _queue.pop(item); }

    /// Returns the number of elements currently in the buffer.
    /// Note: this is an approximation in a concurrent context.
    [[nodiscard]] size_t size() const noexcept { return _queue.read_available(); }

    /// Returns true if the buffer appears empty.
    [[nodiscard]] bool empty() const noexcept { return _queue.read_available() == 0; }

    /// Returns the fixed capacity of the buffer.
    [[nodiscard]] static constexpr size_t capacity() noexcept { return Capacity; }

    /// Drain all available elements into a callback.
    /// @param callback Function called for each element: void(const T&)
    template <typename Func>
    size_t drain(Func &&callback) noexcept
    {
        size_t count = 0;
        T item;
        while (_queue.pop(item))
        {
            callback(item);
            ++count;
        }
        return count;
    }

private:
    boost::lockfree::spsc_queue<T> _queue;
};

#else
// ═══════════════════════════════════════════════════════════════════════════════
//  FALLBACK IMPLEMENTATION — portable atomic ring buffer (no Boost required)
// ═══════════════════════════════════════════════════════════════════════════════

/// Portable lock-free SPSC ring buffer using std::atomic with acquire/release semantics.
///
/// This implementation mirrors the kernel module's RingHeader pattern.
/// It is a valid fallback when Boost is not available, providing the same API.
template <typename T, size_t Capacity = 4096>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

public:
    LockFreeRingBuffer() : _head(0), _tail(0) {}

    bool push(const T &item) noexcept
    {
        const size_t head = _head.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & (Capacity - 1);

        if (next == _tail.load(std::memory_order_acquire))
            return false; // Buffer full

        _buffer[head] = item;
        _head.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T &item) noexcept
    {
        const size_t tail = _tail.load(std::memory_order_relaxed);

        if (tail == _head.load(std::memory_order_acquire))
            return false; // Buffer empty

        item = _buffer[tail];
        _tail.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_acquire);
        return (head - tail + Capacity) & (Capacity - 1);
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    [[nodiscard]] static constexpr size_t capacity() noexcept { return Capacity; }

    template <typename Func>
    size_t drain(Func &&callback) noexcept
    {
        size_t count = 0;
        T item;
        while (pop(item))
        {
            callback(item);
            ++count;
        }
        return count;
    }

private:
    alignas(64) std::atomic<size_t> _head;
    alignas(64) std::atomic<size_t> _tail;
    std::array<T, Capacity> _buffer;
};

#endif // LPL_USE_BOOST
