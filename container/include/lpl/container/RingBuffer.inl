/**
 * @file RingBuffer.inl
 * @brief Template implementation of the SPSC lock-free ring buffer.
 * @see   RingBuffer.hpp
 */

#ifndef LPL_CONTAINER_RING_BUFFER_INL
#define LPL_CONTAINER_RING_BUFFER_INL

namespace lpl::container {

template <typename T, core::usize C>
requires(std::has_single_bit(C))
bool RingBuffer<T, C>::push(const T &item)
{
    auto tail = _tail.load(std::memory_order_relaxed);
    auto next = (tail + 1) & kMask;

    if (next == _head.load(std::memory_order_acquire))
        return false;

    _buffer[tail] = item;
    _tail.store(next, std::memory_order_release);
    return true;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C))
bool RingBuffer<T, C>::push(T &&item)
{
    auto tail = _tail.load(std::memory_order_relaxed);
    auto next = (tail + 1) & kMask;

    if (next == _head.load(std::memory_order_acquire))
        return false;

    _buffer[tail] = std::move(item);
    _tail.store(next, std::memory_order_release);
    return true;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C))
bool RingBuffer<T, C>::pop(T &item)
{
    auto head = _head.load(std::memory_order_relaxed);

    if (head == _tail.load(std::memory_order_acquire))
        return false;

    // Move out for non-trivial T (steals heap storage); a copy for POD T.
    item = std::move(_buffer[head]);
    _head.store((head + 1) & kMask, std::memory_order_release);
    return true;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C))
core::usize RingBuffer<T, C>::drain(std::span<T> out)
{
    core::usize count = 0;
    while (count < out.size() && pop(out[count]))
        ++count;
    return count;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C))
bool RingBuffer<T, C>::isFull() const
{
    auto next = (_tail.load(std::memory_order_relaxed) + 1) & kMask;
    return next == _head.load(std::memory_order_acquire);
}

template <typename T, core::usize C>
requires(std::has_single_bit(C))
bool RingBuffer<T, C>::isEmpty() const
{
    return _head.load(std::memory_order_acquire) == _tail.load(std::memory_order_acquire);
}

template <typename T, core::usize C>
requires(std::has_single_bit(C))
core::usize RingBuffer<T, C>::size() const
{
    auto h = _head.load(std::memory_order_acquire);
    auto t = _tail.load(std::memory_order_acquire);
    return (t - h) & kMask;
}

} // namespace lpl::container

#endif // LPL_CONTAINER_RING_BUFFER_INL
