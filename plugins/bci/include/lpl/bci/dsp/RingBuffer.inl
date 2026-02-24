/**
 * @file RingBuffer.inl
 * @brief Template implementation for RingBuffer<T, Capacity>.
 * @author MasterLaplace
 */

#pragma once

namespace lpl::bci::dsp {

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
RingBuffer<T, Capacity>::RingBuffer()
    : _queue(Capacity)
{
}

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
bool RingBuffer<T, Capacity>::push(const T &item) noexcept
{
    return _queue.push(item);
}

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
bool RingBuffer<T, Capacity>::pop(T &item) noexcept
{
    return _queue.pop(item);
}

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
bool RingBuffer<T, Capacity>::tryPop(T &item) noexcept
{
    return _queue.pop(item);
}

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
template <typename Func>
std::size_t RingBuffer<T, Capacity>::drain(Func &&callback) noexcept
{
    std::size_t count = 0;
    T item;
    while (_queue.pop(item)) {
        callback(item);
        ++count;
    }
    return count;
}

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
std::size_t RingBuffer<T, Capacity>::size() const noexcept
{
    return _queue.read_available();
}

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
bool RingBuffer<T, Capacity>::empty() const noexcept
{
    return _queue.read_available() == 0;
}

template <typename T, std::size_t Capacity>
    requires (std::has_single_bit(Capacity))
constexpr std::size_t RingBuffer<T, Capacity>::capacity() noexcept
{
    return Capacity;
}

} // namespace lpl::bci::dsp
