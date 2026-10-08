/**
 * @file RingBuffer.inl
 * @brief Template implementation of the SPSC lock-free ring buffer.
 * @see   RingBuffer.hpp
 */

#ifndef LPL_CONTAINER_RING_BUFFER_INL
#define LPL_CONTAINER_RING_BUFFER_INL

namespace lpl::container {

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
template <typename Fn>
void RingBuffer<T, C>::forEachSlot(core::u32 first, core::u32 count,
                                   Fn &&fn) noexcept(std::is_nothrow_invocable_v<Fn &, T *, core::u32>)
{
    const core::u32 start = first & kMask;
    const core::u32 untilWrap = static_cast<core::u32>(C) - start;
    const core::u32 firstRun = count < untilWrap ? count : untilWrap;

    for (core::u32 offset = 0u; offset < firstRun; ++offset)
        fn(&_slots[start + offset].value, offset);
    for (core::u32 offset = firstRun; offset < count; ++offset)
        fn(&_slots[offset - firstRun].value, offset);
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
core::u32 RingBuffer<T, C>::freeSlots(core::u32 writeIndex, core::u32 wanted) noexcept
{
    core::u32 free = static_cast<core::u32>(C) - (writeIndex - _producer.cachedReadIndex);

    if (free < wanted)
    {
        _producer.cachedReadIndex = _consumer.readIndex.load(std::memory_order_acquire);
        free = static_cast<core::u32>(C) - (writeIndex - _producer.cachedReadIndex);
    }
    return free;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
core::u32 RingBuffer<T, C>::readySlots(core::u32 readIndex, core::u32 wanted) noexcept
{
    core::u32 ready = _consumer.cachedWriteIndex - readIndex;

    if (ready < wanted)
    {
        _consumer.cachedWriteIndex = _producer.writeIndex.load(std::memory_order_acquire);
        ready = _consumer.cachedWriteIndex - readIndex;
    }
    return ready;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
void RingBuffer<T, C>::countRejected(core::usize count) noexcept
{
    if (count != 0u)
        _producer.rejected.store(_producer.rejected.load(std::memory_order_relaxed) + static_cast<core::u32>(count),
                                 std::memory_order_relaxed);
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
RingBuffer<T, C>::~RingBuffer()
{
    if constexpr (!std::is_trivially_destructible_v<T>)
    {
        const core::u32 readIndex = _consumer.readIndex.load(std::memory_order_relaxed);
        const core::u32 writeIndex = _producer.writeIndex.load(std::memory_order_relaxed);

        forEachSlot(readIndex, writeIndex - readIndex, [](T *element, core::u32) noexcept { element->~T(); });
    }
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
template <typename... Args>
bool RingBuffer<T, C>::emplace(Args &&...args) noexcept(std::is_nothrow_constructible_v<T, Args &&...>)
{
    const core::u32 writeIndex = _producer.writeIndex.load(std::memory_order_relaxed);

    if (freeSlots(writeIndex, 1u) == 0u)
    {
        countRejected(1u);
        return false;
    }
    ::new (static_cast<void *>(&_slots[writeIndex & kMask].value)) T(std::forward<Args>(args)...);
    _producer.writeIndex.store(writeIndex + 1u, std::memory_order_release);
    return true;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
bool RingBuffer<T, C>::push(const T &item) noexcept(std::is_nothrow_copy_constructible_v<T>)
{
    return emplace(item);
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
bool RingBuffer<T, C>::push(T &&item) noexcept(std::is_nothrow_move_constructible_v<T>)
{
    return emplace(std::move(item));
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
core::usize RingBuffer<T, C>::pushBulk(std::span<const T> items) noexcept(std::is_nothrow_copy_constructible_v<T>)
{
    const core::u32 writeIndex = _producer.writeIndex.load(std::memory_order_relaxed);
    const core::u32 wanted = items.size() < C ? static_cast<core::u32>(items.size()) : static_cast<core::u32>(C);
    const core::u32 free = freeSlots(writeIndex, wanted);
    const core::u32 count = wanted < free ? wanted : free;

    if constexpr (std::is_nothrow_copy_constructible_v<T>)
    {
        forEachSlot(writeIndex, count, [&items](T *slot, core::u32 offset) noexcept {
            ::new (static_cast<void *>(slot)) T(items[offset]);
        });
        _producer.writeIndex.store(writeIndex + count, std::memory_order_release);
    }
    else
    {
        for (core::u32 offset = 0u; offset < count; ++offset)
        {
            ::new (static_cast<void *>(&_slots[(writeIndex + offset) & kMask].value)) T(items[offset]);
            _producer.writeIndex.store(writeIndex + offset + 1u, std::memory_order_release);
        }
    }
    countRejected(items.size() - count);
    return count;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
bool RingBuffer<T, C>::pop(T &item) noexcept(std::is_nothrow_move_assignable_v<T>)
{
    const core::u32 readIndex = _consumer.readIndex.load(std::memory_order_relaxed);

    if (readySlots(readIndex, 1u) == 0u)
        return false;

    T &element = _slots[readIndex & kMask].value;
    item = std::move(element);
    element.~T();
    _consumer.readIndex.store(readIndex + 1u, std::memory_order_release);
    return true;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
T *RingBuffer<T, C>::front() noexcept
{
    const core::u32 readIndex = _consumer.readIndex.load(std::memory_order_relaxed);

    return readySlots(readIndex, 1u) == 0u ? nullptr : &_slots[readIndex & kMask].value;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
bool RingBuffer<T, C>::popFront() noexcept
{
    const core::u32 readIndex = _consumer.readIndex.load(std::memory_order_relaxed);

    if (readySlots(readIndex, 1u) == 0u)
        return false;

    _slots[readIndex & kMask].value.~T();
    _consumer.readIndex.store(readIndex + 1u, std::memory_order_release);
    return true;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
template <typename Visit>
requires std::is_nothrow_invocable_v<Visit &, T &>
core::usize RingBuffer<T, C>::consume(Visit &&visit, core::usize maxCount) noexcept
{
    const core::u32 readIndex = _consumer.readIndex.load(std::memory_order_relaxed);
    const core::u32 wanted = maxCount < C ? static_cast<core::u32>(maxCount) : static_cast<core::u32>(C);
    const core::u32 ready = readySlots(readIndex, wanted);
    const core::u32 count = wanted < ready ? wanted : ready;

    if (count == 0u)
        return 0u;

    forEachSlot(readIndex, count, [&visit](T *element, core::u32) noexcept {
        visit(*element);
        element->~T();
    });
    _consumer.readIndex.store(readIndex + count, std::memory_order_release);
    return count;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
core::usize RingBuffer<T, C>::drain(std::span<T> out) noexcept(std::is_nothrow_move_assignable_v<T>)
{
    if constexpr (std::is_nothrow_move_assignable_v<T>)
    {
        core::usize filled = 0u;

        return consume([&out, &filled](T &element) noexcept { out[filled++] = std::move(element); }, out.size());
    }
    else
    {
        core::usize filled = 0u;

        while (filled < out.size() && pop(out[filled]))
            ++filled;
        return filled;
    }
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
core::usize RingBuffer<T, C>::size() const noexcept
{
    const core::u32 readIndex = _consumer.readIndex.load(std::memory_order_acquire);
    const core::u32 writeIndex = _producer.writeIndex.load(std::memory_order_acquire);
    const core::u32 used = writeIndex - readIndex;

    return used < C ? used : C;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
bool RingBuffer<T, C>::isEmpty() const noexcept
{
    return size() == 0u;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
bool RingBuffer<T, C>::isFull() const noexcept
{
    return size() == C;
}

template <typename T, core::usize C>
requires(std::has_single_bit(C) && C <= (core::usize{1u} << 31u))
core::u32 RingBuffer<T, C>::rejectedCount() const noexcept
{
    return _producer.rejected.load(std::memory_order_relaxed);
}

} // namespace lpl::container

#endif // LPL_CONTAINER_RING_BUFFER_INL
