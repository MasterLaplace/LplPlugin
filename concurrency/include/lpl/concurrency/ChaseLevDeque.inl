#ifndef LPL_CONCURRENCY_CHASELEVDEQUE_INL
#define LPL_CONCURRENCY_CHASELEVDEQUE_INL

namespace lpl::concurrency {

template <typename T>
inline ChaseLevDeque<T>::ChaseLevDeque(core::u32 capacity) : _mask(capacity - 1), _buffer(new std::atomic<T>[capacity])
{
    LPL_ASSERT((capacity & (capacity - 1)) == 0 && "Capacity must be a power of two");
    for (core::u32 i = 0; i < capacity; ++i)
    {
        _buffer[i].store(nullptr, std::memory_order_relaxed);
    }
}

template <typename T> inline void ChaseLevDeque<T>::push(T item)
{
    core::i64 b = _bottom.load(std::memory_order_relaxed);
    _buffer[b & _mask].store(item, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    _bottom.store(b + 1, std::memory_order_relaxed);
}

template <typename T> inline T ChaseLevDeque<T>::pop()
{
    core::i64 b = _bottom.load(std::memory_order_relaxed) - 1;
    _bottom.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    core::i64 t = _top.load(std::memory_order_relaxed);

    if (t <= b)
    {
        T item = _buffer[b & _mask].load(std::memory_order_relaxed);
        if (t == b)
        {
            if (!_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
            {
                item = nullptr;
            }
            _bottom.store(b + 1, std::memory_order_relaxed);
        }
        return item;
    }
    else
    {
        _bottom.store(b + 1, std::memory_order_relaxed);
        return nullptr;
    }
}

template <typename T> inline T ChaseLevDeque<T>::steal()
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

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_CHASELEVDEQUE_INL
