/**
 * @file FlatAtomicHashMap.inl
 * @brief Template implementation of lock-free flat hash map with pool recycling.
 * @see   FlatAtomicHashMap.hpp
 */

#ifndef LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_INL
#define LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_INL

#include <algorithm>
#include <bit>
#include <future>
#include <thread>

namespace lpl::container {

namespace detail {

constexpr core::u64 packEntry(core::u64 key, core::u32 index)
{
    // Pack key (42 bits) | poolIndex+1 (22 bits).
    // +1 so that poolIndex 0 does not collide with kEmpty.
    return ((key & ((core::u64{1} << 42) - 1)) << 22) | (((index + 1u)) & ((1u << 22) - 1));
}

constexpr core::u64 extractKey(core::u64 packed) { return packed >> 22; }

constexpr core::u32 extractIndex(core::u64 packed) { return static_cast<core::u32>(packed & ((1u << 22) - 1)) - 1u; }

} // namespace detail

// ========================================================================== //
//  Constructor                                                               //
// ========================================================================== //

template <typename V>
FlatAtomicHashMap<V>::FlatAtomicHashMap(core::u32 capacity)
    : _poolCapacity(capacity),
      _mapCapacity(static_cast<core::u32>(std::bit_ceil(static_cast<core::u64>(capacity) * 2u))),
      _mask(_mapCapacity - 1)
{
    _entries = std::make_unique<std::atomic<core::u64>[]>(_mapCapacity);
    for (core::u32 i = 0; i < _mapCapacity; ++i)
        _entries[i].store(kEmpty, std::memory_order_relaxed);

    _pool = std::make_unique<V[]>(_poolCapacity);

    _freeIndices.reserve(_poolCapacity);
    for (core::u32 i = 0; i < _poolCapacity; ++i)
        _freeIndices.push_back((_poolCapacity - 1u) - i);

    _activeSlots.reserve(_poolCapacity);
}

// ========================================================================== //
//  Lookup                                                                    //
// ========================================================================== //

template <typename V> V *FlatAtomicHashMap<V>::get(core::u64 key)
{
    core::u32 idx = static_cast<core::u32>(key) & _mask;
    for (core::u32 i = 0; i < _mapCapacity; ++i)
    {
        auto entry = _entries[(idx + i) & _mask].load(std::memory_order_acquire);
        if (entry == kEmpty)
            return nullptr;
        if (entry == kTombstone)
            continue;
        if (detail::extractKey(entry) == key)
            return &_pool[detail::extractIndex(entry)];
    }
    return nullptr;
}

template <typename V> const V *FlatAtomicHashMap<V>::get(core::u64 key) const
{
    return const_cast<FlatAtomicHashMap *>(this)->get(key);
}

// ========================================================================== //
//  Insert (default-constructed)                                              //
// ========================================================================== //

template <typename V> V *FlatAtomicHashMap<V>::insert(core::u64 key)
{
    core::u32 slotIndex;
    {
        concurrency::SpinLockGuard lock(_allocLock);
        if (_freeIndices.empty())
            return nullptr;
        slotIndex = _freeIndices.back();
        _freeIndices.pop_back();
    }

    _pool[slotIndex] = V{};

    if (linkKeyToSlot(key, slotIndex))
    {
        concurrency::SpinLockGuard lock(_allocLock);
        _activeSlots.push_back(slotIndex);
        return &_pool[slotIndex];
    }

    // Key already exists — recycle the slot and return existing value.
    {
        concurrency::SpinLockGuard lock(_allocLock);
        _freeIndices.push_back(slotIndex);
    }
    return get(key);
}

// ========================================================================== //
//  Insert (move semantics)                                                   //
// ========================================================================== //

template <typename V> V *FlatAtomicHashMap<V>::insert(core::u64 key, V &&value)
{
    core::u32 slotIndex;
    {
        concurrency::SpinLockGuard lock(_allocLock);
        if (_freeIndices.empty())
            return nullptr;
        slotIndex = _freeIndices.back();
        _freeIndices.pop_back();
    }

    _pool[slotIndex] = std::move(value);

    if (linkKeyToSlot(key, slotIndex))
    {
        concurrency::SpinLockGuard lock(_allocLock);
        _activeSlots.push_back(slotIndex);
        return &_pool[slotIndex];
    }

    {
        concurrency::SpinLockGuard lock(_allocLock);
        _freeIndices.push_back(slotIndex);
    }
    return get(key);
}

// ========================================================================== //
//  Remove                                                                    //
// ========================================================================== //

template <typename V> bool FlatAtomicHashMap<V>::remove(core::u64 key)
{
    core::u32 slot = static_cast<core::u32>(key) & _mask;
    for (core::u32 i = 0; i < _mapCapacity; ++i)
    {
        auto &entry = _entries[(slot + i) & _mask];
        auto current = entry.load(std::memory_order_acquire);
        if (current == kEmpty)
            return false;
        if (current == kTombstone)
            continue;
        if (detail::extractKey(current) == key)
        {
            if (entry.compare_exchange_strong(current, kTombstone, std::memory_order_release))
            {
                core::u32 poolIndex = detail::extractIndex(current);
                {
                    concurrency::SpinLockGuard lock(_allocLock);
                    _freeIndices.push_back(poolIndex);
                    auto it = std::find(_activeSlots.begin(), _activeSlots.end(), poolIndex);
                    if (it != _activeSlots.end())
                    {
                        *it = _activeSlots.back();
                        _activeSlots.pop_back();
                    }
                }
                _size.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
        }
    }
    return false;
}

// ========================================================================== //
//  linkKeyToSlot (internal)                                                  //
// ========================================================================== //

template <typename V> bool FlatAtomicHashMap<V>::linkKeyToSlot(core::u64 key, core::u32 slotIndex)
{
    core::u64 packed = detail::packEntry(key, slotIndex);
    core::u32 idx = static_cast<core::u32>(key) & _mask;

    for (core::u32 i = 0; i < _mapCapacity; ++i)
    {
        auto &entry = _entries[(idx + i) & _mask];
        core::u64 prev = entry.load(std::memory_order_relaxed);

        if (prev == kEmpty || prev == kTombstone)
        {
            if (entry.compare_exchange_weak(prev, packed, std::memory_order_release, std::memory_order_relaxed))
            {
                _size.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            // CAS failed — retry same slot.
            --i;
            continue;
        }

        if (detail::extractKey(prev) == key)
            return false; // Already exists.
    }
    return false; // Map full.
}

// ========================================================================== //
//  Iteration                                                                 //
// ========================================================================== //

template <typename V> template <typename Fn> void FlatAtomicHashMap<V>::forEach(Fn &&fn)
{
    std::vector<core::u32> snapshot;
    {
        concurrency::SpinLockGuard lock(_allocLock);
        snapshot.reserve(_activeSlots.size());
        snapshot.insert(snapshot.end(), _activeSlots.begin(), _activeSlots.end());
    }
    for (core::u32 poolIdx : snapshot)
        fn(_pool[poolIdx]);
}

template <typename V>
template <typename TP, typename Fn>
void FlatAtomicHashMap<V>::forEachParallel(TP &pool, Fn &&fn, core::u32 minPerThread)
{
    std::vector<core::u32> snapshot;
    {
        concurrency::SpinLockGuard lock(_allocLock);
        snapshot.reserve(_activeSlots.size());
        snapshot.insert(snapshot.end(), _activeSlots.begin(), _activeSlots.end());
    }

    const auto totalItems = static_cast<core::u32>(snapshot.size());
    if (totalItems == 0)
        return;

    core::u32 nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0)
        nThreads = 1;

    core::u32 nBatches = nThreads;
    if (totalItems < nBatches * minPerThread)
    {
        nBatches = (totalItems + minPerThread - 1u) / minPerThread;
        if (nBatches == 0)
            nBatches = 1;
    }

    const core::u32 batchSize = (totalItems + nBatches - 1u) / nBatches;

    if (nBatches == 1)
    {
        for (core::u32 i = 0; i < totalItems; ++i)
        {
            if constexpr (std::is_invocable_v<Fn, V &, core::u32>)
                fn(_pool[snapshot[i]], 0u);
            else
                fn(_pool[snapshot[i]]);
        }
        return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(nBatches);

    for (core::u32 b = 0; b < nBatches; ++b)
    {
        core::u32 start = b * batchSize;
        core::u32 end = std::min(start + batchSize, totalItems);
        if (start >= end)
            break;

        futures.emplace_back(pool.enqueue([this, start, end, b, &snapshot, &fn]() {
            for (core::u32 i = start; i < end; ++i)
            {
                if constexpr (std::is_invocable_v<Fn, V &, core::u32>)
                    fn(_pool[snapshot[i]], b);
                else
                    fn(_pool[snapshot[i]]);
            }
        }));
    }

    for (auto &f : futures)
        f.wait();
}

// ========================================================================== //
//  Utilities                                                                 //
// ========================================================================== //

template <typename V> void FlatAtomicHashMap<V>::snapshotActiveSlots(std::vector<core::u32> &out)
{
    concurrency::SpinLockGuard lock(_allocLock);
    out.clear();
    out.reserve(_activeSlots.size());
    out.insert(out.end(), _activeSlots.begin(), _activeSlots.end());
}

template <typename V> V &FlatAtomicHashMap<V>::getByPoolIndex(core::u32 poolIdx) noexcept { return _pool[poolIdx]; }

} // namespace lpl::container

#endif // LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_INL
