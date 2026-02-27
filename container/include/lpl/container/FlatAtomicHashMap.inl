/**
 * @file FlatAtomicHashMap.inl
 * @brief Template implementation of lock-free flat hash map.
 * @see   FlatAtomicHashMap.hpp
 */

#ifndef LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_INL
    #define LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_INL

    #include <bit>

namespace lpl::container {

namespace detail {

constexpr core::u64 packEntry(core::u64 key, core::u32 index)
{
    return ((key & ((core::u64{1} << 42) - 1)) << 22) | (index & ((1u << 22) - 1));
}

constexpr core::u64 extractKey(core::u64 packed)
{
    return packed >> 22;
}

constexpr core::u32 extractIndex(core::u64 packed)
{
    return static_cast<core::u32>(packed & ((1u << 22) - 1));
}

} // namespace detail

template <typename V>
FlatAtomicHashMap<V>::FlatAtomicHashMap(core::u32 capacity)
    : _capacity(std::bit_ceil(capacity))
    , _mask(_capacity - 1)
{
    _entries.resize(_capacity);
    for (auto &e : _entries)
        e.store(kEmpty, std::memory_order_relaxed);
    _pool.resize(_capacity);
}

template <typename V>
V *FlatAtomicHashMap<V>::get(core::u64 key)
{
    core::u32 idx = static_cast<core::u32>(key) & _mask;
    for (core::u32 i = 0; i < _capacity; ++i) {
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

template <typename V>
const V *FlatAtomicHashMap<V>::get(core::u64 key) const
{
    return const_cast<FlatAtomicHashMap *>(this)->get(key);
}

template <typename V>
V *FlatAtomicHashMap<V>::insert(core::u64 key)
{
    core::u32 poolIdx = _poolNext.fetch_add(1, std::memory_order_relaxed);
    if (poolIdx >= _capacity)
        return nullptr;

    core::u64 packed = detail::packEntry(key, poolIdx);
    core::u32 slot   = static_cast<core::u32>(key) & _mask;

    for (core::u32 i = 0; i < _capacity; ++i) {
        core::u64 expected = kEmpty;
        auto &entry = _entries[(slot + i) & _mask];
        if (entry.compare_exchange_strong(expected, packed, std::memory_order_release)) {
            _size.fetch_add(1, std::memory_order_relaxed);
            return &_pool[poolIdx];
        }
        if (expected == kTombstone) {
            if (entry.compare_exchange_strong(expected, packed, std::memory_order_release)) {
                _size.fetch_add(1, std::memory_order_relaxed);
                return &_pool[poolIdx];
            }
        }
    }
    return nullptr;
}

template <typename V>
bool FlatAtomicHashMap<V>::remove(core::u64 key)
{
    core::u32 slot = static_cast<core::u32>(key) & _mask;
    for (core::u32 i = 0; i < _capacity; ++i) {
        auto &entry  = _entries[(slot + i) & _mask];
        auto current = entry.load(std::memory_order_acquire);
        if (current == kEmpty)
            return false;
        if (current == kTombstone)
            continue;
        if (detail::extractKey(current) == key) {
            if (entry.compare_exchange_strong(current, kTombstone, std::memory_order_release)) {
                _size.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
        }
    }
    return false;
}

template <typename V>
template <typename Fn>
void FlatAtomicHashMap<V>::forEach(Fn &&fn)
{
    for (core::u32 i = 0; i < _capacity; ++i) {
        auto entry = _entries[i].load(std::memory_order_acquire);
        if (entry == kEmpty || entry == kTombstone)
            continue;
        fn(detail::extractKey(entry), _pool[detail::extractIndex(entry)]);
    }
}

} // namespace lpl::container

#endif // LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_INL
