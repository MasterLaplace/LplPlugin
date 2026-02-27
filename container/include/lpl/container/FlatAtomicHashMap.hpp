/**
 * @file FlatAtomicHashMap.hpp
 * @brief Lock-free flat hash map with 64-bit packed atomic entries.
 *
 * Keys are 42-bit Morton codes, values are 22-bit pool indices, packed
 * into a single 64-bit atomic word.  Open addressing with linear probing
 * and tombstone support.  get() is wait-free, insert()/remove() are
 * lock-free (CAS loops).
 *
 * @tparam V Value type stored in a separate contiguous pool.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_HPP
    #define LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_HPP

    #include <lpl/core/Types.hpp>
    #include <lpl/core/Assert.hpp>

    #include <atomic>
    #include <functional>
    #include <vector>

namespace lpl::container {

/**
 * @brief Lock-free flat hash map with contiguous value pool.
 * @tparam V Value type.
 */
template <typename V>
class FlatAtomicHashMap final {
public:
    static constexpr core::u64 kEmpty     = 0;
    static constexpr core::u64 kTombstone = ~core::u64{0};
    static constexpr core::u32 kKeyBits   = 42;
    static constexpr core::u32 kIndexBits = 22;

    /**
     * @brief Construct the map with a fixed capacity.
     * @param capacity Maximum number of entries (must be power of two).
     */
    explicit FlatAtomicHashMap(core::u32 capacity);

    /**
     * @brief Wait-free lookup.
     * @param key Morton key.
     * @return Pointer to value, or nullptr if absent.
     */
    [[nodiscard]] V *get(core::u64 key);
    [[nodiscard]] const V *get(core::u64 key) const;

    /**
     * @brief Lock-free insertion (CAS).
     * @param key Morton key.
     * @return Pointer to the newly inserted value, or nullptr on failure.
     */
    V *insert(core::u64 key);

    /**
     * @brief Lock-free removal via tombstone.
     * @param key Morton key.
     * @return True if the entry was found and removed.
     */
    bool remove(core::u64 key);

    /**
     * @brief Iterate over all live entries under a lock snapshot.
     * @tparam Fn Callable(core::u64 key, V &value).
     * @param fn Visitor.
     */
    template <typename Fn>
    void forEach(Fn &&fn);

    [[nodiscard]] core::u32 size()     const { return _size.load(std::memory_order_relaxed); }
    [[nodiscard]] core::u32 capacity() const { return _capacity; }

private:
    std::vector<std::atomic<core::u64>> _entries;
    std::vector<V>                      _pool;
    std::atomic<core::u32>              _poolNext{0};
    std::atomic<core::u32>              _size{0};
    core::u32                           _capacity = 0;
    core::u32                           _mask     = 0;
};

} // namespace lpl::container

    #include "FlatAtomicHashMap.inl"

#endif // LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_HPP
