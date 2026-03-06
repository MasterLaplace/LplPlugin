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
#    define LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_HPP

#    include <lpl/concurrency/SpinLock.hpp>
#    include <lpl/core/Assert.hpp>
#    include <lpl/core/Types.hpp>

#    include <atomic>
#    include <memory>
#    include <vector>

namespace lpl::container {

/**
 * @brief Lock-free flat hash map with contiguous value pool and slot recycling.
 *
 * Pool indices are recycled on removal via a free-list, and an activeSlots
 * vector enables O(n_active) iteration instead of O(capacity).
 *
 * @tparam V Value type.
 */
template <typename V> class FlatAtomicHashMap final {
public:
    static constexpr core::u64 kEmpty = 0;
    static constexpr core::u64 kTombstone = ~core::u64{0};
    static constexpr core::u32 kKeyBits = 42;
    static constexpr core::u32 kIndexBits = 22;

    /**
     * @brief Construct the map with a fixed capacity.
     * @param capacity Maximum number of pool entries.
     *                 Map capacity is 2x rounded to power of two.
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
     * @brief Insert with default-constructed value.
     * @param key Morton key.
     * @return Pointer to the value, or the existing value if key already present.
     *         nullptr if pool exhausted.
     */
    V *insert(core::u64 key);

    /**
     * @brief Insert with move semantics — value is fully written to pool
     *        BEFORE the map entry becomes visible (no data race window).
     * @param key Morton key.
     * @param value Value to move into the pool slot.
     * @return Pointer to the stored value, or the existing value if key
     *         already present. nullptr if pool exhausted.
     */
    V *insert(core::u64 key, V &&value);

    /**
     * @brief Lock-free removal via tombstone + pool slot recycling.
     * @param key Morton key.
     * @return True if the entry was found and removed.
     */
    bool remove(core::u64 key);

    /**
     * @brief Iterate over active entries only — O(n_active).
     *        Takes a snapshot of activeSlots under lock, then iterates
     *        lock-free.
     * @tparam Fn Callable(V &value).
     * @param fn Visitor.
     */
    template <typename Fn> void forEach(Fn &&fn);

    /**
     * @brief Parallel iteration over active slots using a thread pool.
     * @tparam TP Thread pool type (must have enqueue() returning future).
     * @tparam Fn Callable(V&) or Callable(V&, core::u32 batchIdx).
     * @param pool Thread pool reference.
     * @param fn Function to execute per element.
     * @param minPerThread Minimum items per batch to avoid overhead.
     */
    template <typename TP, typename Fn> void forEachParallel(TP &pool, Fn &&fn, core::u32 minPerThread = 64);

    /**
     * @brief Copy active pool indices snapshot (thread-safe).
     */
    void snapshotActiveSlots(std::vector<core::u32> &out);

    /**
     * @brief Direct access to a pool element by index.
     * @param poolIdx Must come from snapshotActiveSlots().
     */
    [[nodiscard]] V &getByPoolIndex(core::u32 poolIdx) noexcept;

    [[nodiscard]] core::u32 size() const { return _size.load(std::memory_order_relaxed); }
    [[nodiscard]] core::u32 capacity() const { return _poolCapacity; }

private:
    [[nodiscard]] bool linkKeyToSlot(core::u64 key, core::u32 slotIndex);

    std::unique_ptr<std::atomic<core::u64>[]> _entries;
    std::unique_ptr<V[]> _pool;
    std::vector<core::u32> _freeIndices;
    std::vector<core::u32> _activeSlots;
    mutable concurrency::SpinLock _allocLock;
    std::atomic<core::u32> _size{0};
    core::u32 _poolCapacity = 0;
    core::u32 _mapCapacity = 0;
    core::u32 _mask = 0;
};

} // namespace lpl::container

#    include "FlatAtomicHashMap.inl"

#endif // LPL_CONTAINER_FLAT_ATOMIC_HASH_MAP_HPP
