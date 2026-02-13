/**
 * @file FlatAtomicsHashMap.hpp
 * @brief High-Performance Lock-Free Hash Map for World Partitioning.
 *
 * A specialized hash map designed for:
 * - Zero-Copy storage (Internal Pool/Arena).
 * - Thread-Safe access (Wait-Free reads, Lock-Free map insertion).
 * - Cache coherency (Open Addressing, Linear Probing).
 * - Stable Pointers (Pool storage guarantees pointer validity until removal).
 *
 * Architecture:
 * - Data is stored in a contiguous `_pool` (SoA-like structure friendly to GPU copy).
 * - Indexing is managed by `_map` using 64-bit atomic PackedEntries.
 * - Supports 42-bit Morton Keys and 22-bit Pool Indices (4M items max).
 *
 * @note This implementation uses Tombstones for deletion and supports slot recycling.
 * @warning Key (Morton) is limited to 42 bits. Value (Index) is limited to 22 bits.
 *
 * @author @MasterLaplace
 * @version 1.0
 * @date 2026-02-07
 *
 * @see Inspired by:
 * - @johnBuffer Stable Index Vector (SIV) for the ID/Handle mechanism.
 * - @ryanfleury Arena Allocators for memory layout.
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <algorithm>
#include "SpinLock.hpp"
#if __cpp_lib_bit_cast >= 201806L
#include <bit>
#else
#include <cmath>

namespace std {
    [[nodiscard]] inline uint64_t bit_ceil(uint64_t n) noexcept
    {
        if (n == 0)
            return 1;
        if ((n & (n - 1)) == 0)
            return n;
        return static_cast<uint64_t>(1) << (static_cast<uint64_t>(std::log2(n)) + 1);
    }
}
#endif

template <typename T>
class FlatAtomicsHashMap {
private:
    static constexpr bool ALREADY_EXISTS = false;
    static constexpr bool SUCCESS = true;

private:
    static constexpr uint32_t MAX_CAPACITY = 0x400000; // 2^22
    static constexpr uint64_t TOMBSTONE = 0xFFFFFFFFFFFFFFFFULL;
    static constexpr uint64_t EMPTY = 0u;

private:
    /**
     * @brief Atomic entry packing Key (42 bits) and Pool Index (22 bits).
     * * Layout: [ KKKK... (42) ...KKKK | IIII... (22) ...IIII ]
     */
    struct PackedEntry {
        std::atomic<uint64_t> data;

        static constexpr uint64_t KEY_MASK = 0xFFFFFFFFFFC00000ULL;
        static constexpr uint64_t VAL_MASK = 0x00000000003FFFFFULL;
        static constexpr unsigned KEY_SHIFT = 22u;

        [[nodiscard]] inline uint64_t load_raw() const noexcept
        {
            return data.load(std::memory_order_acquire);
        }

        [[nodiscard]] inline static uint64_t extract_key(uint64_t raw) noexcept
        {
            return (raw & KEY_MASK) >> KEY_SHIFT;
        }

        [[nodiscard]] inline static uint32_t extract_pool_index(uint64_t raw) noexcept
        {
            return static_cast<uint32_t>(raw & VAL_MASK) - 1u;
        }
    };

public:
    /**
     * @brief Constructs the map with a pre-allocated pool.
     * @param maxChunks Maximum number of elements (must fit in 22 bits).
     */
    explicit FlatAtomicsHashMap(const uint32_t maxChunks)
        : _maxItems(maxChunks)
    {
        uint64_t mapCapacity = std::bit_ceil(static_cast<uint64_t>(maxChunks) * 2u);

        _capacityMask = mapCapacity - 1u;
        _map = std::make_unique<PackedEntry[]>(mapCapacity);
        _pool = std::make_unique<T[]>(maxChunks);
        _freeIndices.reserve(maxChunks);

        for (uint32_t index = 0u; index < maxChunks; ++index)
            _freeIndices.push_back((maxChunks - 1u) - index);

        _activeSlots.reserve(maxChunks);
    }

    ~FlatAtomicsHashMap() = default;

    /**
     * @brief Inserts an object into the map using Move Semantics.
     * Thread-Safe.
     */
    T *insert(const uint64_t key, T &&chunk)
    {
        uint32_t slotIndex;
        {
            LocalGuard lock(_allocLock);
            if (_freeIndices.empty())
                return nullptr;

            slotIndex = _freeIndices.back();
            _freeIndices.pop_back();
        }

        _pool[slotIndex] = std::move(chunk);
        if (linkKeyToSlot(key, slotIndex))
        {
            LocalGuard lock(_allocLock);
            _activeSlots.push_back(slotIndex);
            return &_pool[slotIndex];
        }

        {
            LocalGuard lock(_allocLock);
            _freeIndices.push_back(slotIndex);
        }
        return get(key);
    }

    /**
     * @brief Removes an entry and recycles its pool slot.
     * Thread-Safe.
     */
    void remove(const uint64_t key)
    {
        uint32_t index = key & _capacityMask;

    remove_loop:
        uint64_t raw = _map[index].data.load(std::memory_order_acquire);

        if (raw == EMPTY)
            return;

        if (raw == TOMBSTONE)
        {
            index = (index + 1u) & _capacityMask;
            goto remove_loop;
        }

        uint64_t foundKey = PackedEntry::extract_key(raw);

        if (foundKey == key)
        {
            _map[index].data.store(TOMBSTONE, std::memory_order_release);
            uint32_t poolIndex = PackedEntry::extract_pool_index(raw);
            LocalGuard lock(_allocLock);
            _freeIndices.push_back(poolIndex);
            auto it = std::find(_activeSlots.begin(), _activeSlots.end(), poolIndex);
            if (it != _activeSlots.end()) { *it = _activeSlots.back(); _activeSlots.pop_back(); }
            return;
        }

        index = (index + 1u) & _capacityMask;
    goto remove_loop;
    }

    /**
     * @brief Retrieves a pointer to the stored object.
     * Wait-Free.
     */
     [[nodiscard]] T *get(const uint64_t key) const
    {
        uint32_t index = key & _capacityMask;

    get_loop:
        uint64_t raw = _map[index].data.load(std::memory_order_acquire);

        if (raw == EMPTY)
            return nullptr;

        if (raw == TOMBSTONE)
        {
            index = (index + 1u) & _capacityMask;
            goto get_loop;
        }

        uint64_t foundKey = (raw & PackedEntry::KEY_MASK) >> PackedEntry::KEY_SHIFT;

        if (foundKey == key)
        {
            uint32_t poolIndex = PackedEntry::extract_pool_index(raw);
            return &_pool[poolIndex];
        }

        index = (index + 1u) & _capacityMask;
    goto get_loop;
    }

    template <typename Callable>
    void forEach(Callable &&func)
    {
        // Copie des indices sous verrou, itération sans verrou
        std::vector<uint32_t> snapshot;
        {
            LocalGuard lock(_allocLock);
            snapshot = _activeSlots;
        }
        for (uint32_t poolIdx : snapshot)
            func(_pool[poolIdx]);
    }

private:
    /**
     * @brief Tries to link a Key to a Pool Index in the hash map.
     * Handles Linear Probing and Tombstone recycling.
     */
    [[nodiscard]] bool linkKeyToSlot(const uint64_t key, uint32_t slotIndex) const
    {
        uint64_t packed = (key << PackedEntry::KEY_SHIFT) | ((slotIndex + 1u) & PackedEntry::VAL_MASK);
        uint32_t index = key & _capacityMask;

    linkKeyToSlot_loop:
        uint64_t prev = _map[index].data.load(std::memory_order_relaxed);

        if (prev == EMPTY || prev == TOMBSTONE)
        {
            if (_map[index].data.compare_exchange_weak(prev, packed, std::memory_order_release, std::memory_order_relaxed) == true)
                return SUCCESS;
            goto linkKeyToSlot_loop;
        }

        const uint64_t existingKey = (prev & PackedEntry::KEY_MASK) >> PackedEntry::KEY_SHIFT;

        if (existingKey == key)
            return ALREADY_EXISTS;
        index = (index + 1u) & _capacityMask;
    goto linkKeyToSlot_loop;
    }

private:
    uint32_t _maxItems;
    uint64_t _capacityMask;
    std::unique_ptr<PackedEntry[]> _map;
    std::unique_ptr<T[]> _pool;
    std::vector<uint32_t> _freeIndices;
    std::vector<uint32_t> _activeSlots;
    mutable SpinLock _allocLock;
};
