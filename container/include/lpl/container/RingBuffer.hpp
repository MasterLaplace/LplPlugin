/**
 * @file RingBuffer.hpp
 * @brief Single-producer single-consumer lock-free ring buffer.
 *
 * One thread pushes and one thread pops. Each end owns one control block: its own index, and its
 * last read of the other end's index. The two blocks and the slots sit kDestructiveInterferenceSize
 * apart, so in steady state an end touches only its own line, and reads the other end's line only
 * when its copy says the ring is full or empty. Indices are free-running 32-bit counters masked into
 * the slots, so a ring of Capacity slots holds Capacity elements.
 *
 * An element is constructed in its slot when pushed and destroyed when popped: the ring never
 * default-constructs a T, and a ring destroyed with elements left destroys them. Nothing allocates
 * and nothing throws unless T's own construction does, so the same ring runs on the host and in
 * ring 0.
 *
 * @tparam T        Element type, constructed from what is pushed.
 * @tparam Capacity Number of slots: a power of two, at most 2^31.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CONTAINER_RING_BUFFER_HPP
#    define LPL_CONTAINER_RING_BUFFER_HPP

#    include <lpl/core/Platform.hpp>
#    include <lpl/core/Types.hpp>

#    include <atomic>
#    include <bit>
#    include <new>
#    include <span>
#    include <type_traits>
#    include <utility>

namespace lpl::container {

/**
 * @brief SPSC lock-free circular buffer with cached indices.
 *
 * The producer calls push, emplace and pushBulk; the consumer calls pop, front, popFront, drain and
 * consume. An end publishes its index with a release store and reads the other end's index with an
 * acquire load, only when its cached copy runs out. size, isEmpty and isFull are snapshots, safe from
 * either end and from a third thread, but stale as soon as they return.
 *
 * What the producer cannot fit is not pushed and is counted in rejectedCount, so a producer that
 * drops the newest element on a full ring may ignore the result of push.
 */
template <typename T, core::usize Capacity>
requires(std::has_single_bit(Capacity) && Capacity <= (core::usize{1u} << 31u))
class RingBuffer final {
public:
    RingBuffer() noexcept = default;
    ~RingBuffer();

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;
    RingBuffer(RingBuffer &&) = delete;
    RingBuffer &operator=(RingBuffer &&) = delete;

    /**
     * @brief Producer: copies @p item into the next slot.
     * @param item Element to enqueue.
     * @return True when pushed, false when the ring is full (counted in rejectedCount).
     */
    bool push(const T &item) noexcept(std::is_nothrow_copy_constructible_v<T>);

    /**
     * @brief Producer: moves @p item into the next slot.
     * @param item Element to enqueue, left moved-from when pushed.
     * @return True when pushed, false when the ring is full (counted in rejectedCount).
     */
    bool push(T &&item) noexcept(std::is_nothrow_move_constructible_v<T>);

    /**
     * @brief Producer: constructs an element in the next slot from @p args, with no copy and no move.
     * @param args Arguments of T's constructor.
     * @return True when constructed, false when the ring is full (counted in rejectedCount).
     */
    template <typename... Args> bool emplace(Args &&...args) noexcept(std::is_nothrow_constructible_v<T, Args &&...>);

    /**
     * @brief Producer: copies as many of @p items as fit, in order, and publishes them at once.
     * @param items Elements to enqueue.
     * @return How many were pushed; the rest are counted in rejectedCount.
     */
    core::usize pushBulk(std::span<const T> items) noexcept(std::is_nothrow_copy_constructible_v<T>);

    /**
     * @brief Consumer: moves the oldest element into @p item and removes it.
     * @param[out] item Receives the element.
     * @return True when an element was popped, false when the ring is empty.
     */
    [[nodiscard]] bool pop(T &item) noexcept(std::is_nothrow_move_assignable_v<T>);

    /**
     * @brief Consumer: the oldest element, in its slot, without removing it.
     * @return The element, valid until popFront, or nullptr when the ring is empty.
     */
    [[nodiscard]] T *front() noexcept;

    /**
     * @brief Consumer: destroys and removes the oldest element.
     * @return True when an element was removed, false when the ring is empty.
     */
    bool popFront() noexcept;

    /**
     * @brief Consumer: moves up to out.size() elements into @p out, in order, and frees their slots at once.
     * @param[out] out Receives the elements.
     * @return How many were moved.
     */
    [[nodiscard]] core::usize drain(std::span<T> out) noexcept(std::is_nothrow_move_assignable_v<T>);

    /**
     * @brief Consumer: hands up to @p maxCount elements to @p visit in place, oldest first, then destroys
     *        them and frees their slots at once.
     * @param visit    Called once per element with a T&; it must not throw.
     * @param maxCount Most elements to visit.
     * @return How many were visited.
     */
    template <typename Visit>
    requires std::is_nothrow_invocable_v<Visit &, T &>
    core::usize consume(Visit &&visit, core::usize maxCount) noexcept;

    /** @brief Elements in the ring, as one snapshot. */
    [[nodiscard]] core::usize size() const noexcept;

    /** @brief True when the snapshot holds no element. */
    [[nodiscard]] bool isEmpty() const noexcept;

    /** @brief True when the snapshot holds Capacity elements. */
    [[nodiscard]] bool isFull() const noexcept;

    /** @brief Elements the producer could not fit since the ring was built. */
    [[nodiscard]] core::u32 rejectedCount() const noexcept;

    /** @brief Number of slots, all usable. */
    [[nodiscard]] static constexpr core::usize capacity() noexcept { return Capacity; }

private:
    static constexpr core::u32 kMask = static_cast<core::u32>(Capacity - 1u);

    /** @brief Storage for one T, which the ring constructs and destroys itself. */
    union Slot {
        Slot() noexcept {}
        ~Slot() {}
        T value; /**< Alive between its push and its pop. */
    };

    /** @brief What the producer writes, alone in its interference span. */
    struct alignas(kDestructiveInterferenceSize) ProducerEnd {
        std::atomic<core::u32> writeIndex{0u}; /**< Next slot to fill, published with release. */
        core::u32 cachedReadIndex = 0u;        /**< Last readIndex the producer read. */
        std::atomic<core::u32> rejected{0u};   /**< Elements that did not fit, written by the producer only. */
    };

    /** @brief What the consumer writes, alone in its interference span. */
    struct alignas(kDestructiveInterferenceSize) ConsumerEnd {
        std::atomic<core::u32> readIndex{0u}; /**< Next slot to read, published with release. */
        core::u32 cachedWriteIndex = 0u;      /**< Last writeIndex the consumer read. */
    };

    static_assert(sizeof(ProducerEnd) == kDestructiveInterferenceSize &&
                      sizeof(ConsumerEnd) == kDestructiveInterferenceSize,
                  "each end of the ring fills exactly one interference span");
    static_assert(std::atomic<core::u32>::is_always_lock_free, "the ring needs lock-free 32-bit atomics");

    /**
     * @brief Calls @p fn on the elements of @p count slots from index @p first, in two contiguous runs
     *        split at the end of the slots.
     */
    template <typename Fn>
    void forEachSlot(core::u32 first, core::u32 count,
                     Fn &&fn) noexcept(std::is_nothrow_invocable_v<Fn &, T *, core::u32>);

    /**
     * @brief Free slots after @p writeIndex, reading the consumer's index again only when the cached copy shows
     *        fewer than @p wanted.
     */
    [[nodiscard]] core::u32 freeSlots(core::u32 writeIndex, core::u32 wanted) noexcept;

    /**
     * @brief Filled slots from @p readIndex, reading the producer's index again only when the cached copy shows
     *        fewer than @p wanted.
     */
    [[nodiscard]] core::u32 readySlots(core::u32 readIndex, core::u32 wanted) noexcept;

    /** @brief Adds @p count to the rejected elements, as their only writer. */
    void countRejected(core::usize count) noexcept;

    ProducerEnd _producer;
    ConsumerEnd _consumer;
    alignas(kDestructiveInterferenceSize) Slot _slots[Capacity];
};

} // namespace lpl::container

#    include "RingBuffer.inl"

#endif // LPL_CONTAINER_RING_BUFFER_HPP
