/**
 * @file SpinLock.hpp
 * @brief Lightweight spin-lock using std::atomic_flag with back-off.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_CONCURRENCY_SPINLOCK_HPP
#    define LPL_CONCURRENCY_SPINLOCK_HPP

#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Platform.hpp>
#    include <lpl/core/Types.hpp>

#    include <atomic>

namespace lpl::concurrency {

/**
 * @class SpinLock
 * @brief Test-and-set spin-lock with exponential pause back-off.
 *
 * Suitable for very short critical sections (< 100 cycles). For longer
 * durations prefer a thread-pool or OS mutex.
 *
 * Models the @c Lockable concept (core/Concepts.hpp).
 */
class SpinLock final : public core::NonCopyable<SpinLock> {
public:
    /**
     * Ceiling on one run of pauses. Past it a waiter answers late enough to the release
     * that the back-off costs more than the contention it saves, for sections this short.
     */
    static constexpr core::u32 kMaximumBackoffPauses = 64u;

    /** @brief Default-constructs in unlocked state. */
    SpinLock() noexcept = default;

    /**
     * @brief Acquires the lock, spinning with exponential pause back-off.
     *
     * @details The wait doubles its run of pauses each time it finds the lock still held,
     *          up to @ref kMaximumBackoffPauses, and starts over from one after each failed
     *          acquire. A waiter that re-reads the flag every cycle keeps the cache line
     *          bouncing between cores for the whole wait, which costs the holder the very
     *          bandwidth it needs to finish and let go.
     */
    void lock() noexcept
    {
        for (;;)
        {
            if (!_flag.test_and_set(std::memory_order_acquire))
            {
                return;
            }

            core::u32 backoff = 1u;
            while (_flag.test(std::memory_order_relaxed))
            {
                for (core::u32 pause = 0u; pause < backoff; ++pause)
                {
                    LPL_CPU_PAUSE();
                }
                if (backoff < kMaximumBackoffPauses)
                {
                    backoff <<= 1;
                }
            }
        }
    }

    /**
     * @brief Attempts a single acquire without spinning.
     * @return @c true if the lock was successfully acquired.
     */
    [[nodiscard]] bool tryLock() noexcept { return !_flag.test_and_set(std::memory_order_acquire); }

    /** @brief Releases the lock. */
    void unlock() noexcept { _flag.clear(std::memory_order_release); }

private:
    std::atomic_flag _flag = ATOMIC_FLAG_INIT;
};

/**
 * @class SpinLockGuard
 * @brief RAII guard for SpinLock — acquires on construction, releases on
 *        destruction.
 */
class SpinLockGuard final : public core::NonCopyable<SpinLockGuard> {
public:
    /**
     * @brief Acquires the given spin-lock.
     * @param lock Reference to the SpinLock to guard.
     */
    explicit SpinLockGuard(SpinLock &lock) noexcept : _lock{lock} { _lock.lock(); }

    /** @brief Releases the spin-lock. */
    ~SpinLockGuard() noexcept { _lock.unlock(); }

private:
    SpinLock &_lock;
};

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_SPINLOCK_HPP
