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
    #define LPL_CONCURRENCY_SPINLOCK_HPP

#include <lpl/core/Platform.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <atomic>

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
class SpinLock final : public core::NonCopyable<SpinLock>
{
public:
    /** @brief Default-constructs in unlocked state. */
    SpinLock() noexcept = default;

    /** @brief Acquires the lock, spinning with CPU back-off. */
    void lock() noexcept
    {
        for (;;)
        {
            if (!_flag.test_and_set(std::memory_order_acquire))
            {
                return;
            }

            while (_flag.test(std::memory_order_relaxed))
            {
                LPL_CPU_PAUSE();
            }
        }
    }

    /**
     * @brief Attempts a single acquire without spinning.
     * @return @c true if the lock was successfully acquired.
     */
    [[nodiscard]] bool tryLock() noexcept
    {
        return !_flag.test_and_set(std::memory_order_acquire);
    }

    /** @brief Releases the lock. */
    void unlock() noexcept
    {
        _flag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag _flag = ATOMIC_FLAG_INIT;
};

/**
 * @class SpinLockGuard
 * @brief RAII guard for SpinLock — acquires on construction, releases on
 *        destruction.
 */
class SpinLockGuard final : public core::NonCopyable<SpinLockGuard>
{
public:
    /**
     * @brief Acquires the given spin-lock.
     * @param lock Reference to the SpinLock to guard.
     */
    explicit SpinLockGuard(SpinLock& lock) noexcept
        : _lock{lock}
    {
        _lock.lock();
    }

    /** @brief Releases the spin-lock. */
    ~SpinLockGuard() noexcept
    {
        _lock.unlock();
    }

private:
    SpinLock& _lock;
};

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_SPINLOCK_HPP
