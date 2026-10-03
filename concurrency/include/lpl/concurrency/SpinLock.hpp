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

#    include <algorithm>
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
    /** @brief Default-constructs in unlocked state. */
    SpinLock() noexcept = default;

    /**
     * @brief Acquires the lock, spinning until it is free.
     *
     * @pre The calling thread does not hold it already: the lock is not reentrant, and a
     *      second lock() from its holder spins forever.
     * @pre The caller is not code that can interrupt the holder on the holder's core, such as
     *      an interrupt or a signal handler: it would spin on a lock whose holder cannot resume
     *      until the handler returns.
     * @note Not fair: waiters are not served in arrival order, so under steady contention
     *       one of them can be overtaken again and again.
     */
    void lock() noexcept
    {
        while (!tryLock())
            waitWhileHeld();
    }

    /**
     * @brief Attempts a single acquire without spinning.
     * @return @c true if the lock was successfully acquired.
     */
    [[nodiscard]] bool tryLock() noexcept { return !_flag.test_and_set(std::memory_order_acquire); }

    /** @brief Releases the lock. */
    void unlock() noexcept { _flag.clear(std::memory_order_release); }

private:
    /**
     * Ceiling on the pauses per poll, so that a waiter does not answer the release later
     * than the contention it saves is worth. Not measured yet: the contention benchmark of
     * issue #122 sets the value.
     */
    static constexpr core::u32 kMaximumBackoffPauses = 64u;

    /**
     * Polls with a plain read, which leaves the cache line shared while the lock is held,
     * rather than test_and_set, which would take it exclusive on every try. The pauses per
     * poll grow so that, at the release, the waiters do not all rush the line at once.
     */
    void waitWhileHeld() const noexcept
    {
        core::u32 pausesPerPoll = 1u;
        while (_flag.test(std::memory_order_relaxed))
        {
            for (core::u32 pause = 0u; pause < pausesPerPoll; ++pause)
                LPL_CPU_PAUSE();
            pausesPerPoll = std::min(pausesPerPoll * 2u, kMaximumBackoffPauses);
        }
    }

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
