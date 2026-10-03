/**
 * @file test_spinlock.cpp
 * @brief The spin-lock is mutually exclusive: concurrent increments under it lose nothing,
 *        and a held lock refuses a second taker.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-10-03
 * @copyright MIT License
 */

#include <lpl/concurrency/SpinLock.hpp>

#include <cstdio>
#include <thread>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void expect(bool condition, const char *what)
{
    ++checks;
    if (condition)
        return;
    ++failures;
    std::printf("  FAIL  %s\n", what);
}

constexpr lpl::core::u32 kThreads = 8u;
constexpr lpl::core::u32 kAcquisitionsPerThread = 100'000u;

lpl::core::u64 countUnderContention()
{
    lpl::concurrency::SpinLock lock;
    lpl::core::u64 counter = 0u;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (lpl::core::u32 started = 0u; started < kThreads; ++started)
    {
        threads.emplace_back([&lock, &counter] {
            for (lpl::core::u32 acquisition = 0u; acquisition < kAcquisitionsPerThread; ++acquisition)
            {
                lock.lock();
                ++counter;
                lock.unlock();
            }
        });
    }
    for (std::thread &thread : threads)
        thread.join();

    return counter;
}

bool tryLockFromAnotherThread(lpl::concurrency::SpinLock &lock)
{
    bool acquired = false;
    std::thread([&lock, &acquired] { acquired = lock.tryLock(); }).join();
    return acquired;
}

} // namespace

int main()
{
    expect(countUnderContention() == lpl::core::u64{kThreads} * kAcquisitionsPerThread,
           "every increment made under the lock by contending threads is kept");

    lpl::concurrency::SpinLock lock;
    expect(lock.tryLock(), "tryLock takes a free lock");
    expect(!tryLockFromAnotherThread(lock), "tryLock fails on a lock another thread holds");
    expect(!lock.tryLock(), "tryLock fails on a lock its own thread holds: the lock is not reentrant");
    lock.unlock();
    expect(tryLockFromAnotherThread(lock), "tryLock takes the lock again once it is released");
    lock.unlock();

    {
        const lpl::concurrency::SpinLockGuard guard{lock};
        expect(!tryLockFromAnotherThread(lock), "a guard holds the lock for its whole scope");
    }
    expect(tryLockFromAnotherThread(lock), "a guard releases the lock when its scope ends");

    if (failures == 0)
        std::printf("ALL PASS (0 failures, %d checks)\n", checks);
    else
        std::printf("FAILED (%d failures, %d checks)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}
