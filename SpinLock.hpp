#pragma once

#include <atomic>
#include <immintrin.h>

class SpinLock {
public:
    void lock() noexcept
    {
        while (flag.test_and_set(std::memory_order_acquire))
        {
            _mm_pause();
        }
    }

    void unlock() noexcept
    {
        flag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

class LocalGuard {
public:
    explicit LocalGuard(SpinLock &lock) noexcept : _lock(lock) { _lock.lock(); };
    ~LocalGuard() noexcept { _lock.unlock(); };
    LocalGuard(const SpinLock&) = delete;
    LocalGuard& operator=(const SpinLock&) = delete;

private:
    SpinLock &_lock;
};
