#pragma once

#include <atomic>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

class SpinLock {
public:
    void lock() noexcept
    {
        while (flag.test_and_set(std::memory_order_acquire))
        {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
            __asm__ __volatile__("yield");
#endif
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
