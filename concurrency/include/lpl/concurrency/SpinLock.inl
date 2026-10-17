#ifndef LPL_CONCURRENCY_SPINLOCK_INL
#define LPL_CONCURRENCY_SPINLOCK_INL

namespace lpl::concurrency {

inline void SpinLock::lock() noexcept
{
    while (!tryLock())
        waitWhileHeld();
}

inline void SpinLock::waitWhileHeld() const noexcept
{
    core::u32 pausesPerPoll = 1u;
    while (_flag.test(std::memory_order_relaxed))
    {
        for (core::u32 pause = 0u; pause < pausesPerPoll; ++pause)
            LPL_CPU_PAUSE();
        pausesPerPoll = std::min(pausesPerPoll * 2u, kMaximumBackoffPauses);
    }
}

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_SPINLOCK_INL
