#include <lpl/concurrency/ThreadPool.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

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

int printVerdict()
{
    if (failures == 0)
        std::printf("ALL PASS (0 failures, %d checks)\n", checks);
    else
        std::printf("FAILED (%d failures, %d checks)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}

constexpr lpl::core::u32 kWorkersPerPool = 4u;
constexpr lpl::core::u32 kPoolsPerPhase = 2'000u;
constexpr lpl::core::u32 kDetachedTasksPerPool = 8u;
constexpr std::chrono::seconds kDeadline{60};

/**
 * @brief How far the stress has gone, for the watchdog to say where it stuck.
 */
struct StressProgress {
    std::atomic<const char *> phase{"starting"};      /**< Name of the phase running. */
    std::atomic<lpl::core::u32> poolsDoneInPhase{0u}; /**< Pools of that phase whose shutdown returned. */
};

/**
 * @brief Builds a pool and lets its destructor shut it down at once, while its workers are still
 *        starting or going to sleep: the moment a wake-up sent without the lock is lost.
 */
void buildAndDestroyIdlePool() { const lpl::concurrency::ThreadPool pool{kWorkersPerPool}; }

/**
 * @brief Waits for one task submitted to an idle pool, then shuts the pool down: the worker that ran
 *        the task goes back to sleep just as the shutdown starts.
 */
void awaitOneTaskBeforeShutdown()
{
    lpl::concurrency::ThreadPool pool{kWorkersPerPool};
    pool.enqueue([] {}).get();
}

bool drainsDetachedTasksOnShutdown()
{
    std::atomic<lpl::core::u32> tasksRun{0u};
    lpl::concurrency::ThreadPool pool{kWorkersPerPool};
    for (lpl::core::u32 task = 0u; task < kDetachedTasksPerPool; ++task)
        pool.enqueueDetached([&tasksRun] { tasksRun.fetch_add(1u, std::memory_order_relaxed); });
    pool.shutdown();
    return tasksRun.load(std::memory_order_relaxed) == kDetachedTasksPerPool;
}

template <typename PoolLifetime> void runPhase(StressProgress &progress, const char *phase, PoolLifetime poolLifetime)
{
    progress.poolsDoneInPhase.store(0u);
    progress.phase.store(phase);
    for (lpl::core::u32 index = 0u; index < kPoolsPerPhase; ++index)
    {
        poolLifetime();
        progress.poolsDoneInPhase.store(index + 1u);
    }
}

/**
 * @brief Runs the three phases of the stress, one pool after the other.
 *
 * @param progress Where the stress is, updated after every pool.
 * @return The pools of the last phase whose detached tasks had all run when shutdown returned.
 */
lpl::core::u32 runStress(StressProgress &progress)
{
    runPhase(progress, "idle pools built and destroyed", buildAndDestroyIdlePool);
    runPhase(progress, "one task awaited before shutdown", awaitOneTaskBeforeShutdown);

    lpl::core::u32 poolsDrained = 0u;
    runPhase(progress, "detached tasks drained by shutdown", [&poolsDrained] {
        if (drainsDetachedTasksOnShutdown())
            ++poolsDrained;
    });
    return poolsDrained;
}

/**
 * @brief Says where the stress stuck, prints the verdict and leaves at once.
 *
 * A worker that missed its wake-up never returns, so the stress thread cannot be joined, and
 * destroying a joinable std::thread would terminate the process without a verdict.
 *
 * @param progress Where the stress was when the deadline passed.
 */
[[noreturn]] void abandonStuckStress(const StressProgress &progress)
{
    std::printf("  stuck in phase \"%s\" after %u of %u pools shut down, %lld s after the start\n",
                progress.phase.load(), progress.poolsDoneInPhase.load(), kPoolsPerPhase,
                static_cast<long long>(kDeadline.count()));
    printVerdict();
    std::fflush(stdout);
    std::_Exit(EXIT_FAILURE);
}

} // namespace

int main()
{
    StressProgress progress;
    std::promise<lpl::core::u32> finished;
    std::future<lpl::core::u32> poolsDrained = finished.get_future();
    std::thread stress{[&progress, &finished] { finished.set_value(runStress(progress)); }};

    const bool finishedInTime = poolsDrained.wait_for(kDeadline) == std::future_status::ready;
    expect(finishedInTime, "every pool finishes before the deadline: no worker misses its wake-up");
    if (!finishedInTime)
        abandonStuckStress(progress);
    stress.join();

    expect(poolsDrained.get() == kPoolsPerPhase, "shutdown returns only once every task submitted before it has run");

    return printVerdict();
}
