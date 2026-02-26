// /////////////////////////////////////////////////////////////////////////////
/// @file JobSystem.hpp
/// @brief Work-stealing job system with per-thread deques and counters.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lpl::concurrency {

// /////////////////////////////////////////////////////////////////////////////
/// @struct JobHandle
/// @brief Opaque handle used to wait on a group of jobs via an atomic counter.
// /////////////////////////////////////////////////////////////////////////////
struct JobHandle
{
    std::atomic<core::i32> counter{0};
};

// /////////////////////////////////////////////////////////////////////////////
/// @class JobSystem
/// @brief Lock-free (per-deque) work-stealing scheduler.
///
/// Each worker thread owns a local deque. Jobs are pushed to the issuing
/// thread's deque and stolen from the back by idle workers.
///
/// @par Usage
/// @code
///   JobSystem js{4};
///   JobHandle handle{};
///   js.kickJob([](){ /* work */ }, handle);
///   js.kickJob([](){ /* work */ }, handle);
///   js.waitForCounter(handle, 0);
/// @endcode
// /////////////////////////////////////////////////////////////////////////////
class JobSystem final : public core::NonCopyable<JobSystem>
{
public:
    /// @brief Creates a job system with the given number of worker threads.
    /// @param workerCount Number of workers (0 = hardware_concurrency).
    explicit JobSystem(core::u32 workerCount = 0);

    /// @brief Shuts down all workers after draining pending jobs.
    ~JobSystem();

    // --------------------------------------------------------------------- //
    //  Job submission                                                        //
    // --------------------------------------------------------------------- //

    /// @brief Submits a job and increments the handle counter.
    /// @param job Callable to execute.
    /// @param handle Associated handle whose counter is decremented on
    ///        completion.
    void kickJob(std::function<void()> job, JobHandle& handle);

    /// @brief Spins (assisting with work) until @p handle.counter reaches
    ///        @p targetValue.
    /// @param handle Handle to wait on.
    /// @param targetValue Target counter value (typically 0).
    void waitForCounter(const JobHandle& handle, core::i32 targetValue) const;

    /// @brief Returns the worker thread count.
    [[nodiscard]] core::u32 workerCount() const noexcept;

private:
    struct WorkerData
    {
        std::deque<std::pair<std::function<void()>, JobHandle*>> localQueue;
        std::mutex                                                mutex;
    };

    void workerLoop(core::u32 workerIndex);

    bool trySteal(core::u32 thiefIndex,
                  std::pair<std::function<void()>, JobHandle*>& outJob);

    std::vector<std::thread>                     workers_;
    std::vector<std::unique_ptr<WorkerData>>     workerData_;
    std::atomic<bool>                            stopping_{false};
};

} // namespace lpl::concurrency
