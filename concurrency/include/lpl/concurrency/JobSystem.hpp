/**
 * @file JobSystem.hpp
 * @brief Work-stealing job system with per-thread deques and counters.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_CONCURRENCY_JOBSYSTEM_HPP
#    define LPL_CONCURRENCY_JOBSYSTEM_HPP

#    include <lpl/concurrency/ChaseLevDeque.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>

#    include <atomic>
#    include <functional>
#    include <memory>
#    include <thread>
#    include <vector>

namespace lpl::concurrency {

/**
 * @struct JobHandle
 * @brief Opaque handle used to wait on a group of jobs via an atomic counter.
 */
struct JobHandle {
    std::atomic<core::i32> counter{0};
};

/**
 * @class JobSystem
 * @brief Lock-free (per-deque) work-stealing scheduler.
 *
 * Each worker thread owns a local deque. Jobs are submitted to a shared
 * queue, which the workers drain into their deques, and an idle worker
 * steals from the top of the other workers' deques.
 *
 * @par Usage
 * @code{.cpp}
 *   JobSystem js{4};
 *   JobHandle handle{};
 *   js.kickJob([](){ ... }, handle);
 *   js.kickJob([](){ ... }, handle);
 *   js.waitForCounter(handle, 0);
 * @endcode
 */
class JobSystem final : public core::NonCopyable<JobSystem> {
public:
    /**
     * @brief Creates a job system with the given number of worker threads.
     * @param workerCount Number of workers (0 = hardware_concurrency).
     */
    explicit JobSystem(core::u32 workerCount = 0);

    /**
     * @brief Stops the workers and joins them, without draining the jobs still pending.
     */
    ~JobSystem();

    /**
     * @brief Submits a job and increments the handle counter.
     *
     * The job goes to the shared submission queue, under its mutex, so any thread may submit;
     * the workers drain that queue into their local deques.
     *
     * @param job Callable to execute.
     * @param handle Associated handle whose counter is decremented on
     *        completion.
     */
    void kickJob(std::function<void()> job, JobHandle &handle);

    /**
     * @brief Spins until @p handle.counter reaches @p targetValue.
     * @param handle Handle to wait on.
     * @param targetValue Target counter value (typically 0).
     */
    void waitForCounter(const JobHandle &handle, core::i32 targetValue) const;

    /**
     * @brief Returns the worker thread count.
     */
    [[nodiscard]] core::u32 workerCount() const noexcept;

private:
    struct Job {
        std::function<void()> func;
        JobHandle *handle;
    };

    struct WorkerData {
        ChaseLevDeque<Job *> localQueue;
    };

    /**
     * @brief Runs jobs until the system stops.
     *
     * Each turn moves one job, if any, from the shared submission queue into the worker's local
     * deque, then runs the job its deque pops or, when the deque is empty, one stolen from another
     * worker.
     *
     * @param workerIndex The worker this thread is.
     */
    void workerLoop(core::u32 workerIndex);

    /**
     * @brief Steals a job from the other workers' deques, trying each in turn from the one after
     *        @p thiefIndex.
     *
     * @param thiefIndex The worker that steals.
     * @param[out] outJob Receives the stolen job.
     * @return True when a job was stolen.
     */
    bool trySteal(core::u32 thiefIndex, Job *&outJob);

    std::vector<std::thread> _workers;
    std::vector<std::unique_ptr<WorkerData>> _workerData;
    std::atomic<bool> _stopping{false};

    std::mutex _submissionMutex;         /**< Guards _submissionQueue. */
    std::vector<Job *> _submissionQueue; /**< Jobs submitted from any thread, waiting for a worker. */
};

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_JOBSYSTEM_HPP
