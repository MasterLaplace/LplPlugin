/**
 * @file JobSystem.cpp
 * @brief Work-stealing job system implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/concurrency/JobSystem.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Platform.hpp>

#include <mutex>

namespace lpl::concurrency {

// -------------------------------------------------------------------------- //
//  Construction / Destruction                                                //
// -------------------------------------------------------------------------- //

JobSystem::JobSystem(core::u32 workerCount)
{
    const core::u32 count =
        (workerCount == 0) ? static_cast<core::u32>(std::thread::hardware_concurrency()) : workerCount;

    LPL_ASSERT(count > 0);

    _workerData.reserve(count);
    for (core::u32 i = 0; i < count; ++i)
    {
        _workerData.push_back(std::make_unique<WorkerData>());
    }
    _workers.reserve(count);

    for (core::u32 i = 0; i < count; ++i)
    {
        _workers.emplace_back(&JobSystem::workerLoop, this, i);
    }
}

JobSystem::~JobSystem()
{
    _stopping.store(true, std::memory_order_release);

    for (auto &w : _workers)
    {
        if (w.joinable())
        {
            w.join();
        }
    }
}

// -------------------------------------------------------------------------- //
//  Public API                                                                //
// -------------------------------------------------------------------------- //

void JobSystem::kickJob(std::function<void()> job, JobHandle &handle)
{
    handle.counter.fetch_add(1, std::memory_order_acq_rel);

    Job *newJob = new Job{std::move(job), &handle};

    // Push to shared submission queue (thread-safe from any thread).
    // Workers drain this queue into their local deques.
    {
        std::lock_guard lock{_submissionMutex};
        _submissionQueue.push_back(newJob);
    }
}

void JobSystem::waitForCounter(const JobHandle &handle, core::i32 targetValue) const
{
    while (handle.counter.load(std::memory_order_acquire) != targetValue)
    {
        LPL_CPU_PAUSE();
    }
}

core::u32 JobSystem::workerCount() const noexcept { return static_cast<core::u32>(_workers.size()); }

// -------------------------------------------------------------------------- //
//  Private                                                                   //
// -------------------------------------------------------------------------- //

void JobSystem::workerLoop(core::u32 workerIndex)
{
    auto &data = *_workerData[workerIndex];

    while (!_stopping.load(std::memory_order_acquire))
    {
        // Drain one job from the shared submission queue into our local deque
        {
            std::lock_guard lock{_submissionMutex};
            if (!_submissionQueue.empty())
            {
                data.localQueue.push(_submissionQueue.back());
                _submissionQueue.pop_back();
            }
        }

        Job *job = data.localQueue.pop();

        if (!job)
        {
            if (!trySteal(workerIndex, job))
            {
                LPL_CPU_PAUSE();
                continue;
            }
        }

        if (job)
        {
            job->func();
            job->handle->counter.fetch_sub(1, std::memory_order_acq_rel);
            delete job;
        }
    }
}

bool JobSystem::trySteal(core::u32 thiefIndex, JobSystem::Job *&outJob)
{
    const auto workerCount = static_cast<core::u32>(_workerData.size());

    // Basic work-stealing loop: examine other workers' deques
    for (core::u32 offset = 1; offset < workerCount; ++offset)
    {
        const core::u32 victimIndex = (thiefIndex + offset) % workerCount;
        auto &victim = *_workerData[victimIndex];

        outJob = victim.localQueue.steal();
        if (outJob)
        {
            return true;
        }
    }

    return false;
}

} // namespace lpl::concurrency
