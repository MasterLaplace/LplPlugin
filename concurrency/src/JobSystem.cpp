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

#include <random>

namespace lpl::concurrency {

// -------------------------------------------------------------------------- //
//  Construction / Destruction                                                //
// -------------------------------------------------------------------------- //

JobSystem::JobSystem(core::u32 workerCount)
{
    const core::u32 count = (workerCount == 0)
        ? static_cast<core::u32>(std::thread::hardware_concurrency())
        : workerCount;

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

    for (auto& w : _workers)
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

void JobSystem::kickJob(std::function<void()> job, JobHandle& handle)
{
    handle.counter.fetch_add(1, std::memory_order_acq_rel);

    thread_local static std::mt19937 rng{std::random_device{}()};
    const core::u32 target = rng() % static_cast<core::u32>(_workerData.size());

    {
        std::lock_guard<std::mutex> lock{_workerData[target]->mutex};
        _workerData[target]->localQueue.emplace_back(std::move(job), &handle);
    }
}

void JobSystem::waitForCounter(const JobHandle& handle, core::i32 targetValue) const
{
    while (handle.counter.load(std::memory_order_acquire) != targetValue)
    {
        LPL_CPU_PAUSE();
    }
}

core::u32 JobSystem::workerCount() const noexcept
{
    return static_cast<core::u32>(_workers.size());
}

// -------------------------------------------------------------------------- //
//  Private                                                                   //
// -------------------------------------------------------------------------- //

void JobSystem::workerLoop(core::u32 workerIndex)
{
    auto& data = *_workerData[workerIndex];

    while (!_stopping.load(std::memory_order_acquire))
    {
        std::pair<std::function<void()>, JobHandle*> job{nullptr, nullptr};

        {
            std::lock_guard<std::mutex> lock{data.mutex};
            if (!data.localQueue.empty())
            {
                job = std::move(data.localQueue.front());
                data.localQueue.pop_front();
            }
        }

        if (!job.first)
        {
            if (!trySteal(workerIndex, job))
            {
                LPL_CPU_PAUSE();
                continue;
            }
        }

        job.first();
        job.second->counter.fetch_sub(1, std::memory_order_acq_rel);
    }
}

bool JobSystem::trySteal(core::u32 thiefIndex,
                         std::pair<std::function<void()>, JobHandle*>& outJob)
{
    const auto workerCount = static_cast<core::u32>(_workerData.size());

    for (core::u32 offset = 1; offset < workerCount; ++offset)
    {
        const core::u32 victimIndex = (thiefIndex + offset) % workerCount;
        auto& victim = *_workerData[victimIndex];

        std::lock_guard<std::mutex> lock{victim.mutex};
        if (!victim.localQueue.empty())
        {
            outJob = std::move(victim.localQueue.back());
            victim.localQueue.pop_back();
            return true;
        }
    }

    return false;
}

} // namespace lpl::concurrency
