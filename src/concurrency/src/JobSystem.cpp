// /////////////////////////////////////////////////////////////////////////////
/// @file JobSystem.cpp
/// @brief Work-stealing job system implementation.
// /////////////////////////////////////////////////////////////////////////////

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

    workerData_.reserve(count);
    for (core::u32 i = 0; i < count; ++i)
    {
        workerData_.push_back(std::make_unique<WorkerData>());
    }
    workers_.reserve(count);

    for (core::u32 i = 0; i < count; ++i)
    {
        workers_.emplace_back(&JobSystem::workerLoop, this, i);
    }
}

JobSystem::~JobSystem()
{
    stopping_.store(true, std::memory_order_release);

    for (auto& w : workers_)
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
    const core::u32 target = rng() % static_cast<core::u32>(workerData_.size());

    {
        std::lock_guard<std::mutex> lock{workerData_[target]->mutex};
        workerData_[target]->localQueue.emplace_back(std::move(job), &handle);
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
    return static_cast<core::u32>(workers_.size());
}

// -------------------------------------------------------------------------- //
//  Private                                                                   //
// -------------------------------------------------------------------------- //

void JobSystem::workerLoop(core::u32 workerIndex)
{
    auto& data = *workerData_[workerIndex];

    while (!stopping_.load(std::memory_order_acquire))
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
    const auto workerCount = static_cast<core::u32>(workerData_.size());

    for (core::u32 offset = 1; offset < workerCount; ++offset)
    {
        const core::u32 victimIndex = (thiefIndex + offset) % workerCount;
        auto& victim = *workerData_[victimIndex];

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
