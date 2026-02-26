// /////////////////////////////////////////////////////////////////////////////
/// @file ThreadPool.cpp
/// @brief Implementation of the fixed-size thread pool.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/concurrency/ThreadPool.hpp>
#include <lpl/core/Assert.hpp>

namespace lpl::concurrency {

// -------------------------------------------------------------------------- //
//  Construction / Destruction                                                //
// -------------------------------------------------------------------------- //

ThreadPool::ThreadPool(core::u32 threadCount)
{
    const core::u32 count = (threadCount == 0)
        ? static_cast<core::u32>(std::thread::hardware_concurrency())
        : threadCount;

    LPL_ASSERT(count > 0);

    workers_.reserve(count);
    for (core::u32 i = 0; i < count; ++i)
    {
        workers_.emplace_back(&ThreadPool::workerLoop, this);
    }
}

ThreadPool::~ThreadPool()
{
    shutdown();
}

// -------------------------------------------------------------------------- //
//  Lifecycle                                                                 //
// -------------------------------------------------------------------------- //

void ThreadPool::shutdown()
{
    if (stopping_.exchange(true, std::memory_order_acq_rel))
    {
        return;
    }

    cv_.notify_all();

    for (auto& worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

core::u32 ThreadPool::threadCount() const noexcept
{
    return static_cast<core::u32>(workers_.size());
}

// -------------------------------------------------------------------------- //
//  Private                                                                   //
// -------------------------------------------------------------------------- //

void ThreadPool::workerLoop()
{
    for (;;)
    {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(lock, [this] {
                return stopping_.load(std::memory_order_relaxed) || !tasks_.empty();
            });

            if (tasks_.empty())
            {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop_front();
        }

        task();
    }
}

} // namespace lpl::concurrency
