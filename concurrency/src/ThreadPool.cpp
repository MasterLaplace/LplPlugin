/**
 * @file ThreadPool.cpp
 * @brief Implementation of the fixed-size thread pool.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

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

    _workers.reserve(count);
    for (core::u32 i = 0; i < count; ++i)
    {
        _workers.emplace_back(&ThreadPool::workerLoop, this);
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
    if (_stopping.exchange(true, std::memory_order_acq_rel))
    {
        return;
    }

    _cv.notify_all();

    for (auto& worker : _workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

core::u32 ThreadPool::threadCount() const noexcept
{
    return static_cast<core::u32>(_workers.size());
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
            std::unique_lock<std::mutex> lock{_mutex};
            _cv.wait(lock, [this] {
                return _stopping.load(std::memory_order_relaxed) || !_tasks.empty();
            });

            if (_tasks.empty())
            {
                return;
            }

            task = std::move(_tasks.front());
            _tasks.pop_front();
        }

        task();
    }
}

} // namespace lpl::concurrency
