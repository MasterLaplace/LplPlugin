/**
 * @file WorldPartition.cpp
 * @brief ThreadPool implementation for WorldPartition
 *
 * @author @MasterLaplace
 * @version 5.0
 * @date 2025-11-19
 */

#include "WorldPartition.hpp"

namespace Optimizing::World {

ThreadPool::ThreadPool(size_t threads) : _stop(false)
{
    _workers.reserve(threads);
    for (size_t i = 0; i < threads; ++i)
    {
        _workers.emplace_back([this] {
            while (true)
            {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(_queueMutex);
                    _condition.wait(lock, [this] { return _stop || !_tasks.empty(); });

                    if (_stop && _tasks.empty())
                        return;

                    task = std::move(_tasks.front());
                    _tasks.pop();
                    _active.fetch_add(1, std::memory_order_relaxed);
                }
                task();
                _active.fetch_sub(1, std::memory_order_relaxed);
                _condition.notify_all();
            }
        });
    }
}

ThreadPool::~ThreadPool() { shutdown(); }

void ThreadPool::shutdown()
{
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (_stop)
            return;  // Guard against double shutdown
        _stop = true;
    }
    _condition.notify_all();

    for (std::thread &worker : _workers)
    {
        if (worker.joinable())
            worker.join();
    }
}

void ThreadPool::waitIdle()
{
    std::unique_lock<std::mutex> lock(_queueMutex);
    _condition.wait(lock, [this] { return _tasks.empty() && _active.load(std::memory_order_relaxed) == 0; });
}

} // namespace Optimizing::World
