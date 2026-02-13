#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(uint32_t numThreads = std::thread::hardware_concurrency())
    {
        _active = true;

        if (numThreads == 0u)
            numThreads = 1u;

        _workers.reserve(numThreads);
        for (uint32_t i = 0u; i < numThreads; ++i)
        {
            _workers.emplace_back([this] {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    _condition.wait(lock, [this] {return !_active || !_tasks.empty(); });
                    if (!_active && _tasks.empty())
                        return;

                    task = std::move(_tasks.front());
                    _tasks.pop();
                }
                task();
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _active = false;
        }

        _condition.notify_all();

        for (std::thread &worker : _workers)
        {
            if (worker.joinable())
                worker.join();
        }
    }

    template <typename Func>
    void enqueue(Func &&func)
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            if (!active)
                return;
            _tasks.emplace(std::forward<Func>(func));
        }
        _condition.notify_one();
    }

private:
    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    std::condition_variable _condition;
    bool _active = false;
};
