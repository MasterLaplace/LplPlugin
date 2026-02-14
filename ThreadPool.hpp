#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <future>

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

    template <typename Func, typename... Args>
    auto enqueue(Func&& func, Args&&... args)
        -> std::future<typename std::invoke_result<Func, Args...>::type>
    {
        using ReturnType = typename std::invoke_result<Func, Args...>::type;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
        );

        std::future<ReturnType> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(_mutex);
            if (!_active)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            _tasks.emplace([task]() { (*task)(); });
        }
        _condition.notify_one();
        return res;
    }

    template <typename Func, typename... Args>
    void enqueueDetached(Func&& func, Args&&... args)
    {
        auto task = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
        {
            std::unique_lock<std::mutex> lock(_mutex);
            if (!_active)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            _tasks.emplace(std::move(task));
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
