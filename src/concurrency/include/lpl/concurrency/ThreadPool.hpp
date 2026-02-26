// /////////////////////////////////////////////////////////////////////////////
/// @file ThreadPool.hpp
/// @brief Fixed-size thread pool with std::future-based task submission.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>
#include <lpl/core/Expected.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace lpl::concurrency {

// /////////////////////////////////////////////////////////////////////////////
/// @class ThreadPool
/// @brief Simple fixed-thread-count pool.
///
/// Workers pull tasks from a shared FIFO queue protected by a mutex +
/// condition variable.  Use @ref enqueue for tasks that return a value and
/// @ref enqueueDetached for fire-and-forget work.
///
/// Call @ref shutdown to drain all queued tasks; the destructor calls
/// @c shutdown implicitly.
// /////////////////////////////////////////////////////////////////////////////
class ThreadPool final : public core::NonCopyable<ThreadPool>
{
public:
    /// @brief Creates the pool with @p threadCount worker threads.
    /// @param threadCount Number of worker threads.  Zero means
    ///        @c std::thread::hardware_concurrency().
    explicit ThreadPool(core::u32 threadCount = 0);

    /// @brief Drains pending tasks and joins all workers.
    ~ThreadPool();

    // --------------------------------------------------------------------- //
    //  Task submission                                                       //
    // --------------------------------------------------------------------- //

    /// @brief Enqueues a callable and returns its future.
    /// @tparam F Callable type.
    /// @tparam Args Argument types.
    /// @param func Callable to execute.
    /// @param args Arguments forwarded to @p func.
    /// @return @c std::future holding the return value.
    template <typename F, typename... Args>
    [[nodiscard]] auto enqueue(F&& func, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>;

    /// @brief Enqueues a fire-and-forget callable.
    /// @tparam F Callable type.
    /// @param func Callable to execute.
    template <typename F>
    void enqueueDetached(F&& func);

    // --------------------------------------------------------------------- //
    //  Lifecycle                                                             //
    // --------------------------------------------------------------------- //

    /// @brief Signals workers to finish and blocks until all pending tasks
    ///        are processed.
    void shutdown();

    /// @brief Returns the number of worker threads.
    [[nodiscard]] core::u32 threadCount() const noexcept;

private:
    /// @brief Worker loop — waits on the CV and processes tasks.
    void workerLoop();

    std::vector<std::thread>            workers_;
    std::deque<std::function<void()>>   tasks_;
    std::mutex                          mutex_;
    std::condition_variable             cv_;
    std::atomic<bool>                   stopping_{false};
};

// /////////////////////////////////////////////////////////////////////////////
//  Template implementations                                                  //
// /////////////////////////////////////////////////////////////////////////////

template <typename F, typename... Args>
auto ThreadPool::enqueue(F&& func, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>>
{
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(func), std::forward<Args>(args)...)
    );

    std::future<ReturnType> future = task->get_future();

    {
        std::lock_guard<std::mutex> lock{mutex_};
        tasks_.emplace_back([task]() { (*task)(); });
    }
    cv_.notify_one();

    return future;
}

template <typename F>
void ThreadPool::enqueueDetached(F&& func)
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        tasks_.emplace_back(std::forward<F>(func));
    }
    cv_.notify_one();
}

} // namespace lpl::concurrency
