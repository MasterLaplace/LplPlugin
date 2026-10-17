#ifndef LPL_CONCURRENCY_THREADPOOL_INL
#define LPL_CONCURRENCY_THREADPOOL_INL

namespace lpl::concurrency {

template <typename F, typename... Args>
auto ThreadPool::enqueue(F &&func, Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>
{
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(func), std::forward<Args>(args)...));

    std::future<ReturnType> future = task->get_future();

    {
        std::lock_guard<std::mutex> lock{_mutex};
        _tasks.emplace_back([task]() { (*task)(); });
    }
    _cv.notify_one();

    return future;
}

template <typename F> void ThreadPool::enqueueDetached(F &&func)
{
    {
        std::lock_guard<std::mutex> lock{_mutex};
        _tasks.emplace_back(std::forward<F>(func));
    }
    _cv.notify_one();
}

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_THREADPOOL_INL
