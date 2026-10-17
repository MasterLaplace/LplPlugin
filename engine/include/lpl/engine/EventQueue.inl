#ifndef LPL_ENGINE_EVENTQUEUE_INL
#define LPL_ENGINE_EVENTQUEUE_INL

namespace lpl::engine {

template <typename T> inline void TypedQueue<T>::push(T event)
{
    pmr::lock_guard<pmr::mutex> lock{mutex_};
    queue_.push_back(std::move(event));
}

template <typename T> inline pmr::vector<T> TypedQueue<T>::drain()
{
    pmr::lock_guard<pmr::mutex> lock{mutex_};
    pmr::vector<T> result;
    result.swap(queue_);
    return result;
}

template <typename T> inline bool TypedQueue<T>::empty() const
{
    pmr::lock_guard<pmr::mutex> lock{mutex_};
    return queue_.empty();
}

} // namespace lpl::engine

#endif // LPL_ENGINE_EVENTQUEUE_INL
