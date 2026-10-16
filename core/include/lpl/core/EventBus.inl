#ifndef LPL_CORE_EVENTBUS_INL
#define LPL_CORE_EVENTBUS_INL

namespace lpl::core {

namespace detail {
inline u32 getNextEventId() noexcept
{
    static std::atomic<u32> id{0};
    return id.fetch_add(1, std::memory_order_relaxed);
}
} // namespace detail

template <typename T> inline u32 getEventId() noexcept
{
    static const u32 id = detail::getNextEventId();
    return id;
}

inline EventBus &EventBus::instance()
{
    static EventBus instance;
    return instance;
}

template <typename T> inline u32 EventBus::subscribe(lpl::pmr::function<void(const T &)> callback)
{
    lpl::pmr::lock_guard lock{_mutex};
    const u32 id = ++_nextId;
    auto &handlers = _handlers[getEventId<T>()];

    handlers.push_back({id, [cb = std::move(callback)](const void *ev) { cb(*static_cast<const T *>(ev)); }});

    return id;
}

template <typename T> inline void EventBus::unsubscribe(u32 id)
{
    lpl::pmr::lock_guard lock{_mutex};
    const auto it = _handlers.find(getEventId<T>());
    if (it == _handlers.end())
        return;

    auto &handlers = it->second;
    for (auto handlerIt = handlers.begin(); handlerIt != handlers.end();)
    {
        if (handlerIt->id == id)
        {
            handlerIt = handlers.erase(handlerIt);
        }
        else
        {
            ++handlerIt;
        }
    }
}

template <typename T> inline void EventBus::publish(const T &event)
{
    lpl::pmr::vector<lpl::pmr::function<void(const void *)>> callbacksToInvoke;

    {
        lpl::pmr::lock_guard lock{_mutex};
        const auto it = _handlers.find(getEventId<T>());
        if (it != _handlers.end())
        {
            for (const auto &handler : it->second)
            {
                callbacksToInvoke.push_back(handler.callback);
            }
        }
    }

    for (const auto &cb : callbacksToInvoke)
    {
        cb(&event);
    }
}

inline void EventBus::clear()
{
    lpl::pmr::lock_guard lock{_mutex};
    _handlers.clear();
}

} // namespace lpl::core

#endif // LPL_CORE_EVENTBUS_INL
