/**
 * @file EventBus.hpp
 * @brief Thread-safe Observer pattern for generic event pub/sub.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_CORE_EVENTBUS_HPP
#    define LPL_CORE_EVENTBUS_HPP

#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>

#    include <atomic>
#    include <functional>
#    include <memory>
#    include <mutex>
#    include <unordered_map>
#    include <vector>

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

/**
 * @class EventBus
 * @brief Thread-safe Pub/Sub event bus implementing the Observer pattern.
 */
class EventBus final : public NonCopyable<EventBus> {
public:
    /** @brief Global singleton instance access. */
    static EventBus &instance()
    {
        static EventBus instance;
        return instance;
    }

    /**
     * @brief Subscribe to an event type.
     * @tparam T Event type.
     * @param callback Function to invoke when event occurs.
     * @return Subscription ID, used for unsubscription.
     */
    template <typename T> u32 subscribe(std::function<void(const T &)> callback)
    {
        std::lock_guard lock{_mutex};
        const u32 id = ++_nextId;
        auto &handlers = _handlers[getEventId<T>()];

        handlers.push_back({id, [cb = std::move(callback)](const void *ev) { cb(*static_cast<const T *>(ev)); }});

        return id;
    }

    /**
     * @brief Unsubscribe from an event type using subscription ID.
     * @tparam T Event type.
     * @param id Subscription ID.
     */
    template <typename T> void unsubscribe(u32 id)
    {
        std::lock_guard lock{_mutex};
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

    /**
     * @brief Publish an event to all subscribers.
     * @tparam T Event type.
     * @param event The event instance.
     */
    template <typename T> void publish(const T &event)
    {
        std::vector<std::function<void(const void *)>> callbacksToInvoke;

        {
            std::lock_guard lock{_mutex};
            const auto it = _handlers.find(getEventId<T>());
            if (it != _handlers.end())
            {
                for (const auto &handler : it->second)
                {
                    callbacksToInvoke.push_back(handler.callback);
                }
            }
        }

        // Invoke outside the lock to prevent deadlocks if a callback publishes
        for (const auto &cb : callbacksToInvoke)
        {
            cb(&event);
        }
    }

    /** @brief Clears all subscriptions. */
    void clear()
    {
        std::lock_guard lock{_mutex};
        _handlers.clear();
    }

private:
    EventBus() = default;
    ~EventBus() = default;

    struct Handler {
        u32 id;
        std::function<void(const void *)> callback;
    };

    std::mutex _mutex;
    u32 _nextId{0};
    std::unordered_map<u32, std::vector<Handler>> _handlers;
};

} // namespace lpl::core

#endif // LPL_CORE_EVENTBUS_HPP
