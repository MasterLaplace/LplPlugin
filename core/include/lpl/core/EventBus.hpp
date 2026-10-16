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

#    include <lpl/std/functional.hpp>
#    include <lpl/std/mutex.hpp>
#    include <lpl/std/unordered_map.hpp>
#    include <lpl/std/vector.hpp>

#    include <atomic>
#    include <memory>

namespace lpl::core {

namespace detail {
inline u32 getNextEventId() noexcept;
} // namespace detail

template <typename T> inline u32 getEventId() noexcept;

/**
 * @class EventBus
 * @brief Thread-safe Pub/Sub event bus implementing the Observer pattern.
 */
class EventBus final : public NonCopyable<EventBus> {
public:
    /**
     * @brief Global singleton instance access.
     */
    static EventBus &instance();

    /**
     * @brief Subscribe to an event type.
     * @tparam T Event type.
     * @param callback Function to invoke when event occurs.
     * @return Subscription ID, used for unsubscription.
     */
    template <typename T> u32 subscribe(lpl::pmr::function<void(const T &)> callback);

    /**
     * @brief Unsubscribe from an event type using subscription ID.
     * @tparam T Event type.
     * @param id Subscription ID.
     */
    template <typename T> void unsubscribe(u32 id);

    /**
     * @brief Publish an event to all subscribers.
     *
     * The callbacks are copied under the lock and invoked outside it, to prevent deadlocks if a
     * callback publishes.
     *
     * @tparam T Event type.
     * @param event The event instance.
     */
    template <typename T> void publish(const T &event);

    /**
     * @brief Clears all subscriptions.
     */
    void clear();

private:
    EventBus() = default;
    ~EventBus() = default;

    struct Handler {
        u32 id;
        lpl::pmr::function<void(const void *)> callback;
    };

    lpl::pmr::mutex _mutex;
    u32 _nextId{0};
    lpl::pmr::unordered_map<u32, lpl::pmr::vector<Handler>> _handlers;
};

} // namespace lpl::core

#    include "EventBus.inl"

#endif // LPL_CORE_EVENTBUS_HPP
