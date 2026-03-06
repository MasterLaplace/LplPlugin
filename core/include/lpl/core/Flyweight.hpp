/**
 * @file Flyweight.hpp
 * @brief Flyweight pattern implementation for shared caching.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_CORE_FLYWEIGHT_HPP
#    define LPL_CORE_FLYWEIGHT_HPP

#    include <memory>
#    include <mutex>
#    include <unordered_map>

namespace lpl::core {

/**
 * @class FlyweightCache
 * @brief Thread-safe cache holding shared immutable objects.
 *
 * Ensures only one instance of an object with a given Key exists. Uses
 * weak pointers to allow instances to be destroyed when no longer used
 * elsewhere in the application, automatically cleaning up the cache.
 */
template <typename Key, typename T> class FlyweightCache {
public:
    FlyweightCache() = default;
    ~FlyweightCache() = default;

    /**
     * @brief Gets an existing shared instance, or creates a new one.
     * @param key Identifies the unique object state.
     * @param creator Callable that returns a `std::shared_ptr<const T>`
     *                if a new instance needs to be created.
     * @return Shared pointer to the cached or newly created instance.
     */
    template <typename CreatorCallable> std::shared_ptr<const T> getOrCreate(const Key &key, CreatorCallable &&creator)
    {
        std::lock_guard lock{_mutex};

        auto it = _cache.find(key);
        if (it != _cache.end())
        {
            if (auto shared = it->second.lock())
            {
                return shared;
            }
            // Pointer expired, remove stale entry.
            _cache.erase(it);
        }

        // Create new
        std::shared_ptr<const T> newInstance = creator();
        if (newInstance)
        {
            _cache.emplace(key, newInstance);
        }
        return newInstance;
    }

    /** @brief Triggers a cleanup of any expired weak pointers. */
    void prune()
    {
        std::lock_guard lock{_mutex};
        for (auto it = _cache.begin(); it != _cache.end();)
        {
            if (it->second.expired())
            {
                it = _cache.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    /** @brief Current number of cached weak keys. */
    [[nodiscard]] usize size() const
    {
        std::lock_guard lock{_mutex};
        return _cache.size();
    }

private:
    mutable std::mutex _mutex;
    std::unordered_map<Key, std::weak_ptr<const T>> _cache;
};

} // namespace lpl::core

#endif // LPL_CORE_FLYWEIGHT_HPP
