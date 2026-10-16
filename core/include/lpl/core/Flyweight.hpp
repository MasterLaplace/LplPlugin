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

#    include <lpl/core/Types.hpp>
#    include <lpl/std/mutex.hpp>
#    include <lpl/std/unordered_map.hpp>

#    include <memory>

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
     *
     * An entry whose instance has expired is removed before a new one is created.
     *
     * @param key Identifies the unique object state.
     * @param creator Callable that returns a `std::shared_ptr<const T>`
     *                if a new instance needs to be created.
     * @return Shared pointer to the cached or newly created instance.
     */
    template <typename CreatorCallable> std::shared_ptr<const T> getOrCreate(const Key &key, CreatorCallable &&creator);

    /**
     * @brief Triggers a cleanup of any expired weak pointers.
     */
    void prune();

    /**
     * @brief Current number of cached weak keys.
     */
    [[nodiscard]] usize size() const;

private:
    mutable lpl::pmr::mutex _mutex;
    lpl::pmr::unordered_map<Key, std::weak_ptr<const T>> _cache;
};

} // namespace lpl::core

#    include "Flyweight.inl"

#endif // LPL_CORE_FLYWEIGHT_HPP
