/**
 * @file ResourceManager.hpp
 * @brief Shared, thread-safe asset cache (load-once).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-21
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_RESOURCEMANAGER_HPP
#    define LPL_ENGINE_RESOURCEMANAGER_HPP

#    include <lpl/concurrency/SpinLock.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/std/memory.hpp>
#    include <lpl/std/unordered_map.hpp>

#    include <type_traits>
#    include <utility>

namespace lpl::engine {

/**
 * @class IResource
 * @brief Base for anything the ResourceManager owns. A resource is any loaded
 *        asset (mesh, texture, audio clip, …); the only requirement is a virtual
 *        destructor so the manager can own heterogeneous types by base pointer.
 */
class IResource {
public:
    virtual ~IResource() = default;
};

/**
 * @class ResourceManager
 * @brief Owns loaded assets and hands them out by key, loading each at most once.
 *
 * This is the "load-once" cache Flakkari used, made shareable WITHOUT being a
 * singleton: one instance is owned at the top (the Engine for a single World,
 * the server for many) and passed to each World by reference. Several managers
 * can coexist, it is trivially testable, and there is no hidden global state.
 *
 * Thread-safe by construction: every access is guarded by a SpinLock, because
 * the server ticks many Worlds in parallel and they share one manager. The lock
 * is held across the get-or-create so a key is never loaded twice under a race.
 * Kernel-safe: SpinLock + lpl::pmr containers, no libm, no shared_ptr.
 *
 * Ownership model: the manager owns every resource for its whole lifetime and
 * hands out non-owning references/pointers. Assets outlive the frames that use
 * them, exactly as a load-once cache should; consumers never free them.
 */
class ResourceManager final : public core::NonCopyable<ResourceManager> {
public:
    ResourceManager() = default;

    /**
     * @brief Returns the resource at @p key, constructing it once if absent.
     * @tparam T Concrete resource type (must derive from IResource).
     * @param key Stable identifier (e.g. a hashed asset path).
     * @param args Forwarded to T's constructor on the first call for @p key.
     * @return Reference to the owned resource (stable for the manager's life).
     */
    template <typename T, typename... Args> [[nodiscard]] T &getOrCreate(core::u64 key, Args &&...args);

    /**
     * @brief Returns the resource at @p key, or nullptr if it was never loaded.
     * @tparam T Concrete resource type to cast to (caller's responsibility).
     */
    template <typename T> [[nodiscard]] T *find(core::u64 key);

    /**
     * @brief True if @p key currently holds a resource.
     */
    [[nodiscard]] bool contains(core::u64 key);

    /**
     * @brief Number of cached resources.
     */
    [[nodiscard]] core::usize size() const;

    /**
     * @brief Drops every cached resource.
     */
    void clear();

private:
    mutable concurrency::SpinLock _lock;
    lpl::pmr::unordered_map<core::u64, lpl::pmr::unique_ptr<IResource>> _resources;
};

} // namespace lpl::engine

#    include "ResourceManager.inl"

#endif // LPL_ENGINE_RESOURCEMANAGER_HPP
