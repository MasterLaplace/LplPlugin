#ifndef LPL_ENGINE_RESOURCEMANAGER_INL
#define LPL_ENGINE_RESOURCEMANAGER_INL

namespace lpl::engine {

template <typename T, typename... Args> inline T &ResourceManager::getOrCreate(core::u64 key, Args &&...args)
{
    static_assert(std::is_base_of_v<IResource, T>, "T must derive from IResource");
    concurrency::SpinLockGuard guard{_lock};
    if (auto it = _resources.find(key); it != _resources.end())
        return static_cast<T &>(*it->second);
    auto owned = lpl::pmr::make_unique<T>(std::forward<Args>(args)...);
    T &ref = *owned;
    _resources.emplace(key, std::move(owned));
    return ref;
}

template <typename T> inline T *ResourceManager::find(core::u64 key)
{
    static_assert(std::is_base_of_v<IResource, T>, "T must derive from IResource");
    concurrency::SpinLockGuard guard{_lock};
    auto it = _resources.find(key);
    return it == _resources.end() ? nullptr : static_cast<T *>(it->second.get());
}

inline bool ResourceManager::contains(core::u64 key)
{
    concurrency::SpinLockGuard guard{_lock};
    return _resources.find(key) != _resources.end();
}

inline core::usize ResourceManager::size() const
{
    concurrency::SpinLockGuard guard{_lock};
    return _resources.size();
}

inline void ResourceManager::clear()
{
    concurrency::SpinLockGuard guard{_lock};
    _resources.clear();
}

} // namespace lpl::engine

#endif // LPL_ENGINE_RESOURCEMANAGER_INL
