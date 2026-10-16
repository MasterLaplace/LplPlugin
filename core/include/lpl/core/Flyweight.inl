#ifndef LPL_CORE_FLYWEIGHT_INL
#define LPL_CORE_FLYWEIGHT_INL

namespace lpl::core {

template <typename Key, typename T>
template <typename CreatorCallable>
inline std::shared_ptr<const T> FlyweightCache<Key, T>::getOrCreate(const Key &key, CreatorCallable &&creator)
{
    lpl::pmr::lock_guard lock{_mutex};

    auto it = _cache.find(key);
    if (it != _cache.end())
    {
        if (auto shared = it->second.lock())
        {
            return shared;
        }
        _cache.erase(it);
    }

    std::shared_ptr<const T> newInstance = creator();
    if (newInstance)
    {
        _cache.emplace(key, newInstance);
    }
    return newInstance;
}

template <typename Key, typename T> inline void FlyweightCache<Key, T>::prune()
{
    lpl::pmr::lock_guard lock{_mutex};
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

template <typename Key, typename T> inline usize FlyweightCache<Key, T>::size() const
{
    lpl::pmr::lock_guard lock{_mutex};
    return _cache.size();
}

} // namespace lpl::core

#endif // LPL_CORE_FLYWEIGHT_INL
