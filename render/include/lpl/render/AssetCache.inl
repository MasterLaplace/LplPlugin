#ifndef LPL_RENDER_ASSETCACHE_INL
#define LPL_RENDER_ASSETCACHE_INL

namespace lpl::render {

inline AssetCache &AssetCache::instance()
{
    static AssetCache s_instance;
    return s_instance;
}

} // namespace lpl::render

#endif // LPL_RENDER_ASSETCACHE_INL
