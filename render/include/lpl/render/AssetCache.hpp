/**
 * @file AssetCache.hpp
 * @brief Centralized GPU asset cache using the Flyweight pattern.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_ASSETCACHE_HPP
#    define LPL_RENDER_ASSETCACHE_HPP

#    include <lpl/core/Flyweight.hpp>
#    include <lpl/render/Material.hpp>
#    include <lpl/render/Mesh.hpp>
#    include <lpl/render/Shader.hpp>

#    include <string>

namespace lpl::render {

/**
 * @class AssetCache
 * @brief Singleton cache for sharing immutable render assets.
 *
 * Uses core::FlyweightCache to provide thread-safe pooling of
 * heavy objects like Meshes and Shaders to avoid duplication.
 */
class AssetCache {
public:
    static AssetCache &instance()
    {
        static AssetCache s_instance;
        return s_instance;
    }

    core::FlyweightCache<std::string, Mesh> meshes;
    core::FlyweightCache<std::string, Shader> shaders;
    core::FlyweightCache<std::string, Material> materials;

private:
    AssetCache() = default;
    ~AssetCache() = default;

    AssetCache(const AssetCache &) = delete;
    AssetCache &operator=(const AssetCache &) = delete;
};

} // namespace lpl::render

#endif // LPL_RENDER_ASSETCACHE_HPP
