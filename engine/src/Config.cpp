/**
 * @file Config.cpp
 * @brief Config::Builder implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/engine/Config.hpp>

namespace lpl::engine {

Config::Builder& Config::Builder::tickRate(core::u32 hz) noexcept
{
    _tickRate = hz;
    return *this;
}

Config::Builder& Config::Builder::maxEntities(core::u32 n) noexcept
{
    _maxEntities = n;
    return *this;
}

Config::Builder& Config::Builder::maxChunks(core::u32 n) noexcept
{
    _maxChunks = n;
    return *this;
}

Config::Builder& Config::Builder::serverMode(bool enabled) noexcept
{
    _serverMode = enabled;
    return *this;
}

Config::Builder& Config::Builder::headless(bool enabled) noexcept
{
    _headless = enabled;
    return *this;
}

Config::Builder& Config::Builder::arenaSize(core::usize bytes) noexcept
{
    _arenaSize = bytes;
    return *this;
}

Config::Builder& Config::Builder::enableBci(bool enabled) noexcept
{
    _enableBci = enabled;
    return *this;
}

Config::Builder& Config::Builder::enableGpu(bool enabled) noexcept
{
    _enableGpu = enabled;
    return *this;
}

Config Config::Builder::build() const noexcept
{
    Config cfg;
    cfg._tickRate    = _tickRate;
    cfg._maxEntities = _maxEntities;
    cfg._maxChunks   = _maxChunks;
    cfg._serverMode  = _serverMode;
    cfg._headless    = _headless;
    cfg._arenaSize   = _arenaSize;
    cfg._enableBci   = _enableBci;
    cfg._enableGpu   = _enableGpu;
    return cfg;
}

} // namespace lpl::engine
