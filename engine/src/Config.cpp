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

Config::Builder &Config::Builder::tickRate(core::u32 hz) noexcept
{
    _tickRate = hz;
    return *this;
}

Config::Builder &Config::Builder::maxEntities(core::u32 n) noexcept
{
    _maxEntities = n;
    return *this;
}

Config::Builder &Config::Builder::maxChunks(core::u32 n) noexcept
{
    _maxChunks = n;
    return *this;
}

Config::Builder &Config::Builder::worldCellCapacity(core::u32 n) noexcept
{
    _worldCellCapacity = n;
    return *this;
}

Config::Builder &Config::Builder::serverMode(bool enabled) noexcept
{
    _serverMode = enabled;
    return *this;
}

Config::Builder &Config::Builder::headless(bool enabled) noexcept
{
    _headless = enabled;
    return *this;
}

Config::Builder &Config::Builder::arenaSize(core::usize bytes) noexcept
{
    _arenaSize = bytes;
    return *this;
}

Config::Builder &Config::Builder::worldArenaSize(core::usize bytes) noexcept
{
    _worldArenaSize = bytes;
    return *this;
}

Config::Builder &Config::Builder::enableBci(bool enabled) noexcept
{
    _enableBci = enabled;
    return *this;
}

Config::Builder &Config::Builder::enablePhysics(bool enabled) noexcept
{
    _enablePhysics = enabled;
    return *this;
}

Config::Builder &Config::Builder::enableNetworking(bool enabled) noexcept
{
    _enableNetworking = enabled;
    return *this;
}

Config::Builder &Config::Builder::enableRendering(bool enabled) noexcept
{
    _enableRendering = enabled;
    return *this;
}

Config::Builder &Config::Builder::enableRealTimeGuard(bool enabled) noexcept
{
    _enableRealTimeGuard = enabled;
    return *this;
}

Config::Builder &Config::Builder::enableGpu(bool enabled) noexcept
{
    _enableGpu = enabled;
    return *this;
}

Config::Builder &Config::Builder::serverAddress(pmr::string addr) noexcept
{
    _serverAddress = std::move(addr);
    return *this;
}

Config::Builder &Config::Builder::serverWorkerThreads(core::u32 n) noexcept
{
    _serverWorkerThreads = n;
    return *this;
}

Config::Builder &Config::Builder::replaySnapshotInterval(core::u32 ticks) noexcept
{
    _replaySnapshotInterval = ticks;
    return *this;
}

Config::Builder &Config::Builder::maxPacketsPerTick(core::u32 n) noexcept
{
    _maxPacketsPerTick = n;
    return *this;
}

Config::Builder &Config::Builder::serverPort(core::u16 port) noexcept
{
    _serverPort = port;
    return *this;
}

Config Config::Builder::build() const noexcept
{
    Config cfg;
    cfg._tickRate = _tickRate;
    cfg._maxEntities = _maxEntities;
    cfg._maxChunks = _maxChunks;
    cfg._worldCellCapacity = _worldCellCapacity;
    cfg._serverWorkerThreads = _serverWorkerThreads;
    cfg._replaySnapshotInterval = _replaySnapshotInterval;
    cfg._maxPacketsPerTick = _maxPacketsPerTick;
    cfg._serverMode = _serverMode;
    cfg._headless = _headless;
    cfg._arenaSize = _arenaSize;
    cfg._worldArenaSize = _worldArenaSize;
    cfg._enableBci = _enableBci;
    cfg._enablePhysics = _enablePhysics;
    cfg._enableNetworking = _enableNetworking;
    cfg._enableRendering = _enableRendering;
    cfg._enableRealTimeGuard = _enableRealTimeGuard;
    cfg._enableGpu = _enableGpu;
    cfg._serverAddress = _serverAddress;
    cfg._serverPort = _serverPort;
    return cfg;
}

} // namespace lpl::engine
