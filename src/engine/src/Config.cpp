// /////////////////////////////////////////////////////////////////////////////
/// @file Config.cpp
/// @brief Config::Builder implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/engine/Config.hpp>

namespace lpl::engine {

Config::Builder& Config::Builder::tickRate(core::u32 hz) noexcept
{
    tickRate_ = hz;
    return *this;
}

Config::Builder& Config::Builder::maxEntities(core::u32 n) noexcept
{
    maxEntities_ = n;
    return *this;
}

Config::Builder& Config::Builder::maxChunks(core::u32 n) noexcept
{
    maxChunks_ = n;
    return *this;
}

Config::Builder& Config::Builder::serverMode(bool enabled) noexcept
{
    serverMode_ = enabled;
    return *this;
}

Config::Builder& Config::Builder::headless(bool enabled) noexcept
{
    headless_ = enabled;
    return *this;
}

Config::Builder& Config::Builder::arenaSize(core::usize bytes) noexcept
{
    arenaSize_ = bytes;
    return *this;
}

Config::Builder& Config::Builder::enableBci(bool enabled) noexcept
{
    enableBci_ = enabled;
    return *this;
}

Config::Builder& Config::Builder::enableGpu(bool enabled) noexcept
{
    enableGpu_ = enabled;
    return *this;
}

Config Config::Builder::build() const noexcept
{
    Config cfg;
    cfg.tickRate_    = tickRate_;
    cfg.maxEntities_ = maxEntities_;
    cfg.maxChunks_   = maxChunks_;
    cfg.serverMode_  = serverMode_;
    cfg.headless_    = headless_;
    cfg.arenaSize_   = arenaSize_;
    cfg.enableBci_   = enableBci_;
    cfg.enableGpu_   = enableGpu_;
    return cfg;
}

} // namespace lpl::engine
