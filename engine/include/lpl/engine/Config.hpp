/**
 * @file Config.hpp
 * @brief Engine configuration (Builder pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_ENGINE_CONFIG_HPP
    #define LPL_ENGINE_CONFIG_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/Constants.hpp>

namespace lpl::engine {

/** @brief Immutable engine configuration. */
class Config
{
public:
    /** @brief Fluent builder for Config. */
    class Builder
    {
    public:
        Builder& tickRate(core::u32 hz) noexcept;
        Builder& maxEntities(core::u32 n) noexcept;
        Builder& maxChunks(core::u32 n) noexcept;
        Builder& serverMode(bool enabled) noexcept;
        Builder& headless(bool enabled) noexcept;
        Builder& arenaSize(core::usize bytes) noexcept;
        Builder& enableBci(bool enabled) noexcept;
        Builder& enableGpu(bool enabled) noexcept;

        [[nodiscard]] Config build() const noexcept;

    private:
        core::u32 _tickRate{core::kTickRate};
        core::u32 _maxEntities{core::kMaxEntities};
        core::u32 _maxChunks{core::kMaxChunks};
        bool _serverMode{false};
        bool _headless{false};
        core::usize _arenaSize{64 * 1024 * 1024};
        bool _enableBci{false};
        bool _enableGpu{false};
    };

    [[nodiscard]] core::u32  tickRate()    const noexcept { return _tickRate; }
    [[nodiscard]] core::u32  maxEntities() const noexcept { return _maxEntities; }
    [[nodiscard]] core::u32  maxChunks()   const noexcept { return _maxChunks; }
    [[nodiscard]] bool       serverMode()  const noexcept { return _serverMode; }
    [[nodiscard]] bool       headless()    const noexcept { return _headless; }
    [[nodiscard]] core::usize arenaSize()  const noexcept { return _arenaSize; }
    [[nodiscard]] bool       enableBci()   const noexcept { return _enableBci; }
    [[nodiscard]] bool       enableGpu()   const noexcept { return _enableGpu; }

private:
    friend class Builder;

    core::u32  _tickRate{core::kTickRate};
    core::u32  _maxEntities{core::kMaxEntities};
    core::u32  _maxChunks{core::kMaxChunks};
    bool       _serverMode{false};
    bool       _headless{false};
    core::usize _arenaSize{64 * 1024 * 1024};
    bool       _enableBci{false};
    bool       _enableGpu{false};
};

} // namespace lpl::engine

#endif // LPL_ENGINE_CONFIG_HPP
