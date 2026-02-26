// /////////////////////////////////////////////////////////////////////////////
/// @file Config.hpp
/// @brief Engine configuration (Builder pattern).
///
/// Immutable configuration object constructed via a fluent Builder.
/// Centralises all tuneable engine parameters.
// /////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Constants.hpp>

namespace lpl::engine {

/// @brief Immutable engine configuration.
class Config
{
public:
    /// @brief Fluent builder for Config.
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
        core::u32 tickRate_{core::kTickRate};
        core::u32 maxEntities_{core::kMaxEntities};
        core::u32 maxChunks_{core::kMaxChunks};
        bool serverMode_{false};
        bool headless_{false};
        core::usize arenaSize_{64 * 1024 * 1024};
        bool enableBci_{false};
        bool enableGpu_{false};
    };

    [[nodiscard]] core::u32  tickRate()    const noexcept { return tickRate_; }
    [[nodiscard]] core::u32  maxEntities() const noexcept { return maxEntities_; }
    [[nodiscard]] core::u32  maxChunks()   const noexcept { return maxChunks_; }
    [[nodiscard]] bool       serverMode()  const noexcept { return serverMode_; }
    [[nodiscard]] bool       headless()    const noexcept { return headless_; }
    [[nodiscard]] core::usize arenaSize()  const noexcept { return arenaSize_; }
    [[nodiscard]] bool       enableBci()   const noexcept { return enableBci_; }
    [[nodiscard]] bool       enableGpu()   const noexcept { return enableGpu_; }

private:
    friend class Builder;

    core::u32  tickRate_{core::kTickRate};
    core::u32  maxEntities_{core::kMaxEntities};
    core::u32  maxChunks_{core::kMaxChunks};
    bool       serverMode_{false};
    bool       headless_{false};
    core::usize arenaSize_{64 * 1024 * 1024};
    bool       enableBci_{false};
    bool       enableGpu_{false};
};

} // namespace lpl::engine
