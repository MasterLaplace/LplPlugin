// /////////////////////////////////////////////////////////////////////////////
/// @file INetcodeStrategy.hpp
/// @brief Abstract netcode strategy interface (Strategy pattern).
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

#include <span>

namespace lpl::net::netcode {

// /////////////////////////////////////////////////////////////////////////////
/// @class INetcodeStrategy
/// @brief Strategy interface for netcode models.
///
/// Concrete strategies:
///   - @c AuthoritativeStrategy — server-authoritative with client prediction.
///   - @c RollbackStrategy — GGPO-style rollback for competitive scenarios.
// /////////////////////////////////////////////////////////////////////////////
class INetcodeStrategy
{
public:
    virtual ~INetcodeStrategy() = default;

    /// @brief Processes a received input payload.
    /// @param playerId Source player.
    /// @param inputData Serialised input bits.
    /// @param sequence  Sequence number of the input.
    [[nodiscard]] virtual core::Expected<void> onInputReceived(
        core::u32 playerId,
        std::span<const core::byte> inputData,
        core::u32 sequence) = 0;

    /// @brief Processes a received state snapshot / delta.
    /// @param snapshotData Serialised state bytes.
    /// @param sequence     Server tick number.
    [[nodiscard]] virtual core::Expected<void> onStateReceived(
        std::span<const core::byte> snapshotData,
        core::u32 sequence) = 0;

    /// @brief Advances the netcode by one tick.
    /// @param dt Fixed delta-time.
    virtual void tick(core::f32 dt) = 0;

    /// @brief Returns a human-readable name.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::net::netcode
