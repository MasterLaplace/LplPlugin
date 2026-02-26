// /////////////////////////////////////////////////////////////////////////////
/// @file RollbackStrategy.hpp
/// @brief GGPO-style rollback netcode strategy.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/net/netcode/INetcodeStrategy.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::net::netcode {

// /////////////////////////////////////////////////////////////////////////////
/// @class RollbackStrategy
/// @brief Rollback / re-simulate netcode for latency-sensitive scenarios.
///
/// Both peers simulate locally. When a remote input arrives for an already-
/// simulated tick, the world state is rolled back to that tick, re-simulated
/// with the corrected input, and fast-forwarded to the present.
// /////////////////////////////////////////////////////////////////////////////
class RollbackStrategy final : public INetcodeStrategy,
                                public core::NonCopyable<RollbackStrategy>
{
public:
    /// @brief Constructs a rollback strategy.
    /// @param maxRollbackFrames Maximum number of frames that can be rolled
    ///        back (ring buffer depth for state snapshots).
    explicit RollbackStrategy(core::u32 maxRollbackFrames = 8);
    ~RollbackStrategy() override;

    [[nodiscard]] core::Expected<void> onInputReceived(
        core::u32 playerId,
        std::span<const core::byte> inputData,
        core::u32 sequence) override;

    [[nodiscard]] core::Expected<void> onStateReceived(
        std::span<const core::byte> snapshotData,
        core::u32 sequence) override;

    void tick(core::f32 dt) override;

    [[nodiscard]] const char* name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::net::netcode
