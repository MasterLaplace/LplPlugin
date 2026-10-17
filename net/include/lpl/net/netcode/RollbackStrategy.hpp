/**
 * @file RollbackStrategy.hpp
 * @brief GGPO-style rollback netcode strategy.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_NETCODE_ROLLBACKSTRATEGY_HPP
#    define LPL_NET_NETCODE_ROLLBACKSTRATEGY_HPP

#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/net/netcode/INetcodeStrategy.hpp>

#    include <memory>

namespace lpl::net::netcode {

/**
 * @class RollbackStrategy
 * @brief Rollback / re-simulate netcode for latency-sensitive scenarios.
 *
 * Both peers simulate locally. When a remote input arrives for an already-
 * simulated tick, the world state is rolled back to that tick, re-simulated
 * with the corrected input, and fast-forwarded to the present.
 */
class RollbackStrategy final : public INetcodeStrategy, public core::NonCopyable<RollbackStrategy> {
public:
    /**
     * @brief Constructs a rollback strategy.
     * @param maxRollbackFrames Maximum number of frames that can be rolled
     *        back (ring buffer depth for state snapshots).
     */
    explicit RollbackStrategy(core::u32 maxRollbackFrames = 8);
    ~RollbackStrategy() override;

    /**
     * @brief Flags a rollback when the input is for a past frame.
     *
     * An input for a past frame within the rollback window flags a rollback to the frame where it
     * should have been applied, unless one to an earlier frame is already pending.
     */
    [[nodiscard]] core::Expected<void> onInputReceived(core::u32 playerId, std::span<const core::byte> inputData,
                                                       core::u32 sequence) override;

    /**
     * @brief Stores the authoritative state, and flags a rollback when it is for a past frame.
     *
     * A state for a past frame within the rollback window flags a rollback for correction, unless
     * one to an earlier frame is already pending.
     */
    [[nodiscard]] core::Expected<void> onStateReceived(std::span<const core::byte> snapshotData,
                                                       core::u32 sequence) override;

    /**
     * @brief Advances the current frame.
     *
     * A full implementation would trigger the resimulation loop here. Since systems handle their
     * own data, this one only clears a pending rollback after one tick.
     */
    void tick(core::f32 dt) override;

    [[nodiscard]] const char *name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net::netcode

#endif // LPL_NET_NETCODE_ROLLBACKSTRATEGY_HPP
