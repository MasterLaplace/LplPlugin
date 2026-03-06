/**
 * @file RollbackStrategy.cpp
 * @brief Rollback netcode: ring-buffer of state snapshots, rollback on
 *        late input or stale server state.
 *
 * The strategy maintains a circular buffer of serialized state snapshots.
 * When a late input or authority correction arrives, it flags a rollback
 * so that external systems can:
 *   1. Restore the snapshot at the divergence frame
 *   2. Re-simulate forward with the corrected inputs
 *   3. Resume normal play at the current frame
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/netcode/RollbackStrategy.hpp>

#include <vector>

namespace lpl::net::netcode {

// ========================================================================== //
//  Impl                                                                       //
// ========================================================================== //

struct RollbackStrategy::Impl {
    core::u32 maxRollbackFrames;
    core::u32 currentFrame{0};

    /** @brief Ring buffer of serialised state snapshots. Index = frame % max. */
    std::vector<std::vector<core::byte>> stateHistory;

    /** @brief Latest authoritative state received from server. */
    std::vector<core::byte> latestServerState;
    core::u32 latestServerSequence{0};

    /** @brief Frame at which rollback should start (0 = no rollback). */
    core::u32 rollbackTargetFrame{0};
    bool rollbackPending{false};

    explicit Impl(core::u32 maxFrames) : maxRollbackFrames{maxFrames} { stateHistory.resize(maxFrames); }

    void saveState(core::u32 frame, std::span<const core::byte> data)
    {
        auto &slot = stateHistory[frame % maxRollbackFrames];
        slot.assign(data.begin(), data.end());
    }

    const std::vector<core::byte> &getState(core::u32 frame) const { return stateHistory[frame % maxRollbackFrames]; }
};

// ========================================================================== //
//  Public                                                                     //
// ========================================================================== //

RollbackStrategy::RollbackStrategy(core::u32 maxRollbackFrames) : _impl{std::make_unique<Impl>(maxRollbackFrames)} {}

RollbackStrategy::~RollbackStrategy() = default;

core::Expected<void> RollbackStrategy::onInputReceived(core::u32 playerId, std::span<const core::byte> inputData,
                                                       core::u32 sequence)
{
    (void) playerId;
    (void) inputData;

    // If the input is for a past frame within the rollback window, flag rollback
    if (sequence < _impl->currentFrame)
    {
        core::u32 framesBack = _impl->currentFrame - sequence;
        if (framesBack <= _impl->maxRollbackFrames)
        {
            // Flag rollback to the frame where this input should have been applied
            if (!_impl->rollbackPending || sequence < _impl->rollbackTargetFrame)
            {
                _impl->rollbackTargetFrame = sequence;
                _impl->rollbackPending = true;
                core::Log::debug("RollbackStrategy", "late input detected, rollback flagged");
            }
        }
    }

    return {};
}

core::Expected<void> RollbackStrategy::onStateReceived(std::span<const core::byte> snapshotData, core::u32 sequence)
{
    // Store the authoritative state for reconciliation
    _impl->latestServerState.assign(snapshotData.begin(), snapshotData.end());
    _impl->latestServerSequence = sequence;

    // If the server state is for a past frame, flag rollback for correction
    if (sequence < _impl->currentFrame)
    {
        core::u32 framesBack = _impl->currentFrame - sequence;
        if (framesBack <= _impl->maxRollbackFrames)
        {
            if (!_impl->rollbackPending || sequence < _impl->rollbackTargetFrame)
            {
                _impl->rollbackTargetFrame = sequence;
                _impl->rollbackPending = true;
                core::Log::debug("RollbackStrategy", "server correction received, rollback flagged");
            }
        }
    }

    return {};
}

void RollbackStrategy::tick(core::f32 dt)
{
    (void) dt;

    // If rollback is pending, external code should:
    // 1. Read rollbackTargetFrame
    // 2. Restore state from getState(rollbackTargetFrame)
    // 3. Re-simulate frames [target..current] with corrected inputs
    // 4. Clear the rollback flag
    // This is handled by the higher-level engine code that reads the flag.

    if (_impl->rollbackPending)
    {
        // In a full implementation, this is where we'd trigger the resimulation
        // loop. Since systems handle their own data, we just clear the flag
        // after one tick to signal that the rollback window was consumed.
        _impl->rollbackPending = false;
        _impl->rollbackTargetFrame = 0;
    }

    _impl->currentFrame++;
}

const char *RollbackStrategy::name() const noexcept { return "RollbackStrategy"; }

} // namespace lpl::net::netcode
