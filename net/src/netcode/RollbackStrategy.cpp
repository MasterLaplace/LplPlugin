#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/netcode/RollbackStrategy.hpp>

#include <vector>

namespace lpl::net::netcode {

struct RollbackStrategy::Impl {
    core::u32 maxRollbackFrames;
    core::u32 currentFrame{0};

    std::vector<std::vector<core::byte>> stateHistory; /**< Ring of serialised states, index = frame % max. */

    std::vector<core::byte> latestServerState; /**< Latest authoritative state received from server. */
    core::u32 latestServerSequence{0};

    core::u32 rollbackTargetFrame{0}; /**< Frame at which rollback should start, while rollbackPending. */
    bool rollbackPending{false};

    explicit Impl(core::u32 maxFrames) : maxRollbackFrames{maxFrames} { stateHistory.resize(maxFrames); }

    void saveState(core::u32 frame, std::span<const core::byte> data)
    {
        auto &slot = stateHistory[frame % maxRollbackFrames];
        slot.assign(data.begin(), data.end());
    }

    const std::vector<core::byte> &getState(core::u32 frame) const { return stateHistory[frame % maxRollbackFrames]; }
};

RollbackStrategy::RollbackStrategy(core::u32 maxRollbackFrames) : _impl{std::make_unique<Impl>(maxRollbackFrames)} {}

RollbackStrategy::~RollbackStrategy() = default;

core::Expected<void> RollbackStrategy::onInputReceived(core::u32 playerId, std::span<const core::byte> inputData,
                                                       core::u32 sequence)
{
    (void) playerId;
    (void) inputData;

    if (sequence < _impl->currentFrame)
    {
        core::u32 framesBack = _impl->currentFrame - sequence;
        if (framesBack <= _impl->maxRollbackFrames)
        {
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
    _impl->latestServerState.assign(snapshotData.begin(), snapshotData.end());
    _impl->latestServerSequence = sequence;

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

    if (_impl->rollbackPending)
    {
        _impl->rollbackPending = false;
        _impl->rollbackTargetFrame = 0;
    }

    _impl->currentFrame++;
}

const char *RollbackStrategy::name() const noexcept { return "RollbackStrategy"; }

} // namespace lpl::net::netcode
