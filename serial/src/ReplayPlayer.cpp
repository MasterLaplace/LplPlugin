/**
 * @file ReplayPlayer.cpp
 * @brief ReplayPlayer stub implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/serial/ReplayPlayer.hpp>
#include <stdexcept>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::serial {

struct ReplayPlayer::Impl
{
    PlaybackState state{PlaybackState::Idle};
    core::u64 currentTick{0};
    std::vector<ReplayFrame> frames;
    std::vector<StateSnapshot> snapshots;
};

ReplayPlayer::ReplayPlayer() : _impl{std::make_unique<Impl>()} {}
ReplayPlayer::~ReplayPlayer() = default;

core::Expected<void> ReplayPlayer::loadFromDisk(std::string_view /*path*/)
{
    LPL_ASSERT(false && "unimplemented");
    return {};
}

void ReplayPlayer::play()
{
    _impl->state = PlaybackState::Playing;
}

void ReplayPlayer::pause()
{
    _impl->state = PlaybackState::Paused;
}

core::Expected<ReplayFrame> ReplayPlayer::advanceTick()
{
    if (_impl->currentTick >= _impl->frames.size())
    {
        _impl->state = PlaybackState::Finished;
        return core::makeError(core::ErrorCode::kOutOfRange, "Replay finished");
    }

    const auto& frame = _impl->frames[_impl->currentTick];
    ++_impl->currentTick;
    return frame;
}

core::Expected<void> ReplayPlayer::seekToTick(core::u64 /*tick*/)
{
    LPL_ASSERT(false && "unimplemented");
    return {};
}

core::u64 ReplayPlayer::currentTick() const noexcept
{
    return _impl->currentTick;
}

PlaybackState ReplayPlayer::state() const noexcept
{
    return _impl->state;
}

core::usize ReplayPlayer::totalFrames() const noexcept
{
    return _impl->frames.size();
}

} // namespace lpl::serial
