#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/serial/ReplayPlayer.hpp>

namespace lpl::serial {

struct ReplayPlayer::Impl {
    PlaybackState state{PlaybackState::Idle};
    core::u64 currentTick{0};
    std::vector<ReplayFrame> frames;
    std::vector<StateSnapshot> snapshots;
};

ReplayPlayer::ReplayPlayer() : _impl{std::make_unique<Impl>()} {}
ReplayPlayer::~ReplayPlayer() = default;

core::Expected<void> ReplayPlayer::loadFromDisk(std::string_view /*path*/)
{
    // TODO(#141): read [frameCount][frames...][snapshotCount][snapshots...] from the replay file.
    LPL_ASSERT(false && "unimplemented");
    return {};
}

void ReplayPlayer::play() { _impl->state = PlaybackState::Playing; }

void ReplayPlayer::pause() { _impl->state = PlaybackState::Paused; }

core::Expected<ReplayFrame> ReplayPlayer::advanceTick()
{
    if (_impl->currentTick >= _impl->frames.size())
    {
        _impl->state = PlaybackState::Finished;
        return core::makeError(core::ErrorCode::kOutOfRange, "Replay finished");
    }

    const auto &frame = _impl->frames[_impl->currentTick];
    ++_impl->currentTick;
    return frame;
}

core::Expected<void> ReplayPlayer::seekToTick(core::u64 /*tick*/)
{
    // TODO(#141): binary-search the snapshots for the closest tick, then replay forwards.
    LPL_ASSERT(false && "unimplemented");
    return {};
}

core::u64 ReplayPlayer::currentTick() const noexcept { return _impl->currentTick; }

PlaybackState ReplayPlayer::state() const noexcept { return _impl->state; }

core::usize ReplayPlayer::totalFrames() const noexcept { return _impl->frames.size(); }

} // namespace lpl::serial
