// /////////////////////////////////////////////////////////////////////////////
/// @file ReplayPlayer.cpp
/// @brief ReplayPlayer stub implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/serial/ReplayPlayer.hpp>
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

ReplayPlayer::ReplayPlayer() : impl_{std::make_unique<Impl>()} {}
ReplayPlayer::~ReplayPlayer() = default;

core::Expected<void> ReplayPlayer::loadFromDisk(std::string_view /*path*/)
{
    LPL_ASSERT(false && "ReplayPlayer::loadFromDisk not yet implemented");
    return {};
}

void ReplayPlayer::play()
{
    impl_->state = PlaybackState::Playing;
}

void ReplayPlayer::pause()
{
    impl_->state = PlaybackState::Paused;
}

core::Expected<ReplayFrame> ReplayPlayer::advanceTick()
{
    if (impl_->currentTick >= impl_->frames.size())
    {
        impl_->state = PlaybackState::Finished;
        return core::makeError(core::ErrorCode::kOutOfRange, "Replay finished");
    }

    const auto& frame = impl_->frames[impl_->currentTick];
    ++impl_->currentTick;
    return frame;
}

core::Expected<void> ReplayPlayer::seekToTick(core::u64 /*tick*/)
{
    LPL_ASSERT(false && "ReplayPlayer::seekToTick not yet implemented");
    return {};
}

core::u64 ReplayPlayer::currentTick() const noexcept
{
    return impl_->currentTick;
}

PlaybackState ReplayPlayer::state() const noexcept
{
    return impl_->state;
}

core::usize ReplayPlayer::totalFrames() const noexcept
{
    return impl_->frames.size();
}

} // namespace lpl::serial
