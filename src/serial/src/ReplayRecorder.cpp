// /////////////////////////////////////////////////////////////////////////////
/// @file ReplayRecorder.cpp
/// @brief ReplayRecorder implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/serial/ReplayRecorder.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::serial {

ReplayRecorder::ReplayRecorder(core::u32 snapshotInterval)
    : snapshotInterval_{snapshotInterval}
{
    LPL_ASSERT(snapshotInterval > 0);
}

ReplayRecorder::~ReplayRecorder() = default;

void ReplayRecorder::recordFrame(const ReplayFrame& frame)
{
    frames_.push_back(frame);
}

void ReplayRecorder::recordSnapshot(StateSnapshot snapshot)
{
    snapshots_.push_back(std::move(snapshot));
}

core::usize ReplayRecorder::frameCount() const noexcept
{
    return frames_.size();
}

core::usize ReplayRecorder::snapshotCount() const noexcept
{
    return snapshots_.size();
}

const ReplayFrame& ReplayRecorder::frame(core::usize index) const
{
    LPL_ASSERT(index < frames_.size());
    return frames_[index];
}

core::Expected<void> ReplayRecorder::saveToDisk(std::string_view /*path*/) const
{
    LPL_ASSERT(false && "ReplayRecorder::saveToDisk not yet implemented");
    return {};
}

void ReplayRecorder::clear() noexcept
{
    frames_.clear();
    snapshots_.clear();
}

} // namespace lpl::serial
