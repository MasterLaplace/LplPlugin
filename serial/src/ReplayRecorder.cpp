/**
 * @file ReplayRecorder.cpp
 * @brief ReplayRecorder implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/serial/ReplayRecorder.hpp>
#include <stdexcept>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::serial {

ReplayRecorder::ReplayRecorder(core::u32 snapshotInterval)
    : _snapshotInterval{snapshotInterval}
{
    LPL_ASSERT(snapshotInterval > 0);
}

ReplayRecorder::~ReplayRecorder() = default;

void ReplayRecorder::recordFrame(const ReplayFrame& frame)
{
    _frames.push_back(frame);
}

void ReplayRecorder::recordSnapshot(StateSnapshot snapshot)
{
    _snapshots.push_back(std::move(snapshot));
}

core::usize ReplayRecorder::frameCount() const noexcept
{
    return _frames.size();
}

core::usize ReplayRecorder::snapshotCount() const noexcept
{
    return _snapshots.size();
}

const ReplayFrame& ReplayRecorder::frame(core::usize index) const
{
    LPL_ASSERT(index < _frames.size());
    return _frames[index];
}

core::Expected<void> ReplayRecorder::saveToDisk(std::string_view /*path*/) const
{
    LPL_ASSERT(false && "unimplemented");
    return {};
}

void ReplayRecorder::clear() noexcept
{
    _frames.clear();
    _snapshots.clear();
}

} // namespace lpl::serial
