// /////////////////////////////////////////////////////////////////////////////
/// @file ReplayPlayer.hpp
/// @brief Deterministic replay playback engine.
// /////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lpl/serial/ReplayRecorder.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <string_view>
#include <memory>

namespace lpl::serial {

/// @brief Playback state.
enum class PlaybackState : core::u8
{
    Idle,
    Playing,
    Paused,
    Finished
};

/// @brief Plays back a recorded replay by feeding inputs frame-by-frame.
///
/// Loads from disk, seeks to a snapshot for fast-forward, then
/// replays inputs tick-by-tick through the simulation.
class ReplayPlayer
{
public:
    ReplayPlayer();
    ~ReplayPlayer();

    ReplayPlayer(const ReplayPlayer&) = delete;
    ReplayPlayer& operator=(const ReplayPlayer&) = delete;

    /// @brief Load a replay file from disk.
    /// @param path Binary replay file.
    /// @return Success or error.
    [[nodiscard]] core::Expected<void> loadFromDisk(std::string_view path);

    /// @brief Start playback from tick 0 (or last seek position).
    void play();

    /// @brief Pause playback.
    void pause();

    /// @brief Advance one tick, returning the input for that tick.
    /// @return The ReplayFrame for the current tick, or error if finished.
    [[nodiscard]] core::Expected<ReplayFrame> advanceTick();

    /// @brief Seek to a specific tick (uses nearest prior snapshot).
    /// @param tick Target tick number.
    /// @return Success or error.
    [[nodiscard]] core::Expected<void> seekToTick(core::u64 tick);

    /// @brief Current playback tick.
    [[nodiscard]] core::u64 currentTick() const noexcept;

    /// @brief Current playback state.
    [[nodiscard]] PlaybackState state() const noexcept;

    /// @brief Total number of frames in the loaded replay.
    [[nodiscard]] core::usize totalFrames() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::serial
