// /////////////////////////////////////////////////////////////////////////////
/// @file ReplayRecorder.hpp
/// @brief Records tick-by-tick input + snapshots for deterministic replay.
// /////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lpl/serial/StateSnapshot.hpp>
#include <lpl/input/InputState.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <vector>
#include <string_view>

namespace lpl::serial {

/// @brief One frame of replay data.
struct ReplayFrame
{
    core::u64 tick{0};
    input::InputState inputState{};
};

/// @brief Records a replay by storing input frames and periodic snapshots.
///
/// Snapshots are taken every N ticks (configurable) to allow
/// fast-forward / seek during playback.
class ReplayRecorder
{
public:
    /// @param snapshotInterval Ticks between automatic snapshots.
    explicit ReplayRecorder(core::u32 snapshotInterval = 144);
    ~ReplayRecorder();

    ReplayRecorder(const ReplayRecorder&) = delete;
    ReplayRecorder& operator=(const ReplayRecorder&) = delete;

    /// @brief Record one tick's input.
    void recordFrame(const ReplayFrame& frame);

    /// @brief Record a full state snapshot.
    void recordSnapshot(StateSnapshot snapshot);

    /// @brief Total recorded frames.
    [[nodiscard]] core::usize frameCount() const noexcept;

    /// @brief Total recorded snapshots.
    [[nodiscard]] core::usize snapshotCount() const noexcept;

    /// @brief Access a recorded frame by index.
    [[nodiscard]] const ReplayFrame& frame(core::usize index) const;

    /// @brief Save the replay to a binary file.
    /// @param path Output file path.
    /// @return Success or error.
    [[nodiscard]] core::Expected<void> saveToDisk(std::string_view path) const;

    /// @brief Clear all recorded data.
    void clear() noexcept;

private:
    core::u32 snapshotInterval_;
    std::vector<ReplayFrame> frames_;
    std::vector<StateSnapshot> snapshots_;
};

} // namespace lpl::serial
