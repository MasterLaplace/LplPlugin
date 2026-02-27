/**
 * @file ReplayRecorder.hpp
 * @brief Records tick-by-tick input + snapshots for deterministic replay.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_SERIAL_REPLAYRECORDER_HPP
    #define LPL_SERIAL_REPLAYRECORDER_HPP

#include <lpl/serial/StateSnapshot.hpp>
#include <lpl/input/InputState.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <vector>
#include <string_view>

namespace lpl::serial {

/** @brief One frame of replay data. */
struct ReplayFrame
{
    core::u64 tick{0};
    input::InputState inputState{};
};

/**
 * @brief Records a replay by storing input frames and periodic snapshots.
 *
 * Snapshots are taken every N ticks (configurable) to allow
 * fast-forward / seek during playback.
 */
class ReplayRecorder
{
public:
    /// @param snapshotInterval Ticks between automatic snapshots.
    explicit ReplayRecorder(core::u32 snapshotInterval = 144);
    ~ReplayRecorder();

    ReplayRecorder(const ReplayRecorder&) = delete;
    ReplayRecorder& operator=(const ReplayRecorder&) = delete;

    /** @brief Record one tick's input. */
    void recordFrame(const ReplayFrame& frame);

    /** @brief Record a full state snapshot. */
    void recordSnapshot(StateSnapshot snapshot);

    /** @brief Total recorded frames. */
    [[nodiscard]] core::usize frameCount() const noexcept;

    /** @brief Total recorded snapshots. */
    [[nodiscard]] core::usize snapshotCount() const noexcept;

    /** @brief Access a recorded frame by index. */
    [[nodiscard]] const ReplayFrame& frame(core::usize index) const;

    /**
     * @brief Save the replay to a binary file.
     * @param path Output file path.
     * @return Success or error.
     */
    [[nodiscard]] core::Expected<void> saveToDisk(std::string_view path) const;

    /** @brief Clear all recorded data. */
    void clear() noexcept;

private:
    core::u32 _snapshotInterval;
    std::vector<ReplayFrame> _frames;
    std::vector<StateSnapshot> _snapshots;
};

} // namespace lpl::serial

#endif // LPL_SERIAL_REPLAYRECORDER_HPP
