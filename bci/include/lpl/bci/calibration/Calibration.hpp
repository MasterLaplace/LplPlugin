/**
 * @file Calibration.hpp
 * @brief Calibration state machine for BCI baseline acquisition.
 * @author MasterLaplace
 *
 * Implements a three-state machine (Idle → Calibrating → Ready) with an
 * Observer pattern for state-change notifications. During calibration,
 * collects per-channel alpha and beta power samples to compute the baselines
 * required by NeuralMetric normalization.
 *
 * @see NeuralMetric, ChannelBaseline
 */

#pragma once

#include "lpl/bci/core/Error.hpp"
#include "lpl/bci/core/Types.hpp"
#include "lpl/bci/metric/NeuralMetric.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace lpl::bci::calibration {

/**
 * @brief Calibration state machine states.
 */
enum class CalibrationState : std::uint8_t {
    kIdle,
    kCalibrating,
    kReady,
};

/**
 * @brief Observer callback fired on state transitions.
 */
using StateChangeCallback = std::function<void(CalibrationState oldState, CalibrationState newState)>;

/**
 * @brief Configuration for the calibration process.
 */
struct CalibrationConfig {
    std::size_t channelCount   = 8;
    std::size_t requiredTrials = 30;
};

/**
 * @brief Manages BCI calibration and baseline computation.
 *
 * Usage:
 * 1. Call `start()` to transition from Idle to Calibrating
 * 2. Feed alpha/beta power vectors via `addTrial()`
 * 3. When enough trials are collected, automatically transitions to Ready
 * 4. Retrieve baselines via `baselines()`
 */
class Calibration {
public:
    /**
     * @brief Constructs the calibration with the given configuration.
     *
     * @param config Channel count and required trial count
     */
    explicit Calibration(const CalibrationConfig& config = {});

    /**
     * @brief Registers an observer for state-change notifications.
     *
     * @param callback Function to call on state transitions
     */
    void onStateChange(StateChangeCallback callback);

    /**
     * @brief Begins calibration, transitioning from Idle to Calibrating.
     *
     * @return Error if already calibrating or already ready
     */
    [[nodiscard]] ExpectedVoid start();

    /**
     * @brief Feeds one trial's per-channel alpha and beta powers.
     *
     * Automatically transitions to Ready when enough trials have been collected.
     *
     * @param alphaPowers Per-channel alpha power (size == channelCount)
     * @param betaPowers  Per-channel beta power (size == channelCount)
     * @return Error if not currently calibrating or size mismatch
     */
    [[nodiscard]] ExpectedVoid addTrial(
        std::span<const float> alphaPowers,
        std::span<const float> betaPowers);

    /**
     * @brief Returns the computed per-channel baselines.
     *
     * @return Baselines, or Error if calibration is not complete
     */
    [[nodiscard]] Expected<std::vector<metric::ChannelBaseline>> baselines() const;

    /**
     * @brief Resets the calibration to Idle.
     */
    void reset() noexcept;

    /**
     * @brief Current state.
     */
    [[nodiscard]] CalibrationState state() const noexcept;

    /**
     * @brief Number of trials collected so far.
     */
    [[nodiscard]] std::size_t trialCount() const noexcept;

    /**
     * @brief Number of trials required.
     */
    [[nodiscard]] std::size_t requiredTrials() const noexcept;

private:
    void transition(CalibrationState newState);
    void computeBaselines() noexcept;

    CalibrationConfig _config;
    CalibrationState _state = CalibrationState::kIdle;
    std::vector<StateChangeCallback> _observers;

    std::vector<std::vector<float>> _alphaHistory;
    std::vector<std::vector<float>> _betaHistory;
    std::vector<metric::ChannelBaseline> _baselines;
};

} // namespace lpl::bci::calibration
