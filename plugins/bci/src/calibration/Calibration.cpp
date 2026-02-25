/**
 * @file Calibration.cpp
 * @brief Implementation of the calibration state machine.
 */

#include "calibration/Calibration.hpp"
#include "math/Statistics.hpp"

#include <format>

namespace bci::calibration {

Calibration::Calibration(const CalibrationConfig& config)
    : _config(config)
{
    _alphaHistory.resize(_config.channelCount);
    _betaHistory.resize(_config.channelCount);
}

void Calibration::onStateChange(StateChangeCallback callback)
{
    _observers.push_back(std::move(callback));
}

ExpectedVoid Calibration::start()
{
    if (_state != CalibrationState::kIdle)
        return std::unexpected(Error{
            ErrorCode::kInvalidState,
            std::format("cannot start calibration from state {}", static_cast<int>(_state))});

    for (auto& ch : _alphaHistory)
        ch.clear();
    for (auto& ch : _betaHistory)
        ch.clear();

    transition(CalibrationState::kCalibrating);
    return {};
}

ExpectedVoid Calibration::addTrial(
    std::span<const float> alphaPowers,
    std::span<const float> betaPowers)
{
    if (_state != CalibrationState::kCalibrating)
        return std::unexpected(Error{
            ErrorCode::kInvalidState,
            "addTrial() requires CalibrationState::kCalibrating"});

    if (alphaPowers.size() < _config.channelCount || betaPowers.size() < _config.channelCount)
        return std::unexpected(Error{
            ErrorCode::kInvalidArgument,
            std::format("expected {} channels, got alpha={} beta={}",
                _config.channelCount, alphaPowers.size(), betaPowers.size())});

    for (std::size_t ch = 0; ch < _config.channelCount; ++ch) {
        _alphaHistory[ch].push_back(alphaPowers[ch]);
        _betaHistory[ch].push_back(betaPowers[ch]);
    }

    if (_alphaHistory[0].size() >= _config.requiredTrials) {
        computeBaselines();
        transition(CalibrationState::kReady);
    }

    return {};
}

Expected<std::vector<metric::ChannelBaseline>> Calibration::baselines() const
{
    if (_state != CalibrationState::kReady)
        return std::unexpected(Error{
            ErrorCode::kInvalidState,
            "baselines unavailable: calibration not complete"});

    return _baselines;
}

void Calibration::reset() noexcept
{
    _state = CalibrationState::kIdle;

    for (auto& ch : _alphaHistory)
        ch.clear();
    for (auto& ch : _betaHistory)
        ch.clear();

    _baselines.clear();
}

CalibrationState Calibration::state() const noexcept
{
    return _state;
}

std::size_t Calibration::trialCount() const noexcept
{
    if (_alphaHistory.empty())
        return 0;
    return _alphaHistory[0].size();
}

std::size_t Calibration::requiredTrials() const noexcept
{
    return _config.requiredTrials;
}

void Calibration::transition(CalibrationState newState)
{
    const auto oldState = _state;
    _state = newState;

    for (const auto& observer : _observers)
        observer(oldState, newState);
}

void Calibration::computeBaselines() noexcept
{
    _baselines.resize(_config.channelCount);

    for (std::size_t ch = 0; ch < _config.channelCount; ++ch) {
        _baselines[ch].alpha = math::Statistics::computeBaseline(_alphaHistory[ch]);
        _baselines[ch].beta  = math::Statistics::computeBaseline(_betaHistory[ch]);
    }
}

} // namespace bci::calibration
