/**
 * @file NeuralSafetyChain.cpp
 * @brief NeuralSafetyChain and built-in checks implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/bci/NeuralSafetyChain.hpp>
#include <lpl/core/Log.hpp>
#include <cmath>

namespace lpl::bci {

// ─────────────────────────────────────────────────────────────────────────────
// NeuralSafetyChain
// ─────────────────────────────────────────────────────────────────────────────

NeuralSafetyChain::NeuralSafetyChain() = default;
NeuralSafetyChain::~NeuralSafetyChain() = default;

void NeuralSafetyChain::addCheck(std::unique_ptr<ISafetyCheck> check)
{
    _checks.push_back(std::move(check));
}

SafetyVerdict NeuralSafetyChain::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    SafetyVerdict worst = SafetyVerdict::Pass;

    for (const auto& check : _checks)
    {
        const SafetyVerdict v = check->evaluate(state);

        if (v == SafetyVerdict::Reject)
        {
            core::Log::warn("NeuralSafetyChain: REJECT by check");
            return SafetyVerdict::Reject;
        }

        if (v == SafetyVerdict::Warn)
        {
            worst = SafetyVerdict::Warn;
        }
    }

    return worst;
}

core::usize NeuralSafetyChain::checkCount() const noexcept
{
    return _checks.size();
}

// ─────────────────────────────────────────────────────────────────────────────
// AmplitudeBoundsCheck
// ─────────────────────────────────────────────────────────────────────────────

AmplitudeBoundsCheck::AmplitudeBoundsCheck(core::f32 maxAbsValue)
    : _maxAbsValue{maxAbsValue} {}

SafetyVerdict AmplitudeBoundsCheck::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    for (core::usize i = 0; i < input::NeuralInputState::kChannels; ++i)
    {
        const core::f32 val = state.channels[i].toFloat();
        if (std::fabs(val) > _maxAbsValue)
        {
            return SafetyVerdict::Reject;
        }
    }
    return SafetyVerdict::Pass;
}

const char* AmplitudeBoundsCheck::name() const noexcept
{
    return "AmplitudeBoundsCheck";
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfidenceCheck
// ─────────────────────────────────────────────────────────────────────────────

ConfidenceCheck::ConfidenceCheck(core::f32 minConfidence)
    : _minConfidence{minConfidence} {}

SafetyVerdict ConfidenceCheck::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    if (state.confidence.toFloat() < _minConfidence)
    {
        return SafetyVerdict::Reject;
    }
    return SafetyVerdict::Pass;
}

const char* ConfidenceCheck::name() const noexcept
{
    return "ConfidenceCheck";
}

// ─────────────────────────────────────────────────────────────────────────────
// RateOfChangeCheck
// ─────────────────────────────────────────────────────────────────────────────

RateOfChangeCheck::RateOfChangeCheck(core::f32 maxDelta)
    : _maxDelta{maxDelta} {}

SafetyVerdict RateOfChangeCheck::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    if (!_hasPrevious)
    {
        for (core::usize i = 0; i < input::NeuralInputState::kChannels; ++i)
        {
            _previousChannels[i] = state.channels[i];
        }
        _hasPrevious = true;
        return SafetyVerdict::Pass;
    }

    SafetyVerdict result = SafetyVerdict::Pass;

    for (core::usize i = 0; i < input::NeuralInputState::kChannels; ++i)
    {
        const core::f32 delta = std::fabs(
            state.channels[i].toFloat() - _previousChannels[i].toFloat());
        if (delta > _maxDelta)
        {
            result = SafetyVerdict::Warn;
        }
        _previousChannels[i] = state.channels[i];
    }

    return result;
}

const char* RateOfChangeCheck::name() const noexcept
{
    return "RateOfChangeCheck";
}

} // namespace lpl::bci
