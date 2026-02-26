// /////////////////////////////////////////////////////////////////////////////
/// @file NeuralSafetyChain.cpp
/// @brief NeuralSafetyChain and built-in checks implementation.
// /////////////////////////////////////////////////////////////////////////////

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
    checks_.push_back(std::move(check));
}

SafetyVerdict NeuralSafetyChain::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    SafetyVerdict worst = SafetyVerdict::Pass;

    for (const auto& check : checks_)
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
    return checks_.size();
}

// ─────────────────────────────────────────────────────────────────────────────
// AmplitudeBoundsCheck
// ─────────────────────────────────────────────────────────────────────────────

AmplitudeBoundsCheck::AmplitudeBoundsCheck(core::f32 maxAbsValue)
    : maxAbsValue_{maxAbsValue} {}

SafetyVerdict AmplitudeBoundsCheck::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    for (core::usize i = 0; i < input::NeuralInputState::kChannels; ++i)
    {
        const core::f32 val = state.channels[i].toFloat();
        if (std::fabs(val) > maxAbsValue_)
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
    : minConfidence_{minConfidence} {}

SafetyVerdict ConfidenceCheck::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    if (state.confidence.toFloat() < minConfidence_)
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
    : maxDelta_{maxDelta} {}

SafetyVerdict RateOfChangeCheck::evaluate(
    const input::NeuralInputState& state) const noexcept
{
    if (!hasPrevious_)
    {
        for (core::usize i = 0; i < input::NeuralInputState::kChannels; ++i)
        {
            previousChannels_[i] = state.channels[i];
        }
        hasPrevious_ = true;
        return SafetyVerdict::Pass;
    }

    SafetyVerdict result = SafetyVerdict::Pass;

    for (core::usize i = 0; i < input::NeuralInputState::kChannels; ++i)
    {
        const core::f32 delta = std::fabs(
            state.channels[i].toFloat() - previousChannels_[i].toFloat());
        if (delta > maxDelta_)
        {
            result = SafetyVerdict::Warn;
        }
        previousChannels_[i] = state.channels[i];
    }

    return result;
}

const char* RateOfChangeCheck::name() const noexcept
{
    return "RateOfChangeCheck";
}

} // namespace lpl::bci
