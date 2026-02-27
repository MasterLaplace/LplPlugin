/**
 * @file NeuralSafetyChain.hpp
 * @brief Chain of Responsibility for neural signal validation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_BCI_NEURALSAFETYCHAIN_HPP
    #define LPL_BCI_NEURALSAFETYCHAIN_HPP

#include <lpl/input/NeuralInputState.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <memory>
#include <vector>

namespace lpl::bci {

/** @brief Result of a safety check. */
enum class SafetyVerdict : core::u8
{
    Pass,
    Warn,
    Reject
};

/** @brief Abstract safety check link. */
class ISafetyCheck
{
public:
    virtual ~ISafetyCheck() = default;

    /**
     * @brief Evaluate the neural input against this check.
     * @param state The neural input to validate.
     * @return Verdict (Pass / Warn / Reject).
     */
    [[nodiscard]] virtual SafetyVerdict evaluate(
        const input::NeuralInputState& state) const noexcept = 0;

    /** @brief Human-readable name of the check. */
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

/**
 * @brief Chain of Responsibility aggregating multiple ISafetyCheck links.
 *
 * Evaluates each check in order. Returns Reject on first rejection,
 * Warn if any check warns, Pass if all pass.
 */
class NeuralSafetyChain
{
public:
    NeuralSafetyChain();
    ~NeuralSafetyChain();

    NeuralSafetyChain(const NeuralSafetyChain&) = delete;
    NeuralSafetyChain& operator=(const NeuralSafetyChain&) = delete;

    /** @brief Append a check to the chain. */
    void addCheck(std::unique_ptr<ISafetyCheck> check);

    /**
     * @brief Run the full chain on the given state.
     * @param state Neural input to validate.
     * @return Final aggregated verdict.
     */
    [[nodiscard]] SafetyVerdict evaluate(
        const input::NeuralInputState& state) const noexcept;

    /** @brief Number of checks in the chain. */
    [[nodiscard]] core::usize checkCount() const noexcept;

private:
    std::vector<std::unique_ptr<ISafetyCheck>> _checks;
};

// ─────────────────────────────────────────────────────────────────────────────
// Built-in safety checks
// ─────────────────────────────────────────────────────────────────────────────

/** @brief Rejects signals where any channel exceeds an amplitude bound. */
class AmplitudeBoundsCheck final : public ISafetyCheck
{
public:
    explicit AmplitudeBoundsCheck(core::f32 maxAbsValue = 500.0f);
    [[nodiscard]] SafetyVerdict evaluate(
        const input::NeuralInputState& state) const noexcept override;
    [[nodiscard]] const char* name() const noexcept override;

private:
    core::f32 _maxAbsValue;
};

/** @brief Rejects signals below a confidence threshold. */
class ConfidenceCheck final : public ISafetyCheck
{
public:
    explicit ConfidenceCheck(core::f32 minConfidence = 0.5f);
    [[nodiscard]] SafetyVerdict evaluate(
        const input::NeuralInputState& state) const noexcept override;
    [[nodiscard]] const char* name() const noexcept override;

private:
    core::f32 _minConfidence;
};

/** @brief Warns if the rate-of-change between ticks exceeds a threshold. */
class RateOfChangeCheck final : public ISafetyCheck
{
public:
    explicit RateOfChangeCheck(core::f32 maxDelta = 100.0f);
    [[nodiscard]] SafetyVerdict evaluate(
        const input::NeuralInputState& state) const noexcept override;
    [[nodiscard]] const char* name() const noexcept override;

private:
    core::f32 _maxDelta;
    mutable math::Fixed32 _previousChannels[input::NeuralInputState::kChannels]{};
    mutable bool _hasPrevious{false};
};

} // namespace lpl::bci

#endif // LPL_BCI_NEURALSAFETYCHAIN_HPP
