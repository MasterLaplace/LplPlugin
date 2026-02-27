/**
 * @file BciAdapter.hpp
 * @brief Adapter translating raw EEG into NeuralInputState (Adapter pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_BCI_BCIADAPTER_HPP
    #define LPL_BCI_BCIADAPTER_HPP

#include <lpl/bci/IBciDriver.hpp>
#include <lpl/input/NeuralInputState.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <memory>

namespace lpl::bci {

/** @brief Configuration for the BCI adapter DSP pipeline. */
struct BciAdapterConfig
{
    core::f32 sampleRateHz{256.0f};
    core::f32 lowCutHz{1.0f};
    core::f32 highCutHz{50.0f};
    core::u16 fftSize{256};
    core::f32 confidenceThreshold{0.6f};
};

/**
 * @brief Adapter that converts raw EEG samples into engine-usable
 *        NeuralInputState values.
 *
 * Adapter pattern: wraps an IBciDriver and exposes a high-level
 * update() method that the InputManager can call each tick.
 */
class BciAdapter
{
public:
    /// @param driver Owned BCI driver instance.
    /// @param config DSP pipeline configuration.
    explicit BciAdapter(std::unique_ptr<IBciDriver> driver,
                        const BciAdapterConfig& config = {});
    ~BciAdapter();

    BciAdapter(const BciAdapter&) = delete;
    BciAdapter& operator=(const BciAdapter&) = delete;

    /** @brief Connect and start streaming from the BCI device. */
    [[nodiscard]] core::Expected<void> start();

    /** @brief Stop streaming and disconnect. */
    void stop();

    /**
     * @brief Poll driver, run DSP, produce a NeuralInputState.
     *
     * Called once per engine tick. Returns the latest processed
     * neural input or an error if no data is available.
     *
     * If calibration is active, this also feeds band powers to the
     * Calibration state machine. Once calibrated, NeuralMetric
     * normalization is applied (μ ± kσ clamping).
     */
    [[nodiscard]] core::Expected<input::NeuralInputState> update();

    /**
     * @brief Starts the calibration process.
     *
     * During calibration, `update()` will feed alpha/beta band powers
     * to the Calibration module. Once enough trials are collected
     * (default: 30), it auto-transitions to Ready and applies
     * NeuralMetric normalization going forward.
     */
    [[nodiscard]] core::Expected<void> startCalibration();

    /** @brief Returns true if calibration is complete and baselines are available. */
    [[nodiscard]] bool isCalibrated() const noexcept;

    /** @brief Access the underlying driver (read-only). */
    [[nodiscard]] const IBciDriver& driver() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::bci

#endif // LPL_BCI_BCIADAPTER_HPP
