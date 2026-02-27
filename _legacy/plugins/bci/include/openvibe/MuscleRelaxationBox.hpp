/**
 * @file MuscleRelaxationBox.hpp
 * @brief OpenViBE-compatible muscle relaxation monitor.
 * @author MasterLaplace
 *
 * Implements Schumacher et al. (2015) metric: average power in the gamma band
 * (40-70 Hz) across EEG channels as an indicator of muscular artifact activity.
 * High R(t) indicates the user is tensing muscles instead of performing pure
 * motor imagery.
 *
 * This processor is usable both as a standalone component and as the core
 * algorithm inside an OpenViBE CBoxAlgorithm wrapper.
 *
 * OpenViBE integration:
 *   - Input 0  : Signal (EEG channels)
 *   - Output 0 : Streamed Matrix (1x1 — R(t) scalar)
 *   - Setting 0: Lower frequency bound (Hz, default: 40)
 *   - Setting 1: Upper frequency bound (Hz, default: 70)
 *   - Setting 2: Alert threshold (default: 0.5)
 *
 * @see SignalMetric
 */

#pragma once

#include "core/Types.hpp"
#include "metric/SignalMetric.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace bci::openvibe {

/**
 * @brief Configuration for the muscle relaxation processor.
 */
struct MuscleRelaxationConfig {
    float lowerFreqHz    = 40.0f;
    float upperFreqHz    = 70.0f;
    float alertThreshold = 0.5f;
    float sampleRate     = 250.0f;
    std::size_t fftSize  = 256;
};

/**
 * @brief Result of a muscle relaxation computation.
 */
struct MuscleRelaxationResult {
    float gammaRatio = 0.0f;
    bool isAlert     = false;
};

/**
 * @brief Gamma-band spectral ratio processor for muscular artifact detection.
 *
 * Standalone processor that computes the Schumacher-style ratio in the gamma
 * band. Can be wrapped by an OpenViBE CBoxAlgorithm for real-time processing.
 */
class MuscleRelaxationBox {
public:
    /**
     * @brief Constructs the processor with the given configuration.
     *
     * @param config Frequency band and threshold parameters
     */
    explicit MuscleRelaxationBox(const MuscleRelaxationConfig& config = {});

    /**
     * @brief Computes R(t) from per-channel PSD data.
     *
     * R(t) = mean gamma power across channels, normalized by total band power.
     *
     * @param psd Per-channel PSD arrays [channelCount][bins]
     * @return Muscle relaxation result with ratio and alert flag
     */
    [[nodiscard]] MuscleRelaxationResult compute(
        std::span<const std::vector<float>> psd) const noexcept;

    /**
     * @brief Returns the current configuration.
     */
    [[nodiscard]] const MuscleRelaxationConfig& config() const noexcept;

private:
    MuscleRelaxationConfig _config;
    std::size_t _gammaLow;
    std::size_t _gammaHigh;
};

} // namespace bci::openvibe
