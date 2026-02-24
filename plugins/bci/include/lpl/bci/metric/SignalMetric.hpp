/**
 * @file SignalMetric.hpp
 * @brief Schumacher-style spectral ratio metric for EEG analysis.
 * @author MasterLaplace
 *
 * Computes the beta / (alpha + theta) power ratio per channel. This metric
 * captures cognitive engagement: a higher ratio indicates focused attention
 * while a lower ratio indicates drowsiness or relaxation.
 *
 * @see Statistics, BandExtractor
 */

#pragma once

#include "lpl/bci/core/Types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace lpl::bci::metric {

/**
 * @brief Configuration for the Schumacher metric computation.
 */
struct SignalMetricConfig {
    float sampleRate = 250.0f;
    std::size_t fftSize = 256;
    FrequencyBand theta = { .low = 4.0f, .high = 8.0f };
    FrequencyBand alpha = { .low = 8.0f, .high = 13.0f };
    FrequencyBand beta  = { .low = 13.0f, .high = 30.0f };
};

/**
 * @brief Per-channel Schumacher ratio and its component powers.
 */
struct SignalMetricResult {
    float thetaPower = 0.0f;
    float alphaPower = 0.0f;
    float betaPower  = 0.0f;
    float ratio      = 0.0f;
};

/**
 * @brief Computes the Schumacher engagement metric from PSD data.
 *
 * Stateless and reentrant. The PSD is assumed to have already been computed
 * by the DSP pipeline (FftProcessor → per-channel PSD).
 */
class SignalMetric {
public:
    /**
     * @brief Constructs the metric with the given configuration.
     *
     * @param config Frequency band definitions and FFT parameters
     */
    explicit SignalMetric(const SignalMetricConfig& config = {});

    /**
     * @brief Computes the Schumacher ratio for each channel.
     *
     * @param psd        Per-channel PSD arrays [channelCount][bins]
     * @return Per-channel metric results
     */
    [[nodiscard]] std::vector<SignalMetricResult> compute(
        std::span<const std::vector<float>> psd) const noexcept;

    /**
     * @brief Computes the mean Schumacher ratio across all channels.
     *
     * @param psd Per-channel PSD arrays
     * @return Average ratio, or 0 if no channels
     */
    [[nodiscard]] float computeMean(
        std::span<const std::vector<float>> psd) const noexcept;

private:
    std::size_t _thetaLow;
    std::size_t _thetaHigh;
    std::size_t _alphaLow;
    std::size_t _alphaHigh;
    std::size_t _betaLow;
    std::size_t _betaHigh;
};

} // namespace lpl::bci::metric
