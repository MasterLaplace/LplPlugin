/**
 * @file Statistics.hpp
 * @brief Basic statistical utilities for EEG signal analysis.
 * @author MasterLaplace
 *
 * Provides PSD integration, frequency-to-bin conversion, sliding RMS,
 * and baseline computation. These are the building blocks used by
 * higher-level metrics (Schumacher, Neural, Stability).
 *
 * @see SignalMetric, Calibration
 */

#pragma once

#include "lpl/bci/core/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lpl::bci::math {

/**
 * @brief Pure-function statistical utilities for spectral and temporal signals.
 */
class Statistics {
public:
    Statistics() = delete;

    /**
     * @brief Sums PSD values in the bin range [lowerBin, upperBin] (inclusive).
     *
     * Corresponds to the discrete integral of PSD(f) over the specified bins.
     *
     * @param psd       Power Spectral Density array (one value per bin)
     * @param lowerBin  First bin index (inclusive)
     * @param upperBin  Last bin index (inclusive)
     * @return Integrated power in the range
     */
    [[nodiscard]] static float integratePsd(
        std::span<const float> psd,
        std::size_t lowerBin,
        std::size_t upperBin) noexcept;

    /**
     * @brief Converts a frequency in Hz to the corresponding FFT bin index.
     *
     * @param hz         Frequency in Hz
     * @param sampleRate Sampling rate in Hz
     * @param fftSize    Number of FFT points
     * @return Bin index (truncated)
     */
    [[nodiscard]] static std::size_t hzToBin(
        float hz,
        float sampleRate,
        std::size_t fftSize) noexcept;

    /**
     * @brief Computes the RMS over the last windowSize elements of the data.
     *
     * @param data       Input signal
     * @param windowSize Number of trailing elements to consider
     * @return RMS value, or 0 if insufficient data
     */
    [[nodiscard]] static float slidingWindowRms(
        std::span<const float> data,
        std::size_t windowSize) noexcept;

    /**
     * @brief Computes the mean and population standard deviation.
     *
     * @param data Input values
     * @return Baseline with mean and stdDev fields
     */
    [[nodiscard]] static Baseline computeBaseline(
        std::span<const float> data) noexcept;
};

} // namespace lpl::bci::math
