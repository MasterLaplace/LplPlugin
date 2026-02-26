/**
 * @file Statistics.hpp
 * @brief Statistical utilities for signal processing and BCI metrics.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_STATISTICS_HPP
    #define LPL_MATH_STATISTICS_HPP

    #include <lpl/core/Types.hpp>

    #include <span>

namespace lpl::math {

/**
 * @brief Utility functions for PSD integration, RMS, and baseline
 *        computation used by BCI metric pipelines.
 */
class Statistics final {
public:
    Statistics() = delete;

    /**
     * @brief Integrate a PSD array between two frequency bins.
     * @param psd     Power spectral density array.
     * @param binLow  Lower frequency bin (inclusive).
     * @param binHigh Upper frequency bin (exclusive).
     * @return Integrated power.
     */
    [[nodiscard]] static double integratePsd(
        std::span<const double> psd,
        core::u32 binLow,
        core::u32 binHigh
    );

    /**
     * @brief Convert a frequency in Hz to an FFT bin index.
     * @param freqHz     Target frequency.
     * @param sampleRate Sampling rate of the signal.
     * @param fftSize    Number of FFT points.
     * @return Bin index.
     */
    [[nodiscard]] static core::u32 hzToBin(
        double freqHz,
        double sampleRate,
        core::u32 fftSize
    );

    /**
     * @brief Compute a sliding-window RMS over a signal.
     * @param signal     Input samples.
     * @param windowSize Number of samples per window.
     * @param[out] rms   Output RMS values.
     */
    static void slidingWindowRms(
        std::span<const double> signal,
        core::u32 windowSize,
        std::span<double> rms
    );

    /**
     * @brief Compute baseline mean and standard deviation.
     * @param samples   Calibration samples.
     * @param[out] mean Output mean.
     * @param[out] stddev Output standard deviation.
     */
    static void computeBaseline(
        std::span<const double> samples,
        double &mean,
        double &stddev
    );
};

} // namespace lpl::math

#endif // LPL_MATH_STATISTICS_HPP
