/**
 * @file Statistics.cpp
 * @brief Implementation of statistical utilities.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#include "lpl/math/Statistics.hpp"

#include <cmath>

namespace lpl::math {

double Statistics::integratePsd(
    std::span<const double> psd,
    core::u32 binLow,
    core::u32 binHigh
) {
    double sum = 0.0;
    core::u32 hi = (binHigh <= psd.size()) ? binHigh : static_cast<core::u32>(psd.size());
    for (core::u32 i = binLow; i < hi; ++i)
        sum += psd[i];
    return sum;
}

core::u32 Statistics::hzToBin(double freqHz, double sampleRate, core::u32 fftSize)
{
    return static_cast<core::u32>(freqHz * fftSize / sampleRate);
}

void Statistics::slidingWindowRms(
    std::span<const double> signal,
    core::u32 windowSize,
    std::span<double> rms
) {
    if (signal.size() < windowSize)
        return;

    core::usize count = signal.size() - windowSize + 1;
    if (count > rms.size())
        count = rms.size();

    for (core::usize i = 0; i < count; ++i) {
        double sumSq = 0.0;
        for (core::u32 j = 0; j < windowSize; ++j)
            sumSq += signal[i + j] * signal[i + j];
        rms[i] = std::sqrt(sumSq / windowSize);
    }
}

void Statistics::computeBaseline(
    std::span<const double> samples,
    double &mean,
    double &stddev
) {
    if (samples.empty()) {
        mean   = 0.0;
        stddev = 0.0;
        return;
    }

    double sum = 0.0;
    for (auto s : samples)
        sum += s;
    mean = sum / static_cast<double>(samples.size());

    double varSum = 0.0;
    for (auto s : samples) {
        double d = s - mean;
        varSum += d * d;
    }
    stddev = std::sqrt(varSum / static_cast<double>(samples.size()));
}

} // namespace lpl::math
