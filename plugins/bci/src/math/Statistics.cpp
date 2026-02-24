/**
 * @file Statistics.cpp
 * @brief Implementation of basic statistical utilities.
 */

#include "lpl/bci/math/Statistics.hpp"

#include <cmath>
#include <numeric>

namespace bci::math {

float Statistics::integratePsd(
    std::span<const float> psd,
    std::size_t lowerBin,
    std::size_t upperBin) noexcept
{
    if (psd.empty() || lowerBin > upperBin || upperBin >= psd.size())
        return 0.0f;

    return std::accumulate(
        psd.begin() + static_cast<std::ptrdiff_t>(lowerBin),
        psd.begin() + static_cast<std::ptrdiff_t>(upperBin) + 1,
        0.0f);
}

std::size_t Statistics::hzToBin(
    float hz,
    float sampleRate,
    std::size_t fftSize) noexcept
{
    if (sampleRate <= 0.0f || fftSize == 0)
        return 0;

    return static_cast<std::size_t>(hz * static_cast<float>(fftSize) / sampleRate);
}

float Statistics::slidingWindowRms(
    std::span<const float> data,
    std::size_t windowSize) noexcept
{
    if (data.empty() || windowSize == 0)
        return 0.0f;

    const std::size_t count = std::min(windowSize, data.size());
    const auto start = data.end() - static_cast<std::ptrdiff_t>(count);

    float sumSq = 0.0f;
    for (auto it = start; it != data.end(); ++it)
        sumSq += (*it) * (*it);

    return std::sqrt(sumSq / static_cast<float>(count));
}

Baseline Statistics::computeBaseline(
    std::span<const float> data) noexcept
{
    if (data.empty())
        return { .mean = 0.0f, .stdDev = 0.0f };

    const auto n = static_cast<float>(data.size());

    const float mean = std::accumulate(data.begin(), data.end(), 0.0f) / n;

    float varianceSum = 0.0f;
    for (const float x : data) {
        const float diff = x - mean;
        varianceSum += diff * diff;
    }

    return {
        .mean   = mean,
        .stdDev = std::sqrt(varianceSum / n),
    };
}

} // namespace bci::math
