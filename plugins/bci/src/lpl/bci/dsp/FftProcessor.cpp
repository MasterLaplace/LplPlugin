/**
 * @file FftProcessor.cpp
 * @brief Implementation of the Cooley-Tukey radix-2 FFT processor.
 * @author MasterLaplace
 */

#include "lpl/bci/dsp/FftProcessor.hpp"

#include <bit>
#include <cmath>
#include <numbers>

namespace lpl::bci::dsp {

FftProcessor::FftProcessor(std::size_t fftSize)
    : _fftSize(fftSize)
    , _halfSize(fftSize / 2)
    , _normFactor(2.0f / static_cast<float>(fftSize))
{
    if (!std::has_single_bit(fftSize)) {
        _fftSize = std::bit_ceil(fftSize);
        _halfSize = _fftSize / 2;
        _normFactor = 2.0f / static_cast<float>(_fftSize);
    }
}

void FftProcessor::bitReversalPermutation(std::vector<Complex> &x)
{
    const auto N = static_cast<std::uint32_t>(x.size());
    std::uint32_t j = 0;

    for (std::uint32_t i = 1; i < N; ++i) {
        std::uint32_t bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
}

void FftProcessor::butterflyPass(std::vector<Complex> &x)
{
    const auto N = static_cast<std::uint32_t>(x.size());

    for (std::uint32_t len = 2; len <= N; len <<= 1) {
        const float angle = -2.0f * std::numbers::pi_v<float> / static_cast<float>(len);
        const Complex wlen(std::cos(angle), std::sin(angle));

        for (std::uint32_t i = 0; i < N; i += len) {
            Complex w(1.0f, 0.0f);
            const std::uint32_t halfLen = len / 2;

            for (std::uint32_t k = 0; k < halfLen; ++k) {
                Complex u = x[i + k];
                Complex v = x[i + k + halfLen] * w;
                x[i + k] = u + v;
                x[i + k + halfLen] = u - v;
                w *= wlen;
            }
        }
    }
}

Expected<SignalBlock> FftProcessor::process(const SignalBlock &input)
{
    if (input.empty()) {
        return std::unexpected(
            Error::make(ErrorCode::kEmptyInput, "FftProcessor received empty block"));
    }

    if (input.sampleCount() != _fftSize) {
        return std::unexpected(
            Error::make(ErrorCode::kFftSizeMismatch,
                "FftProcessor expects " + std::to_string(_fftSize) +
                " samples, got " + std::to_string(input.sampleCount())));
    }

    SignalBlock output{
        .data = {},
        .sampleRate = input.sampleRate,
        .channelCount = input.channelCount,
        .timestamp = input.timestamp
    };
    output.data.resize(_halfSize, std::vector<float>(input.channelCount, 0.0f));

    std::vector<Complex> buffer(_fftSize);

    for (std::size_t ch = 0; ch < input.channelCount; ++ch) {
        for (std::size_t t = 0; t < _fftSize; ++t) {
            buffer[t] = Complex(input.data[t][ch], 0.0f);
        }

        bitReversalPermutation(buffer);
        butterflyPass(buffer);

        for (std::size_t i = 0; i < _halfSize; ++i) {
            output.data[i][ch] = std::abs(buffer[i]) * _normFactor;
        }
    }

    return output;
}

std::string_view FftProcessor::name() const noexcept
{
    return "FftProcessor";
}

} // namespace lpl::bci::dsp
