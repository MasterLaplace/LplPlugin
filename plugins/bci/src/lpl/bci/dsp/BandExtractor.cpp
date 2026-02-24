/**
 * @file BandExtractor.cpp
 * @brief Implementation of the frequency band power extractor.
 * @author MasterLaplace
 */

#include "lpl/bci/dsp/BandExtractor.hpp"

#include <algorithm>
#include <cmath>

namespace lpl::bci::dsp {

BandExtractor::BandExtractor(
    std::vector<FrequencyBand> bands,
    float sampleRate,
    std::size_t fftSize)
    : _bands(std::move(bands))
    , _sampleRate(sampleRate)
    , _fftSize(fftSize)
    , _halfSize(fftSize / 2)
{
    const float freqRes = sampleRate / static_cast<float>(fftSize);

    _binRanges.reserve(_bands.size());
    for (const auto &band : _bands) {
        auto lowBin = static_cast<std::size_t>(
            std::floor(band.low / freqRes));
        auto highBin = static_cast<std::size_t>(
            std::ceil(band.high / freqRes));
        highBin = std::min(highBin, _halfSize - 1);
        _binRanges.push_back({lowBin, highBin});
    }
}

Expected<SignalBlock> BandExtractor::process(const SignalBlock &input)
{
    if (input.empty()) {
        return std::unexpected(
            Error::make(ErrorCode::kEmptyInput, "BandExtractor received empty block"));
    }

    SignalBlock output{
        .data = {},
        .sampleRate = input.sampleRate,
        .channelCount = input.channelCount,
        .timestamp = input.timestamp
    };

    const std::size_t bandCount = _bands.size();
    output.data.resize(bandCount, std::vector<float>(input.channelCount, 0.0f));

    for (std::size_t b = 0; b < bandCount; ++b) {
        const auto &range = _binRanges[b];

        for (std::size_t ch = 0; ch < input.channelCount; ++ch) {
            float power = 0.0f;
            const std::size_t maxBin = std::min(range.high + 1, input.sampleCount());
            for (std::size_t bin = range.low; bin < maxBin; ++bin) {
                power += input.data[bin][ch];
            }
            output.data[b][ch] = power;
        }
    }

    return output;
}

std::string_view BandExtractor::name() const noexcept
{
    return "BandExtractor";
}

} // namespace lpl::bci::dsp
