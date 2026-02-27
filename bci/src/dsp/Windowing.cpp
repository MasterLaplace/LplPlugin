/**
 * @file Windowing.cpp
 * @brief Implementation of the Hann window DSP stage.
 * @author MasterLaplace
 */

#include "lpl/bci/dsp/Windowing.hpp"

#include <cmath>
#include <numbers>

namespace lpl::bci::dsp {

HannWindow::HannWindow(std::size_t windowSize)
    : _windowSize(windowSize), _coefficients(windowSize)
{
    const auto nMinus1 = static_cast<float>(windowSize - 1);
    for (std::size_t i = 0; i < windowSize; ++i) {
        _coefficients[i] = 0.5f * (1.0f - std::cos(
            2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / nMinus1));
    }
}

Expected<SignalBlock> HannWindow::process(const SignalBlock &input)
{
    if (input.empty()) {
        return std::unexpected(
            Error::make(ErrorCode::kEmptyInput, "HannWindow received empty block"));
    }

    if (input.sampleCount() != _windowSize) {
        return std::unexpected(
            Error::make(ErrorCode::kFftSizeMismatch,
                "HannWindow expects " + std::to_string(_windowSize) +
                " samples, got " + std::to_string(input.sampleCount())));
    }

    SignalBlock output{
        .data = input.data,
        .sampleRate = input.sampleRate,
        .channelCount = input.channelCount,
        .timestamp = input.timestamp
    };

    for (std::size_t t = 0; t < _windowSize; ++t) {
        for (std::size_t ch = 0; ch < output.channelCount; ++ch) {
            output.data[t][ch] *= _coefficients[t];
        }
    }

    return output;
}

std::string_view HannWindow::name() const noexcept
{
    return "HannWindow";
}

} // namespace lpl::bci::dsp
