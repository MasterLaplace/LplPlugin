/**
 * @file LslOutlet.cpp
 * @brief Implementation of the LSL outlet wrapper.
 */

#include "lpl/bci/stream/LslOutlet.hpp"

#include <lsl_cpp.h>
#include <vector>

namespace bci::stream {

struct LslOutlet::Impl {
    std::unique_ptr<lsl::stream_outlet> outlet;
    std::size_t channelCount = 0;
};

ExpectedVoid LslOutlet::open(const LslOutletConfig& config)
{
    if (_impl && _impl->outlet)
        return std::unexpected(Error{
            ErrorCode::kInvalidState,
            "LSL outlet already open"});

    lsl::stream_info info(
        config.streamName,
        config.streamType,
        static_cast<int>(config.channelCount),
        static_cast<double>(config.sampleRate),
        lsl::cf_float32,
        config.streamName + "_uid");

    _impl = std::make_unique<Impl>();
    _impl->channelCount = config.channelCount;
    _impl->outlet = std::make_unique<lsl::stream_outlet>(info);

    return {};
}

void LslOutlet::pushSample(std::span<const float> data) noexcept
{
    if (!_impl || !_impl->outlet)
        return;

    _impl->outlet->push_sample(data.data());
}

void LslOutlet::pushNeuralState(const NeuralState& state) noexcept
{
    if (!_impl || !_impl->outlet)
        return;

    const std::size_t alphaCount = state.channelAlpha.size();
    const std::size_t betaCount  = state.channelBeta.size();

    std::vector<float> buffer;
    buffer.reserve(alphaCount + betaCount + 2);

    buffer.insert(buffer.end(), state.channelAlpha.begin(), state.channelAlpha.end());
    buffer.insert(buffer.end(), state.channelBeta.begin(), state.channelBeta.end());
    buffer.push_back(state.alphaPower);
    buffer.push_back(state.betaPower);

    _impl->outlet->push_sample(buffer.data());
}

bool LslOutlet::isOpen() const noexcept
{
    return _impl && _impl->outlet != nullptr;
}

void LslOutlet::close() noexcept
{
    _impl.reset();
}

} // namespace bci::stream
