// /////////////////////////////////////////////////////////////////////////////
/// @file Prediction.cpp
/// @brief Client-side prediction buffer implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/netcode/Prediction.hpp>

#include <algorithm>

namespace lpl::net::netcode {

Prediction::Prediction(core::u32 maxPrediction)
    : maxPrediction_{maxPrediction}
{
    buffer_.reserve(maxPrediction);
}

Prediction::~Prediction() = default;

void Prediction::push(PredictedInput input)
{
    if (static_cast<core::u32>(buffer_.size()) >= maxPrediction_)
    {
        buffer_.erase(buffer_.begin());
    }
    buffer_.push_back(std::move(input));
}

void Prediction::acknowledge(core::u32 sequence)
{
    buffer_.erase(
        std::remove_if(buffer_.begin(), buffer_.end(),
                       [sequence](const PredictedInput& p) {
                           return p.sequence <= sequence;
                       }),
        buffer_.end());
}

std::vector<PredictedInput> Prediction::getUnacknowledged(core::u32 fromSequence) const
{
    std::vector<PredictedInput> result;
    for (const auto& p : buffer_)
    {
        if (p.sequence > fromSequence)
        {
            result.push_back(p);
        }
    }
    return result;
}

core::u32 Prediction::pendingCount() const noexcept
{
    return static_cast<core::u32>(buffer_.size());
}

} // namespace lpl::net::netcode
