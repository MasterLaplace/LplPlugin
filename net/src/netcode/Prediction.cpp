/**
 * @file Prediction.cpp
 * @brief Client-side prediction buffer implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/netcode/Prediction.hpp>

#include <algorithm>

namespace lpl::net::netcode {

Prediction::Prediction(core::u32 maxPrediction)
    : _maxPrediction{maxPrediction}
{
    _buffer.reserve(maxPrediction);
}

Prediction::~Prediction() = default;

void Prediction::push(PredictedInput input)
{
    if (static_cast<core::u32>(_buffer.size()) >= _maxPrediction)
    {
        _buffer.erase(_buffer.begin());
    }
    _buffer.push_back(std::move(input));
}

void Prediction::acknowledge(core::u32 sequence)
{
    _buffer.erase(
        std::remove_if(_buffer.begin(), _buffer.end(),
                       [sequence](const PredictedInput& p) {
                           return p.sequence <= sequence;
                       }),
        _buffer.end());
}

std::vector<PredictedInput> Prediction::getUnacknowledged(core::u32 fromSequence) const
{
    std::vector<PredictedInput> result;
    for (const auto& p : _buffer)
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
    return static_cast<core::u32>(_buffer.size());
}

} // namespace lpl::net::netcode
