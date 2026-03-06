/**
 * @file AuthoritativeStrategy.cpp
 * @brief Server-authoritative netcode: stores per-player inputs,
 *        applies them each tick. The server is the single source of truth.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/netcode/AuthoritativeStrategy.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace lpl::net::netcode {

// ========================================================================== //
//  Per-player input entry                                                     //
// ========================================================================== //

struct InputEntry {
    core::u32 sequence;
    std::vector<core::byte> data;
};

// ========================================================================== //
//  Impl                                                                       //
// ========================================================================== //

struct AuthoritativeStrategy::Impl {
    core::u32 serverFrame{0};
    core::u32 lastAckedSequence{0};

    /** @brief Per-player pending input queue, consumed each tick. */
    std::unordered_map<core::u32, std::vector<InputEntry>> pendingInputs;

    /** @brief Consumed inputs from the last tick (available for systems). */
    std::unordered_map<core::u32, std::vector<InputEntry>> consumedInputs;
};

// ========================================================================== //
//  Public                                                                     //
// ========================================================================== //

AuthoritativeStrategy::AuthoritativeStrategy() : _impl{std::make_unique<Impl>()} {}

AuthoritativeStrategy::~AuthoritativeStrategy() = default;

core::Expected<void> AuthoritativeStrategy::onInputReceived(core::u32 playerId, std::span<const core::byte> inputData,
                                                            core::u32 sequence)
{
    // Reject stale inputs (already processed)
    if (sequence <= _impl->lastAckedSequence && _impl->lastAckedSequence > 0)
    {
        return {};
    }

    // Store input for consumption at next tick
    InputEntry entry;
    entry.sequence = sequence;
    entry.data.assign(inputData.begin(), inputData.end());

    _impl->pendingInputs[playerId].push_back(std::move(entry));

    // Sort by sequence to process in order
    auto &queue = _impl->pendingInputs[playerId];
    std::sort(queue.begin(), queue.end(),
              [](const InputEntry &a, const InputEntry &b) { return a.sequence < b.sequence; });

    return {};
}

core::Expected<void> AuthoritativeStrategy::onStateReceived(std::span<const core::byte> snapshotData,
                                                            core::u32 sequence)
{
    // Server-side: no-op (we ARE the authority).
    // In a distributed server mesh this would propagate state between nodes.
    (void) snapshotData;
    (void) sequence;
    return {};
}

void AuthoritativeStrategy::tick(core::f32 dt)
{
    (void) dt;

    // Consume all pending inputs → move to consumedInputs for system access
    _impl->consumedInputs.clear();
    _impl->consumedInputs.swap(_impl->pendingInputs);

    // Update acked sequence to current frame
    _impl->serverFrame++;
    _impl->lastAckedSequence = _impl->serverFrame;
}

const char *AuthoritativeStrategy::name() const noexcept { return "AuthoritativeStrategy"; }

} // namespace lpl::net::netcode
