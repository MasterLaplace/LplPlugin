/**
 * @file AuthoritativeStrategy.cpp
 * @brief Server-authoritative netcode implementation stub.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/netcode/AuthoritativeStrategy.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::net::netcode {

struct AuthoritativeStrategy::Impl
{
    core::u32 lastAckedSequence{0};
};

AuthoritativeStrategy::AuthoritativeStrategy()
    : _impl{std::make_unique<Impl>()}
{}

AuthoritativeStrategy::~AuthoritativeStrategy() = default;

core::Expected<void> AuthoritativeStrategy::onInputReceived(
    core::u32 playerId,
    std::span<const core::byte> inputData,
    core::u32 sequence)
{
    (void)playerId;
    (void)inputData;
    // Basic authoritative logic: validate sequence, store input for next tick.
    if (sequence > _impl->lastAckedSequence)
    {
        _impl->lastAckedSequence = sequence;
    }
    
    core::Log::debug("AuthoritativeStrategy", "received input");
    return {};
}

core::Expected<void> AuthoritativeStrategy::onStateReceived(
    std::span<const core::byte> snapshotData,
    core::u32 sequence)
{
    (void)snapshotData;
    // On client: receive authoritative state, reconcile local prediction.
    if (sequence > _impl->lastAckedSequence)
    {
        _impl->lastAckedSequence = sequence;
    }
    
    core::Log::debug("AuthoritativeStrategy", "received state snapshot");
    return {};
}

void AuthoritativeStrategy::tick(core::f32 dt)
{
    // Server: apply inputs, run physics, generate new state.
    // Client: run local prediction, reconcile if new state arrived.
    (void)dt;
}

const char* AuthoritativeStrategy::name() const noexcept
{
    return "AuthoritativeStrategy";
}

} // namespace lpl::net::netcode
