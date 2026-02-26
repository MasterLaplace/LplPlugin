// /////////////////////////////////////////////////////////////////////////////
/// @file AuthoritativeStrategy.cpp
/// @brief Server-authoritative netcode implementation stub.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/netcode/AuthoritativeStrategy.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::net::netcode {

struct AuthoritativeStrategy::Impl
{
    core::u32 lastAckedSequence{0};
};

AuthoritativeStrategy::AuthoritativeStrategy()
    : impl_{std::make_unique<Impl>()}
{}

AuthoritativeStrategy::~AuthoritativeStrategy() = default;

core::Expected<void> AuthoritativeStrategy::onInputReceived(
    core::u32 /*playerId*/,
    std::span<const core::byte> /*inputData*/,
    core::u32 /*sequence*/)
{
    LPL_ASSERT(false && "AuthoritativeStrategy::onInputReceived not yet implemented");
    return {};
}

core::Expected<void> AuthoritativeStrategy::onStateReceived(
    std::span<const core::byte> /*snapshotData*/,
    core::u32 /*sequence*/)
{
    LPL_ASSERT(false && "AuthoritativeStrategy::onStateReceived not yet implemented");
    return {};
}

void AuthoritativeStrategy::tick(core::f32 /*dt*/)
{
    LPL_ASSERT(false && "AuthoritativeStrategy::tick not yet implemented");
}

const char* AuthoritativeStrategy::name() const noexcept
{
    return "AuthoritativeStrategy";
}

} // namespace lpl::net::netcode
