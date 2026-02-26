// /////////////////////////////////////////////////////////////////////////////
/// @file RollbackStrategy.cpp
/// @brief Rollback netcode implementation stub.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/netcode/RollbackStrategy.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <vector>

namespace lpl::net::netcode {

struct RollbackStrategy::Impl
{
    core::u32                             maxRollbackFrames;
    core::u32                             currentFrame{0};
    std::vector<std::vector<core::byte>>  stateHistory;

    explicit Impl(core::u32 maxFrames)
        : maxRollbackFrames{maxFrames}
    {
        stateHistory.resize(maxFrames);
    }
};

RollbackStrategy::RollbackStrategy(core::u32 maxRollbackFrames)
    : impl_{std::make_unique<Impl>(maxRollbackFrames)}
{}

RollbackStrategy::~RollbackStrategy() = default;

core::Expected<void> RollbackStrategy::onInputReceived(
    core::u32 /*playerId*/,
    std::span<const core::byte> /*inputData*/,
    core::u32 /*sequence*/)
{
    LPL_ASSERT(false && "RollbackStrategy::onInputReceived not yet implemented");
    return {};
}

core::Expected<void> RollbackStrategy::onStateReceived(
    std::span<const core::byte> /*snapshotData*/,
    core::u32 /*sequence*/)
{
    LPL_ASSERT(false && "RollbackStrategy::onStateReceived not yet implemented");
    return {};
}

void RollbackStrategy::tick(core::f32 /*dt*/)
{
    LPL_ASSERT(false && "RollbackStrategy::tick not yet implemented");
}

const char* RollbackStrategy::name() const noexcept
{
    return "RollbackStrategy";
}

} // namespace lpl::net::netcode
