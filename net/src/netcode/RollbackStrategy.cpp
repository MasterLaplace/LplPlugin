/**
 * @file RollbackStrategy.cpp
 * @brief Rollback netcode implementation stub.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

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
    : _impl{std::make_unique<Impl>(maxRollbackFrames)}
{}

RollbackStrategy::~RollbackStrategy() = default;

core::Expected<void> RollbackStrategy::onInputReceived(
    core::u32 playerId,
    std::span<const core::byte> inputData,
    core::u32 sequence)
{
    (void)playerId;
    (void)inputData;
    if (sequence < _impl->currentFrame && (_impl->currentFrame - sequence) <= _impl->maxRollbackFrames)
    {
        core::Log::debug("RollbackStrategy", "late input, rolling back");
    }
    return {};
}

core::Expected<void> RollbackStrategy::onStateReceived(
    std::span<const core::byte> snapshotData,
    core::u32 sequence)
{
    (void)snapshotData;
    (void)sequence;
    core::Log::debug("RollbackStrategy", "received authoritative state");
    return {};
}

void RollbackStrategy::tick(core::f32 dt)
{
    _impl->currentFrame++;
    (void)dt;
}

const char* RollbackStrategy::name() const noexcept
{
    return "RollbackStrategy";
}

} // namespace lpl::net::netcode
