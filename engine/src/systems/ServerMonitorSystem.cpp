/**
 * @file ServerMonitorSystem.cpp
 * @brief Periodic performance logging for server diagnostics.
 *
 * Logs entity count, cell count, and connected client count at a
 * configurable interval (default: every 300 ticks ≈ 5 s @ 60 Hz).
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/core/Log.hpp>
#include <lpl/engine/systems/ServerMonitorSystem.hpp>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::SystemDescriptor kMonitorDesc{"ServerMonitor", ecs::SchedulePhase::Network, {}};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct ServerMonitorSystem::Impl {
    net::session::SessionManager &sessionManager;
    ecs::WorldPartition &world;
    core::u32 logInterval;
    core::u32 frameCounter{0};

    Impl(net::session::SessionManager &sm, ecs::WorldPartition &w, core::u32 interval)
        : sessionManager{sm}, world{w}, logInterval{interval}
    {
    }
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

ServerMonitorSystem::ServerMonitorSystem(net::session::SessionManager &sessionManager, ecs::WorldPartition &world,
                                         core::u32 logInterval)
    : _impl{std::make_unique<Impl>(sessionManager, world, logInterval)}
{
}

ServerMonitorSystem::~ServerMonitorSystem() = default;

const ecs::SystemDescriptor &ServerMonitorSystem::descriptor() const noexcept { return kMonitorDesc; }

void ServerMonitorSystem::execute(core::f32 /*dt*/)
{
    ++_impl->frameCounter;

    if (_impl->frameCounter < _impl->logInterval)
        return;

    _impl->frameCounter = 0;

    [[maybe_unused]] const core::u32 cells = _impl->world.cellCount();
    [[maybe_unused]] const core::u32 clients = _impl->sessionManager.activeCount();

    // GC empty cells periodically
    [[maybe_unused]] const core::u32 gcCount = _impl->world.gcEmptyCells();

    core::Log::info("Monitor", "Server heartbeat — periodic GC and diagnostics");
}

} // namespace lpl::engine::systems
