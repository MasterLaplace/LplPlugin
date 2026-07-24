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

// Host-only system (it is gated behind LPL_HAS_NET, which no kernel build
// defines), so the host C library is available to format the counters.
#include <cstdio>

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
    core::u32 instanceId;
    core::u32 frameCounter{0};
    core::u64 totalFrames{0};

    Impl(net::session::SessionManager &sm, ecs::WorldPartition &w, core::u32 interval, core::u32 instance)
        : sessionManager{sm}, world{w}, logInterval{interval}, instanceId{instance}
    {
    }
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

ServerMonitorSystem::ServerMonitorSystem(net::session::SessionManager &sessionManager, ecs::WorldPartition &world,
                                         core::u32 logInterval, core::u32 instanceId)
    : _impl{std::make_unique<Impl>(sessionManager, world, logInterval, instanceId)}
{
}

ServerMonitorSystem::~ServerMonitorSystem() = default;

const ecs::SystemDescriptor &ServerMonitorSystem::descriptor() const noexcept { return kMonitorDesc; }

void ServerMonitorSystem::execute(core::f32 /*dt*/)
{
    ++_impl->frameCounter;
    ++_impl->totalFrames;

    if (_impl->frameCounter < _impl->logInterval)
        return;

    _impl->frameCounter = 0;

    const core::u32 cells = _impl->world.cellCount();
    const core::u32 clients = _impl->sessionManager.activeCount();

    // GC empty cells periodically
    const core::u32 gcCount = _impl->world.gcEmptyCells();

    // The counters above used to be gathered and then dropped on the floor: the
    // heartbeat was a fixed string, so it reported nothing a monitor is for. The
    // legacy server printed frame, clients, entities and chunks (see legacy
    // apps/server/main.cpp "ServerMonitor"), which is what is restored here.
    char line[160];
    std::snprintf(line, sizeof(line), "instance %u | frame %llu | clients %u | cells %u | gc %u", _impl->instanceId,
                  static_cast<unsigned long long>(_impl->totalFrames), clients, cells, gcCount);
    core::Log::info("Monitor", line);
}

} // namespace lpl::engine::systems
