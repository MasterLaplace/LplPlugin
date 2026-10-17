#include <lpl/core/Log.hpp>
#include <lpl/engine/systems/ServerMonitorSystem.hpp>

#include <cstdio>

namespace lpl::engine::systems {

static const ecs::SystemDescriptor kMonitorDesc{"ServerMonitor", ecs::SchedulePhase::Network, {}};

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

    const core::u32 gcCount = _impl->world.gcEmptyCells();

    char line[160];
    std::snprintf(line, sizeof(line), "instance %u | frame %llu | clients %u | cells %u | gc %u", _impl->instanceId,
                  static_cast<unsigned long long>(_impl->totalFrames), clients, cells, gcCount);
    core::Log::info("Monitor", line);
}

} // namespace lpl::engine::systems
