#include <lpl/core/Log.hpp>
#include <lpl/engine/systems/WelcomeSystem.hpp>

namespace lpl::engine::systems {

static const ecs::SystemDescriptor kWelcomeDesc{"Welcome", ecs::SchedulePhase::Input, {}};

struct WelcomeSystem::Impl {
    EventQueues &queues;
    core::u32 &myEntityId;
    bool &connected;

    Impl(EventQueues &q, core::u32 &eid, bool &conn) : queues{q}, myEntityId{eid}, connected{conn} {}
};

WelcomeSystem::WelcomeSystem(EventQueues &queues, core::u32 &myEntityId, bool &connected)
    : _impl{std::make_unique<Impl>(queues, myEntityId, connected)}
{
}

WelcomeSystem::~WelcomeSystem() = default;

const ecs::SystemDescriptor &WelcomeSystem::descriptor() const noexcept { return kWelcomeDesc; }

void WelcomeSystem::execute(core::f32 /*dt*/)
{
    auto events = _impl->queues.welcomes.drain();

    for (const auto &ev : events)
    {
        _impl->myEntityId = ev.entityId;
        _impl->connected = true;
        core::Log::info("Welcome", "Server assigned local entity identity");
    }
}

} // namespace lpl::engine::systems
