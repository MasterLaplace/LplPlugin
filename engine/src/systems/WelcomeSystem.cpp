/**
 * @file WelcomeSystem.cpp
 * @brief Client-side: processes HandshakeAck to establish local entity identity.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/core/Log.hpp>
#include <lpl/engine/systems/WelcomeSystem.hpp>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::SystemDescriptor kWelcomeDesc{"Welcome", ecs::SchedulePhase::Input, {}};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct WelcomeSystem::Impl {
    EventQueues &queues;
    core::u32 &myEntityId;
    bool &connected;

    Impl(EventQueues &q, core::u32 &eid, bool &conn) : queues{q}, myEntityId{eid}, connected{conn} {}
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

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
