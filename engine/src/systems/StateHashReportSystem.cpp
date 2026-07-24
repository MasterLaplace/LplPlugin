/**
 * @file StateHashReportSystem.cpp
 * @brief Client-side desync reporting implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-22
 * @copyright MIT License
 */

#include <lpl/engine/systems/StateHashReportSystem.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/core/Log.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/net/protocol/PacketBuilder.hpp>
#    include <lpl/net/transport/ITransport.hpp>

namespace lpl::engine::systems {

static const ecs::SystemDescriptor kStateHashReportDesc{"StateHashReport", ecs::SchedulePhase::Network, {}};

struct StateHashReportSystem::Impl {
    const World &world;
    std::shared_ptr<net::transport::ITransport> transport;
    const bool &connected;
    core::u32 interval;
    core::u32 sinceLastReport{0};
    core::u64 tick{0};

    Impl(const World &w, std::shared_ptr<net::transport::ITransport> t, const bool &c, core::u32 i)
        : world{w}, transport{std::move(t)}, connected{c}, interval{i == 0 ? 1u : i}
    {
    }
};

StateHashReportSystem::StateHashReportSystem(const World &world, std::shared_ptr<net::transport::ITransport> transport,
                                             const bool &connected, core::u32 interval)
    : _impl{std::make_unique<Impl>(world, std::move(transport), connected, interval)}
{
}

StateHashReportSystem::~StateHashReportSystem() = default;

const ecs::SystemDescriptor &StateHashReportSystem::descriptor() const noexcept { return kStateHashReportDesc; }

core::u64 StateHashReportSystem::currentTick() const noexcept { return _impl->tick; }

void StateHashReportSystem::execute(core::f32 /*dt*/)
{
    // The tick counter advances whether or not we are connected, so it keeps
    // meaning "how many steps this world has taken" rather than "how many we
    // reported".
    ++_impl->tick;

    if (!_impl->connected || !_impl->transport)
        return;

    if (++_impl->sinceLastReport < _impl->interval)
        return;

    _impl->sinceLastReport = 0;

    // nullptr address: the client transport already points at the server.
    [[maybe_unused]] auto result =
        net::protocol::sendStateHashReport(*_impl->transport, nullptr, _impl->tick, _impl->world.stateHash());
}

} // namespace lpl::engine::systems

#endif // LPL_HAS_NET
