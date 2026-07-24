/*
** LplPlugin — server session lifecycle test
**
** Proves a departing client no longer leaks its avatar. reapTimedOut/disconnect
** were defined but never called, so sessions AND their entities accumulated
** forever. Now SessionSystem reaps on explicit disconnect and on idle timeout,
** tearing down the entity + input state + spatial-index entry; InputProcessing
** touches the session each tick a client sends input, so a playing client is
** never reaped as idle.
*/

#include <lpl/engine/systems/SessionSystem.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/engine/systems/InputProcessingSystem.hpp>
#    include <lpl/input/InputManager.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/net/Endpoint.hpp>
#    include <lpl/net/session/SessionManager.hpp>

#    include <lpl/std/vector.hpp>

#    include <chrono>
#    include <cstdio>
#    include <memory>
#    include <thread>

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

class NullTransport final : public net::transport::ITransport {
public:
    core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "NullTransport"; }
    core::Expected<core::u32> receive(std::span<core::byte>, net::Endpoint *) override { return core::u32{0}; }
    core::Expected<core::u32> send(std::span<const core::byte> d, const net::Endpoint *) override
    {
        return static_cast<core::u32>(d.size());
    }
};

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

} // namespace

int main()
{
    std::printf("== server session lifecycle ==\n");

    const auto client = net::Endpoint::fromOctets(127, 0, 0, 1, 40020);
    auto transport = std::make_shared<NullTransport>();

    // ── Explicit disconnect tears everything down ──────────────────────────── //
    {
        net::session::SessionManager sessions;
        engine::EventQueues queues;
        input::InputManager inputs;
        [[maybe_unused]] auto ir = inputs.init();
        ecs::WorldPartition world{math::Fixed32::fromFloat(10.0f), 4096};
        ecs::Registry registry;
        engine::systems::SessionSystem session{sessions, queues, transport, inputs, world, registry, /*timeout*/ 0.0};

        engine::ConnectEvent connect{};
        connect.source = client;
        queues.connects.push(connect);
        session.execute(1.0f / 60.0f);

        check(sessions.activeCount() == 1 && registry.liveCount() == 1, "a client connects: one session, one entity");
        auto *s = sessions.findByAddress(client);
        const core::u32 id = s ? s->playerId() : ecs::EntityId::kNull;
        check(s != nullptr && inputs.hasEntity(id), "its input slot exists");
        const auto center =
            math::Vec3<math::Fixed32>{math::Fixed32::zero(), math::Fixed32::fromFloat(10.0f), math::Fixed32::zero()};
        pmr::vector<ecs::EntityId> hits;
        world.queryRadius(center, math::Fixed32::fromFloat(20.0f), hits);
        check(!hits.empty(), "and it is in the spatial index");

        engine::DisconnectEvent bye{};
        bye.source = client;
        queues.disconnects.push(bye);
        session.execute(1.0f / 60.0f);

        check(sessions.activeCount() == 0, "after disconnect: the session is gone");
        check(registry.liveCount() == 0, "the avatar entity is destroyed (no leak)");
        check(!inputs.hasEntity(id), "the input slot is freed");
        hits.clear();
        world.queryRadius(center, math::Fixed32::fromFloat(20.0f), hits);
        check(hits.empty(), "and it is removed from the spatial index");
    }

    // ── Idle timeout reaps a silent client ─────────────────────────────────── //
    {
        net::session::SessionManager sessions;
        engine::EventQueues queues;
        input::InputManager inputs;
        [[maybe_unused]] auto ir = inputs.init();
        ecs::WorldPartition world{math::Fixed32::fromFloat(10.0f), 4096};
        ecs::Registry registry;
        engine::systems::SessionSystem session{sessions, queues, transport, inputs, world, registry, /*timeout*/ 5.0};

        engine::ConnectEvent connect{};
        connect.source = client;
        queues.connects.push(connect);
        session.execute(1.0f / 60.0f);
        check(registry.liveCount() == 1, "a client connects");

        sleepMs(30); // well past the 5 ms timeout, with no activity
        session.execute(1.0f / 60.0f);

        check(sessions.activeCount() == 0 && registry.liveCount() == 0,
              "a client silent past the timeout is reaped, entity and all");
    }

    // ── An input is a heartbeat: a playing client is NOT reaped ────────────── //
    {
        net::session::SessionManager sessions;
        engine::EventQueues queues;
        input::InputManager inputs;
        [[maybe_unused]] auto ir = inputs.init();
        ecs::WorldPartition world{math::Fixed32::fromFloat(10.0f), 4096};
        ecs::Registry registry;
        engine::systems::SessionSystem session{sessions, queues, transport, inputs, world, registry, /*timeout*/ 100.0};
        engine::systems::InputProcessingSystem inputProc{queues, inputs, &sessions};

        engine::ConnectEvent connect{};
        connect.source = client;
        queues.connects.push(connect);
        session.execute(1.0f / 60.0f);
        auto *s = sessions.findByAddress(client);
        const core::u32 id = s ? s->playerId() : 0;
        check(registry.liveCount() == 1, "a client connects");

        // Send input (the heartbeat) partway through, then wait again — the total
        // idle time exceeds the timeout, but the time SINCE the input does not.
        sleepMs(60);
        engine::InputEvent input{};
        input.entityId = id;
        queues.inputs.push(std::move(input));
        inputProc.execute(1.0f / 60.0f); // touches the session
        sleepMs(60);
        session.execute(1.0f / 60.0f);

        check(sessions.activeCount() == 1 && registry.liveCount() == 1,
              "a client that keeps sending input survives (input resets the idle clock)");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#else

#    include <cstdio>

int main()
{
    std::printf("session lifecycle test skipped: built without LPL_HAS_NET\n");
    return 0;
}

#endif
