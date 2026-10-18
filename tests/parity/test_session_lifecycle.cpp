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

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

class NullTransport final : public lpl::net::transport::ITransport {
public:
    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "NullTransport"; }
    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }
    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> d, const lpl::net::Endpoint *) override
    {
        return static_cast<lpl::core::u32>(d.size());
    }
};

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

/**
 * @brief Explicit disconnect tears everything down.
 *
 * @details The session system runs with idle reaping disabled (a timeout of 0).
 *
 * @param client    The client's address.
 * @param transport The transport the session system sends through.
 */
void checkExplicitDisconnectTearsEverythingDown(const lpl::net::Endpoint &client,
                                                const std::shared_ptr<NullTransport> &transport)
{
    lpl::net::session::SessionManager sessions;
    lpl::engine::EventQueues queues;
    lpl::input::InputManager inputs;
    [[maybe_unused]] auto ir = inputs.init();
    lpl::ecs::WorldPartition world{lpl::math::Fixed32::fromFloat(10.0f), 4096};
    lpl::ecs::Registry registry;
    lpl::engine::systems::SessionSystem session{sessions, queues, transport, inputs, world, registry, 0.0};

    lpl::engine::ConnectEvent connect{};
    connect.source = client;
    queues.connects.push(connect);
    session.execute(1.0f / 60.0f);

    check(sessions.activeCount() == 1 && registry.liveCount() == 1, "a client connects: one session, one entity");
    auto *s = sessions.findByAddress(client);
    const lpl::core::u32 id = s ? s->playerId() : lpl::ecs::EntityId::kNull;
    check(s != nullptr && inputs.hasEntity(id), "its input slot exists");
    const auto center = lpl::math::Vec3<lpl::math::Fixed32>{
        lpl::math::Fixed32::zero(), lpl::math::Fixed32::fromFloat(10.0f), lpl::math::Fixed32::zero()};
    lpl::pmr::vector<lpl::ecs::EntityId> hits;
    world.queryRadius(center, lpl::math::Fixed32::fromFloat(20.0f), hits);
    check(!hits.empty(), "and it is in the spatial index");

    lpl::engine::DisconnectEvent bye{};
    bye.source = client;
    queues.disconnects.push(bye);
    session.execute(1.0f / 60.0f);

    check(sessions.activeCount() == 0, "after disconnect: the session is gone");
    check(registry.liveCount() == 0, "the avatar entity is destroyed (no leak)");
    check(!inputs.hasEntity(id), "the input slot is freed");
    hits.clear();
    world.queryRadius(center, lpl::math::Fixed32::fromFloat(20.0f), hits);
    check(hits.empty(), "and it is removed from the spatial index");
}

/**
 * @brief Idle timeout reaps a silent client.
 *
 * @details The session system runs with a 5 ms idle timeout, and the client stays silent for 30 ms, well past
 *          it, with no activity.
 *
 * @param client    The client's address.
 * @param transport The transport the session system sends through.
 */
void checkIdleTimeoutReapsASilentClient(const lpl::net::Endpoint &client,
                                        const std::shared_ptr<NullTransport> &transport)
{
    lpl::net::session::SessionManager sessions;
    lpl::engine::EventQueues queues;
    lpl::input::InputManager inputs;
    [[maybe_unused]] auto ir = inputs.init();
    lpl::ecs::WorldPartition world{lpl::math::Fixed32::fromFloat(10.0f), 4096};
    lpl::ecs::Registry registry;
    lpl::engine::systems::SessionSystem session{sessions, queues, transport, inputs, world, registry, 5.0};

    lpl::engine::ConnectEvent connect{};
    connect.source = client;
    queues.connects.push(connect);
    session.execute(1.0f / 60.0f);
    check(registry.liveCount() == 1, "a client connects");

    sleepMs(30);
    session.execute(1.0f / 60.0f);

    check(sessions.activeCount() == 0 && registry.liveCount() == 0,
          "a client silent past the timeout is reaped, entity and all");
}

/**
 * @brief An input is a heartbeat: a playing client is NOT reaped.
 *
 * @details The session system runs with a 100 ms idle timeout. The client sends input (the heartbeat), which
 *          the input processing system turns into a touch of its session, partway through, then waits again —
 *          the total idle time exceeds the timeout, but the time SINCE the input does not.
 *
 * @param client    The client's address.
 * @param transport The transport the session system sends through.
 */
void checkInputKeepsAPlayingClientFromBeingReaped(const lpl::net::Endpoint &client,
                                                  const std::shared_ptr<NullTransport> &transport)
{
    lpl::net::session::SessionManager sessions;
    lpl::engine::EventQueues queues;
    lpl::input::InputManager inputs;
    [[maybe_unused]] auto ir = inputs.init();
    lpl::ecs::WorldPartition world{lpl::math::Fixed32::fromFloat(10.0f), 4096};
    lpl::ecs::Registry registry;
    lpl::engine::systems::SessionSystem session{sessions, queues, transport, inputs, world, registry, 100.0};
    lpl::engine::systems::InputProcessingSystem inputProc{queues, inputs, &sessions};

    lpl::engine::ConnectEvent connect{};
    connect.source = client;
    queues.connects.push(connect);
    session.execute(1.0f / 60.0f);
    auto *s = sessions.findByAddress(client);
    const lpl::core::u32 id = s ? s->playerId() : 0;
    check(registry.liveCount() == 1, "a client connects");

    sleepMs(60);
    lpl::engine::InputEvent input{};
    input.entityId = id;
    queues.inputs.push(std::move(input));
    inputProc.execute(1.0f / 60.0f);
    sleepMs(60);
    session.execute(1.0f / 60.0f);

    check(sessions.activeCount() == 1 && registry.liveCount() == 1,
          "a client that keeps sending input survives (input resets the idle clock)");
}

} // namespace

/**
 * @brief Server session lifecycle test.
 *
 * @details Proves a departing client no longer leaks its avatar. reapTimedOut/disconnect were defined but
 *          never called, so sessions AND their entities accumulated forever. Now SessionSystem reaps on
 *          explicit disconnect and on idle timeout, tearing down the entity + input state + spatial-index
 *          entry; InputProcessing touches the session each tick a client sends input, so a playing client is
 *          never reaped as idle.
 */
int main()
{
    std::printf("== server session lifecycle ==\n");

    const auto client = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 40020);
    auto transport = std::make_shared<NullTransport>();

    checkExplicitDisconnectTearsEverythingDown(client, transport);
    checkIdleTimeoutReapsASilentClient(client, transport);
    checkInputKeepsAPlayingClientFromBeingReaped(client, transport);

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
