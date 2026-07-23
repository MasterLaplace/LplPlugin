/*
** LplPlugin — server packet routing test
**
** Proves the multi-instance server fans packets out to the right game instance:
** the server drains ONE shared socket and each datagram must land in the event
** queues of the instance its sender belongs to, never in another's. Also covers
** the legacy parity rule that a handshake retransmitted from an address that is
** already connected must not create a second player.
*/

#include <lpl/engine/Server.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/engine/PacketDispatch.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/net/Endpoint.hpp>
#    include <lpl/net/protocol/PacketBuilder.hpp>
#    include <lpl/net/transport/ITransport.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/std/memory.hpp>

#    include <arpa/inet.h>
#    include <sys/socket.h>
#    include <unistd.h>

#    include <atomic>
#    include <cstdio>

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

/// Builds a Handshake datagram the way a real client would.
[[nodiscard]] std::vector<core::byte> makeHandshake()
{
    core::byte payload[6]{};
    return net::protocol::buildPacket(net::protocol::PacketType::Handshake, {payload, 6});
}

/// A World that only records how often it was stepped, so a fan-out can be
/// checked for stepping every instance exactly once per tick.
class CountingWorld final : public engine::World {
public:
    void onFixedStep(core::f32 dt) override
    {
        engine::World::onFixedStep(dt);
        _steps.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] int steps() const noexcept { return _steps.load(std::memory_order_relaxed); }

private:
    std::atomic<int> _steps{0};
};

/// A World with entities that actually move, so the digest has something to
/// track. Physics is enabled by default, so the built-in PhysicsSystem the
/// server registers integrates them every tick.
class MovingWorld final : public engine::World {
public:
    core::Expected<void> onInit(engine::WorldContext &context) override
    {
        (void) context;

        const ecs::ComponentId ids[] = {ecs::ComponentId::Position, ecs::ComponentId::Velocity,
                                        ecs::ComponentId::AABB, ecs::ComponentId::Mass};
        const ecs::Archetype archetype{ids};
        for (core::u32 i = 0; i < 8; ++i)
            (void) registry().createEntity(archetype);

        using Vec = math::Vec3<math::Fixed32>;
        for (const auto &partition : registry().partitions())
        {
            for (const auto &chunk : partition->chunks())
            {
                auto *positions = static_cast<Vec *>(chunk->writeComponent(ecs::ComponentId::Position));
                auto *velocities = static_cast<Vec *>(chunk->writeComponent(ecs::ComponentId::Velocity));
                auto *sizes = static_cast<Vec *>(chunk->writeComponent(ecs::ComponentId::AABB));
                auto *massWrite = static_cast<math::Fixed32 *>(chunk->writeComponent(ecs::ComponentId::Mass));
                // Mass goes into BOTH buffers, as CubePile::init does: the
                // integrator reads it from the read buffer, and a zero-mass body
                // is inert — the digest would never move.
                auto *massRead =
                    static_cast<math::Fixed32 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Mass)));
                if (!positions || !velocities)
                    continue;

                for (core::u32 i = 0; i < chunk->count(); ++i)
                {
                    positions[i] = {math::Fixed32::fromFloat(static_cast<core::f32>(i)),
                                    math::Fixed32::fromFloat(50.0f), math::Fixed32::zero()};
                    velocities[i] = {math::Fixed32::fromFloat(0.5f), math::Fixed32::zero(), math::Fixed32::zero()};
                    if (sizes)
                        sizes[i] = {math::Fixed32::one(), math::Fixed32::one(), math::Fixed32::one()};
                    if (massWrite)
                        massWrite[i] = math::Fixed32::one();
                    if (massRead)
                        massRead[i] = math::Fixed32::one();
                }
            }
        }
        return {};
    }
};

/// Captures the packet a builder produced, so a test can feed the exact bytes a
/// real client would have put on the wire back into the server's decoder.
class LoopbackTransport final : public net::transport::ITransport {
public:
    core::Expected<void> open() override { return {}; }
    void close() override {}

    core::Expected<core::u32> send(std::span<const core::byte> data, const net::Endpoint *) override
    {
        _last.assign(data.begin(), data.end());
        return static_cast<core::u32>(data.size());
    }

    core::Expected<core::u32> receive(std::span<core::byte>, net::Endpoint *) override { return core::u32{0}; }

    const char *name() const noexcept override { return "LoopbackTransport"; }

    [[nodiscard]] std::span<const core::byte> lastPacket() const { return _last; }

private:
    std::vector<core::byte> _last;
};

} // namespace

int main()
{
    std::printf("== server packet routing ==\n");

    auto config = engine::Config::Builder{}.serverMode(true).tickRate(60).serverPort(45998).build();
    engine::Server server{config};

    // Hosting is refused until the shared transport exists: the instance systems
    // are wired to it, so a null socket would only fault on the first tick.
    check(server.addWorld(lpl::pmr::make_unique<engine::World>()) == engine::Server::kInvalidWorldId,
          "hosting before init() is refused");
    check(server.init().has_value(), "server opens its socket");

    // Two instances, as a server hosting two different games would have.
    const auto worldA = server.addWorld(lpl::pmr::make_unique<engine::World>());
    const auto worldB = server.addWorld(lpl::pmr::make_unique<engine::World>());

    check(worldA != engine::Server::kInvalidWorldId, "first instance hosted");
    check(worldB != engine::Server::kInvalidWorldId, "second instance hosted");
    check(worldA != worldB, "instances get distinct ids");
    check(server.worldCount() == 2, "two live instances");
    check(server.defaultWorld() == worldA, "first instance receives new clients by default");
    check(server.queues(worldA) != nullptr && server.queues(worldB) != nullptr, "each instance has its own queues");
    check(server.queues(worldA) != server.queues(worldB), "queues are not shared between instances");
    check(server.sessions(worldA) != nullptr && server.sessions(worldB) != nullptr,
          "each instance has its own session manager");
    check(server.sessions(worldA) != server.sessions(worldB), "sessions are not shared between instances");

    // --- each hosted instance gets the server-side systems ------------------ //
    // A World hosted by a Server has no Engine to register them, so without this
    // an instance would take inputs and broadcast a state that never moved.
    check(server.world(worldA) != nullptr && server.world(worldB) != nullptr, "both instances are reachable");
    check(server.world(worldA)->spatialPartition() != nullptr,
          "a hosted instance gets a spatial index even if its game never asked for one");
    check(server.world(worldA)->scheduler().systemCount() > 0, "instance A has its server systems registered");
    check(server.world(worldA)->scheduler().systemCount() == server.world(worldB)->scheduler().systemCount(),
          "every instance gets the same server-side chain");

    // Isolation of the OUTGOING path: broadcastState walks one manager, so a
    // client connected to A must be invisible from B, or A's world state would
    // be sent to B's players.
    {
        auto joined = server.sessions(worldA)->connect(1);
        check(joined.has_value(), "a client connects to instance A");
        joined.value()->setAddress(net::Endpoint::fromOctets(127, 0, 0, 1, 41000));
        check(server.sessions(worldA)->activeCount() == 1, "instance A has one client");
        check(server.sessions(worldB)->activeCount() == 0, "instance B sees no client of A");
    }

    // --- routing: a sender is bound to one instance and stays there --------- //
    const auto alice = net::Endpoint::fromOctets(127, 0, 0, 1, 40001);
    const auto bob = net::Endpoint::fromOctets(127, 0, 0, 1, 40002);

    check(server.worldForSender(alice) == engine::Server::kInvalidWorldId, "unknown sender is unbound");
    check(server.routeSenderToWorld(bob, worldB), "a sender can be bound to a chosen instance");
    check(server.worldForSender(bob) == worldB, "the binding is remembered");
    check(!server.routeSenderToWorld(alice, 99), "binding to a non-existent instance is refused");

    // --- dispatch: a packet lands only in its instance's queues ------------- //
    // Decoding is exercised directly (the socket itself needs no real peer here):
    // this is the exact call Server::pumpNetwork makes once it has picked the
    // destination, so it proves the fan-out contract without a live network.
    const auto datagram = makeHandshake();
    net::protocol::PacketHeader header{};
    std::span<const core::byte> payload;
    check(engine::detail::parsePacket(std::span<const core::byte>{datagram.data(), datagram.size()}, header, payload),
          "a well-formed handshake parses");

    engine::detail::dispatchPacket(header, payload, bob, *server.queues(worldB));

    check(!server.queues(worldB)->connects.empty(), "bob's handshake reached instance B");
    check(server.queues(worldA)->connects.empty(), "instance A saw nothing of bob's packet");

    // --- a malformed datagram is rejected, not routed ----------------------- //
    core::byte garbage[8]{};
    net::protocol::PacketHeader ignored{};
    std::span<const core::byte> ignoredPayload;
    check(!engine::detail::parsePacket(std::span<const core::byte>{garbage, sizeof(garbage)}, ignored, ignoredPayload),
          "a datagram without our magic is rejected");

    // --- legacy parity: a retransmitted handshake must not double-connect --- //
    {
        net::session::SessionManager sessions;
        auto first = sessions.connect(100);
        check(first.has_value(), "first connection accepted");
        first.value()->setAddress(alice);

        check(sessions.findByAddress(alice) != nullptr, "a connected client is found by its address");
        check(sessions.findByAddress(bob) == nullptr, "an unconnected address is not found");
        check(sessions.activeCount() == 1, "exactly one session after a duplicate handshake would be skipped");
    }

    // --- the hosted chain actually runs ------------------------------------- //
    // Ticking drives every instance's scheduler; this catches a system wired to
    // a dangling reference (the queues/sessions/input of another instance, or a
    // spatial index that was never created).
    for (int i = 0; i < 4; ++i)
        server.tick(1.0f / 60.0f);
    check(server.worldCount() == 2, "both instances survive a few ticks");

    // --- removing an instance drops its routes ------------------------------ //
    check(server.removeWorld(worldB), "instance B removed");
    check(server.worldForSender(bob) == engine::Server::kInvalidWorldId,
          "routes to a removed instance are dropped");
    check(server.worldCount() == 1, "one live instance remains");

    // --- §6.4 state hashing and desync detection ----------------------------- //
    // The digest folds ONLY authoritative Fixed32 state, and it is what a client
    // reports back so the server can tell whether its simulation still agrees.
    {
        auto hashConfig = engine::Config::Builder{}.serverMode(true).tickRate(60).serverPort(45997).build();
        engine::Server hashed{hashConfig};
        check(hashed.init().has_value(), "hash server opens its socket");

        const auto quiet = hashed.addWorld(lpl::pmr::make_unique<engine::World>());
        const auto moving = hashed.addWorld(lpl::pmr::make_unique<MovingWorld>());
        check(quiet != engine::Server::kInvalidWorldId && moving != engine::Server::kInvalidWorldId,
              "both hash instances hosted");

        hashed.tick(1.0f / 60.0f);
        const core::u64 tick1 = hashed.currentTick();
        const core::u64 movingAt1 = hashed.stateHash(moving);

        check(hashed.stateHash(quiet) == 0, "an empty instance folds to an empty digest");
        check(movingAt1 != 0, "an instance with entities folds to a non-zero digest");

        // A digest must follow the state: gravity moves the entities every tick.
        for (int i = 0; i < 5; ++i)
            hashed.tick(1.0f / 60.0f);
        check(hashed.stateHash(moving) != movingAt1, "the digest tracks the authoritative state as it evolves");

        // The verdict is what the server answers a client that reports a digest.
        check(hashed.checkClientHash(moving, tick1, movingAt1) == engine::Server::DesyncVerdict::Match,
              "a client agreeing on a past tick reads as Match");
        check(hashed.checkClientHash(moving, tick1, movingAt1 ^ 0xDEADBEEFULL) ==
                  engine::Server::DesyncVerdict::Diverged,
              "a client disagreeing on a past tick reads as Diverged");
        check(hashed.checkClientHash(moving, hashed.currentTick() + 100, movingAt1) ==
                  engine::Server::DesyncVerdict::TickUnknown,
              "a tick we have not stepped yet reads as TickUnknown");

        // The ring reuses a slot every kStateHashHistory ticks; the stored tick
        // number is what stops an ancient report from matching a fresh slot.
        for (core::usize i = 0; i < engine::Server::kStateHashHistory + 2; ++i)
            hashed.tick(1.0f / 60.0f);
        check(hashed.checkClientHash(moving, tick1, movingAt1) == engine::Server::DesyncVerdict::TickUnknown,
              "a tick that fell out of history reads as TickUnknown, not a false Match");

        // Instances are hashed independently: one game's state never colours another's.
        check(hashed.stateHash(quiet) != hashed.stateHash(moving), "instances have independent digests");
    }

    // --- §6.4 end to end: a client report crosses the wire format ----------- //
    // The digest logic above is exercised through its API; this drives the WHOLE
    // path a real client uses — build the packet, parse it, dispatch it into the
    // instance's queues, and let the server render its verdict.
    {
        auto wireConfig = engine::Config::Builder{}.serverMode(true).tickRate(60).serverPort(45996).build();
        engine::Server wired{wireConfig};
        check(wired.init().has_value(), "wire server opens its socket");

        const auto instance = wired.addWorld(lpl::pmr::make_unique<MovingWorld>());
        check(instance != engine::Server::kInvalidWorldId, "wire instance hosted");

        wired.tick(1.0f / 60.0f);
        const core::u64 agreedTick = wired.currentTick();
        const core::u64 agreedDigest = wired.stateHash(instance);

        const auto reporter = net::Endpoint::fromOctets(127, 0, 0, 1, 40500);
        check(wired.routeSenderToWorld(reporter, instance), "the reporting client is routed to the instance");

        // Encode exactly as StateHashReportSystem does, then decode exactly as
        // pumpNetwork does: this is the round trip, not a shortcut past it.
        LoopbackTransport loopback;
        check(net::protocol::sendStateHashReport(loopback, &reporter, agreedTick, agreedDigest).has_value(),
              "a client encodes a state hash report");

        net::protocol::PacketHeader reportHeader{};
        std::span<const core::byte> reportPayload;
        check(engine::detail::parsePacket(loopback.lastPacket(), reportHeader, reportPayload),
              "the report parses as one of our packets");
        check(reportHeader.type == net::protocol::PacketType::StateHashReport,
              "and carries the StateHashReport type");

        engine::detail::dispatchPacket(reportHeader, reportPayload, reporter, *wired.queues(instance));
        wired.tick(1.0f / 60.0f); // the server consumes reports at the top of a tick

        check(wired.matchedReportCount() == 1, "an agreeing client is counted as a match");
        check(wired.desyncCount() == 0, "and raises no desync");

        // Now a client that disagrees about the same tick.
        LoopbackTransport badLoopback;
        check(net::protocol::sendStateHashReport(badLoopback, &reporter, agreedTick, agreedDigest ^ 0x5A5A5A5AULL)
                  .has_value(),
              "a diverging client encodes its report");

        net::protocol::PacketHeader badHeader{};
        std::span<const core::byte> badPayload;
        check(engine::detail::parsePacket(badLoopback.lastPacket(), badHeader, badPayload), "the bad report parses");

        engine::detail::dispatchPacket(badHeader, badPayload, reporter, *wired.queues(instance));
        wired.tick(1.0f / 60.0f);

        check(wired.desyncCount() == 1, "a diverging client is detected through the full wire path");
    }

    // --- §6.5: snapshots are kept for post-mortem diagnosis ----------------- //
    {
        auto replayConfig = engine::Config::Builder{}
                                .serverMode(true)
                                .tickRate(60)
                                .serverPort(45995)
                                .replaySnapshotInterval(4)
                                .build();
        engine::Server recorded{replayConfig};
        check(recorded.init().has_value(), "replay server opens its socket");

        const auto instance = recorded.addWorld(lpl::pmr::make_unique<MovingWorld>());
        check(instance != engine::Server::kInvalidWorldId, "replay instance hosted");
        check(recorded.replay(instance) != nullptr, "the instance has a replay recorder");

        for (int i = 0; i < 20; ++i)
            recorded.tick(1.0f / 60.0f);

        check(recorded.replay(instance)->snapshotCount() == 5,
              "one snapshot every replaySnapshotInterval ticks (20/4)");
    }

    // --- backpressure is visible when the receive budget is exceeded -------- //
    // A tiny budget plus a flood of real datagrams: the server cannot drain the
    // socket within its budget, and backpressureEvents must record it. This runs
    // the real SocketTransport (recvmmsg on Linux), not a fake.
    {
        auto floodConfig = engine::Config::Builder{}
                               .serverMode(true)
                               .tickRate(60)
                               .serverPort(45994)
                               .maxPacketsPerTick(8)
                               .build();
        engine::Server flooded{floodConfig};
        check(flooded.init().has_value(), "flood server opens its socket");
        check(flooded.addWorld(lpl::pmr::make_unique<engine::World>()) != engine::Server::kInvalidWorldId,
              "flood instance hosted");
        check(flooded.backpressureEvents() == 0, "no backpressure before any traffic");

        const int client = ::socket(AF_INET, SOCK_DGRAM, 0);
        check(client >= 0, "test client socket opens");
        if (client >= 0)
        {
            sockaddr_in server{};
            server.sin_family = AF_INET;
            server.sin_port = htons(45994);
            server.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

            const auto datagram = makeHandshake();
            // Well past the budget of 8, so the socket is still full when the
            // server stops at its budget.
            for (int i = 0; i < 64; ++i)
                (void) ::sendto(client, datagram.data(), datagram.size(), 0,
                                reinterpret_cast<sockaddr *>(&server), sizeof(server));

            flooded.tick(1.0f / 60.0f);
            check(flooded.backpressureEvents() >= 1, "hitting the receive budget with a full socket is recorded");
            check(flooded.lastBackpressureTick() == flooded.currentTick(),
                  "and the tick it happened on is recorded");

            ::close(client);
        }
    }

    // --- parallel instance tick ---------------------------------------------- //
    // Config::serverWorkerThreads fans the per-instance steps across workers.
    // Every instance must be stepped exactly once per tick — no instance skipped
    // by the fan-out, none stepped twice.
    {
        auto parallelConfig =
            engine::Config::Builder{}.serverMode(true).tickRate(60).serverWorkerThreads(4).serverPort(45999).build();
        engine::Server parallel{parallelConfig};
        check(parallel.init().has_value(), "parallel server opens its socket");

        CountingWorld *instances[4]{};
        for (auto &instance : instances)
        {
            auto owned = lpl::pmr::make_unique<CountingWorld>();
            instance = owned.get();
            check(parallel.addWorld(std::move(owned)) != engine::Server::kInvalidWorldId,
                  "instance hosted on the parallel server");
        }

        constexpr int kTicks = 25;
        for (int i = 0; i < kTicks; ++i)
            parallel.tick(1.0f / 60.0f);

        bool everyInstanceStepped = true;
        for (const auto *instance : instances)
            everyInstanceStepped = everyInstanceStepped && instance->steps() == kTicks;
        check(everyInstanceStepped, "the parallel fan-out steps every instance exactly once per tick");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#else

#    include <cstdio>

int main()
{
    std::printf("server routing test skipped: built without LPL_HAS_NET\n");
    return 0;
}

#endif
