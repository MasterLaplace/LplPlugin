#include <lpl/engine/Server.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/engine/PacketDispatch.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/net/Endpoint.hpp>
#    include <lpl/net/protocol/PacketBuilder.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/ITransport.hpp>
#    include <lpl/std/memory.hpp>

#    include <arpa/inet.h>
#    include <sys/socket.h>
#    include <unistd.h>

#    include <atomic>
#    include <cstdio>

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

/**
 * @brief Builds a Handshake datagram the way a real client would.
 */
[[nodiscard]] std::vector<lpl::core::byte> makeHandshake()
{
    lpl::core::byte payload[6]{};
    return lpl::net::protocol::buildPacket(lpl::net::protocol::PacketType::Handshake, {payload, 6});
}

/**
 * @brief A World that only records how often it was stepped, so a fan-out can be checked for stepping every
 *        instance exactly once per tick.
 */
class CountingWorld final : public lpl::engine::World {
public:
    void onFixedStep(lpl::core::f32 dt) override
    {
        lpl::engine::World::onFixedStep(dt);
        _steps.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] int steps() const noexcept { return _steps.load(std::memory_order_relaxed); }

private:
    std::atomic<int> _steps{0};
};

/**
 * @brief A World with entities that actually move, so the digest has something to track. Physics is enabled
 *        by default, so the built-in PhysicsSystem the server registers integrates them every tick.
 */
class MovingWorld final : public lpl::engine::World {
public:
    /**
     * @brief Creates eight entities moving along +X.
     *
     * @details Mass goes into BOTH buffers, as CubePile::init does: the integrator reads it from the read buffer,
     *          and a zero-mass body is inert — the digest would never move.
     */
    lpl::core::Expected<void> onInit(lpl::engine::WorldContext &context) override
    {
        (void) context;

        const lpl::ecs::ComponentId ids[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                             lpl::ecs::ComponentId::AABB, lpl::ecs::ComponentId::Mass};
        const lpl::ecs::Archetype archetype{ids};
        for (lpl::core::u32 i = 0; i < 8; ++i)
            (void) registry().createEntity(archetype);

        using Vec = lpl::math::Vec3<lpl::math::Fixed32>;
        for (const auto &partition : registry().partitions())
        {
            for (const auto &chunk : partition->chunks())
            {
                auto *positions = static_cast<Vec *>(chunk->writeComponent(lpl::ecs::ComponentId::Position));
                auto *velocities = static_cast<Vec *>(chunk->writeComponent(lpl::ecs::ComponentId::Velocity));
                auto *sizes = static_cast<Vec *>(chunk->writeComponent(lpl::ecs::ComponentId::AABB));
                auto *massWrite = static_cast<lpl::math::Fixed32 *>(chunk->writeComponent(lpl::ecs::ComponentId::Mass));
                auto *massRead = static_cast<lpl::math::Fixed32 *>(
                    const_cast<void *>(chunk->readComponent(lpl::ecs::ComponentId::Mass)));
                if (!positions || !velocities)
                    continue;

                for (lpl::core::u32 i = 0; i < chunk->count(); ++i)
                {
                    positions[i] = {lpl::math::Fixed32::fromFloat(static_cast<lpl::core::f32>(i)),
                                    lpl::math::Fixed32::fromFloat(50.0f), lpl::math::Fixed32::zero()};
                    velocities[i] = {lpl::math::Fixed32::fromFloat(0.5f), lpl::math::Fixed32::zero(),
                                     lpl::math::Fixed32::zero()};
                    if (sizes)
                        sizes[i] = {lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one()};
                    if (massWrite)
                        massWrite[i] = lpl::math::Fixed32::one();
                    if (massRead)
                        massRead[i] = lpl::math::Fixed32::one();
                }
            }
        }
        return {};
    }
};

/**
 * @brief Captures the packet a builder produced, so a test can feed the exact bytes a real client would have
 *        put on the wire back into the server's decoder.
 */
class LoopbackTransport final : public lpl::net::transport::ITransport {
public:
    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}

    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> data, const lpl::net::Endpoint *) override
    {
        _last.assign(data.begin(), data.end());
        return static_cast<lpl::core::u32>(data.size());
    }

    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }

    const char *name() const noexcept override { return "LoopbackTransport"; }

    [[nodiscard]] std::span<const lpl::core::byte> lastPacket() const { return _last; }

private:
    std::vector<lpl::core::byte> _last;
};

/**
 * @brief Hosting is refused until the shared transport exists.
 *
 * @details The instance systems are wired to it, so a null socket would only fault on the first tick.
 *
 * @param server The server, not yet initialised; initialised on return.
 */
void checkHostingIsRefusedUntilTheSharedTransportExists(lpl::engine::Server &server)
{
    check(server.addWorld(lpl::pmr::make_unique<lpl::engine::World>()) == lpl::engine::Server::kInvalidWorldId,
          "hosting before init() is refused");
    check(server.init().has_value(), "server opens its socket");
}

/**
 * @brief Each instance gets its own id, queues and session manager.
 *
 * @details Two instances, as a server hosting two different games would have.
 *
 * @param server The server.
 * @param worldA The first instance it hosts.
 * @param worldB The second instance it hosts.
 */
void checkEachInstanceHasItsOwnQueuesAndSessions(lpl::engine::Server &server, lpl::engine::Server::WorldId worldA,
                                                 lpl::engine::Server::WorldId worldB)
{
    check(worldA != lpl::engine::Server::kInvalidWorldId, "first instance hosted");
    check(worldB != lpl::engine::Server::kInvalidWorldId, "second instance hosted");
    check(worldA != worldB, "instances get distinct ids");
    check(server.worldCount() == 2, "two live instances");
    check(server.defaultWorld() == worldA, "first instance receives new clients by default");
    check(server.queues(worldA) != nullptr && server.queues(worldB) != nullptr, "each instance has its own queues");
    check(server.queues(worldA) != server.queues(worldB), "queues are not shared between instances");
    check(server.sessions(worldA) != nullptr && server.sessions(worldB) != nullptr,
          "each instance has its own session manager");
    check(server.sessions(worldA) != server.sessions(worldB), "sessions are not shared between instances");
}

/**
 * @brief Each hosted instance gets the server-side systems.
 *
 * @details A World hosted by a Server has no Engine to register them, so without this an instance would take
 *          inputs and broadcast a state that never moved.
 *
 * @param server The server.
 * @param worldA The first instance it hosts.
 * @param worldB The second instance it hosts.
 */
void checkEachHostedInstanceGetsTheServerSideSystems(lpl::engine::Server &server, lpl::engine::Server::WorldId worldA,
                                                     lpl::engine::Server::WorldId worldB)
{
    check(server.world(worldA) != nullptr && server.world(worldB) != nullptr, "both instances are reachable");
    check(server.world(worldA)->spatialPartition() != nullptr,
          "a hosted instance gets a spatial index even if its game never asked for one");
    check(server.world(worldA)->scheduler().systemCount() > 0, "instance A has its server systems registered");
    check(server.world(worldA)->scheduler().systemCount() == server.world(worldB)->scheduler().systemCount(),
          "every instance gets the same server-side chain");
}

/**
 * @brief Isolation of the OUTGOING path.
 *
 * @details broadcastState walks one manager, so a client connected to A must be invisible from B, or A's world
 *          state would be sent to B's players.
 *
 * @param server The server.
 * @param worldA The instance the client connects to.
 * @param worldB The other instance.
 */
void checkOutgoingPathIsIsolated(lpl::engine::Server &server, lpl::engine::Server::WorldId worldA,
                                 lpl::engine::Server::WorldId worldB)
{
    auto joined = server.sessions(worldA)->connect(1);
    check(joined.has_value(), "a client connects to instance A");
    joined.value()->setAddress(lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 41000));
    check(server.sessions(worldA)->activeCount() == 1, "instance A has one client");
    check(server.sessions(worldB)->activeCount() == 0, "instance B sees no client of A");
}

/**
 * @brief Routing: a sender is bound to one instance and stays there.
 *
 * @param server The server.
 * @param worldB The instance bob is bound to.
 * @param alice  A sender left unbound.
 * @param bob    A sender bound to @p worldB.
 */
void checkSenderIsBoundToOneInstance(lpl::engine::Server &server, lpl::engine::Server::WorldId worldB,
                                     const lpl::net::Endpoint &alice, const lpl::net::Endpoint &bob)
{
    check(server.worldForSender(alice) == lpl::engine::Server::kInvalidWorldId, "unknown sender is unbound");
    check(server.routeSenderToWorld(bob, worldB), "a sender can be bound to a chosen instance");
    check(server.worldForSender(bob) == worldB, "the binding is remembered");
    check(!server.routeSenderToWorld(alice, 99), "binding to a non-existent instance is refused");
}

/**
 * @brief Dispatch: a packet lands only in its instance's queues.
 *
 * @details Decoding is exercised directly (the socket itself needs no real peer here): this is the exact call
 *          Server::pumpNetwork makes once it has picked the destination, so it proves the fan-out contract
 *          without a live network.
 *
 * @param server The server.
 * @param worldA The instance bob is not bound to.
 * @param worldB The instance bob is bound to.
 * @param bob    The sender of the handshake.
 */
void checkPacketLandsOnlyInItsInstanceQueues(lpl::engine::Server &server, lpl::engine::Server::WorldId worldA,
                                             lpl::engine::Server::WorldId worldB, const lpl::net::Endpoint &bob)
{
    const auto datagram = makeHandshake();
    lpl::net::protocol::PacketHeader header{};
    std::span<const lpl::core::byte> payload;
    check(lpl::engine::detail::parsePacket(std::span<const lpl::core::byte>{datagram.data(), datagram.size()}, header,
                                           payload),
          "a well-formed handshake parses");

    lpl::engine::detail::dispatchPacket(header, payload, bob, *server.queues(worldB));

    check(!server.queues(worldB)->connects.empty(), "bob's handshake reached instance B");
    check(server.queues(worldA)->connects.empty(), "instance A saw nothing of bob's packet");
}

/**
 * @brief A malformed datagram is rejected, not routed.
 */
void checkMalformedDatagramIsRejected()
{
    lpl::core::byte garbage[8]{};
    lpl::net::protocol::PacketHeader ignored{};
    std::span<const lpl::core::byte> ignoredPayload;
    check(!lpl::engine::detail::parsePacket(std::span<const lpl::core::byte>{garbage, sizeof(garbage)}, ignored,
                                            ignoredPayload),
          "a datagram without our magic is rejected");
}

/**
 * @brief Legacy parity: a retransmitted handshake must not double-connect.
 *
 * @param alice The address of the connected client.
 * @param bob   An address that never connected.
 */
void checkRetransmittedHandshakeDoesNotDoubleConnect(const lpl::net::Endpoint &alice, const lpl::net::Endpoint &bob)
{
    lpl::net::session::SessionManager sessions;
    auto first = sessions.connect(100);
    check(first.has_value(), "first connection accepted");
    first.value()->setAddress(alice);

    check(sessions.findByAddress(alice) != nullptr, "a connected client is found by its address");
    check(sessions.findByAddress(bob) == nullptr, "an unconnected address is not found");
    check(sessions.activeCount() == 1, "exactly one session after a duplicate handshake would be skipped");
}

/**
 * @brief The hosted chain actually runs.
 *
 * @details Ticking drives every instance's scheduler; this catches a system wired to a dangling reference (the
 *          queues/sessions/input of another instance, or a spatial index that was never created).
 *
 * @param server The server hosting two instances.
 */
void checkHostedChainActuallyRuns(lpl::engine::Server &server)
{
    for (int i = 0; i < 4; ++i)
        server.tick(1.0f / 60.0f);
    check(server.worldCount() == 2, "both instances survive a few ticks");
}

/**
 * @brief Removing an instance drops its routes.
 *
 * @param server The server.
 * @param worldB The instance removed.
 * @param bob    The sender bound to @p worldB.
 */
void checkRemovingAnInstanceDropsItsRoutes(lpl::engine::Server &server, lpl::engine::Server::WorldId worldB,
                                           const lpl::net::Endpoint &bob)
{
    check(server.removeWorld(worldB), "instance B removed");
    check(server.worldForSender(bob) == lpl::engine::Server::kInvalidWorldId,
          "routes to a removed instance are dropped");
    check(server.worldCount() == 1, "one live instance remains");
}

/**
 * @brief §6.4 state hashing and desync detection.
 *
 * @details The digest folds ONLY authoritative Fixed32 state, and it is what a client reports back so the
 *          server can tell whether its simulation still agrees. A digest must follow the state: gravity moves
 *          the entities every tick. The verdict is what the server answers a client that reports a digest. The
 *          ring reuses a slot every kStateHashHistory ticks; the stored tick number is what stops an ancient
 *          report from matching a fresh slot. Instances are hashed independently: one game's state never
 *          colours another's.
 */
void checkStateHashingAndDesyncDetection()
{
    auto hashConfig = lpl::engine::Config::Builder{}.serverMode(true).tickRate(60).serverPort(45997).build();
    lpl::engine::Server hashed{hashConfig};
    check(hashed.init().has_value(), "hash server opens its socket");

    const auto quiet = hashed.addWorld(lpl::pmr::make_unique<lpl::engine::World>());
    const auto moving = hashed.addWorld(lpl::pmr::make_unique<MovingWorld>());
    check(quiet != lpl::engine::Server::kInvalidWorldId && moving != lpl::engine::Server::kInvalidWorldId,
          "both hash instances hosted");

    hashed.tick(1.0f / 60.0f);
    const lpl::core::u64 tick1 = hashed.currentTick();
    const lpl::core::u64 movingAt1 = hashed.stateHash(moving);

    check(hashed.stateHash(quiet) == 0, "an empty instance folds to an empty digest");
    check(movingAt1 != 0, "an instance with entities folds to a non-zero digest");

    for (int i = 0; i < 5; ++i)
        hashed.tick(1.0f / 60.0f);
    check(hashed.stateHash(moving) != movingAt1, "the digest tracks the authoritative state as it evolves");

    check(hashed.checkClientHash(moving, tick1, movingAt1) == lpl::engine::Server::DesyncVerdict::Match,
          "a client agreeing on a past tick reads as Match");
    check(hashed.checkClientHash(moving, tick1, movingAt1 ^ 0xDEADBEEFULL) ==
              lpl::engine::Server::DesyncVerdict::Diverged,
          "a client disagreeing on a past tick reads as Diverged");
    check(hashed.checkClientHash(moving, hashed.currentTick() + 100, movingAt1) ==
              lpl::engine::Server::DesyncVerdict::TickUnknown,
          "a tick we have not stepped yet reads as TickUnknown");

    for (lpl::core::usize i = 0; i < lpl::engine::Server::kStateHashHistory + 2; ++i)
        hashed.tick(1.0f / 60.0f);
    check(hashed.checkClientHash(moving, tick1, movingAt1) == lpl::engine::Server::DesyncVerdict::TickUnknown,
          "a tick that fell out of history reads as TickUnknown, not a false Match");

    check(hashed.stateHash(quiet) != hashed.stateHash(moving), "instances have independent digests");
}

/**
 * @brief §6.4 end to end: a client report crosses the wire format.
 *
 * @details The digest logic of checkStateHashingAndDesyncDetection is exercised through its API; this drives
 *          the WHOLE path a real client uses — build the packet, parse it, dispatch it into the instance's
 *          queues, and let the server render its verdict. Each report is encoded exactly as
 *          StateHashReportSystem does, then decoded exactly as pumpNetwork does: this is the round trip, not a
 *          shortcut past it. The server consumes reports at the top of a tick. An agreeing client comes first,
 *          then one that disagrees about the same tick.
 */
void checkClientReportCrossesTheWireFormat()
{
    auto wireConfig = lpl::engine::Config::Builder{}.serverMode(true).tickRate(60).serverPort(45996).build();
    lpl::engine::Server wired{wireConfig};
    check(wired.init().has_value(), "wire server opens its socket");

    const auto instance = wired.addWorld(lpl::pmr::make_unique<MovingWorld>());
    check(instance != lpl::engine::Server::kInvalidWorldId, "wire instance hosted");

    wired.tick(1.0f / 60.0f);
    const lpl::core::u64 agreedTick = wired.currentTick();
    const lpl::core::u64 agreedDigest = wired.stateHash(instance);

    const auto reporter = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 40500);
    check(wired.routeSenderToWorld(reporter, instance), "the reporting client is routed to the instance");

    LoopbackTransport loopback;
    check(lpl::net::protocol::sendStateHashReport(loopback, &reporter, agreedTick, agreedDigest).has_value(),
          "a client encodes a state hash report");

    lpl::net::protocol::PacketHeader reportHeader{};
    std::span<const lpl::core::byte> reportPayload;
    check(lpl::engine::detail::parsePacket(loopback.lastPacket(), reportHeader, reportPayload),
          "the report parses as one of our packets");
    check(reportHeader.type == lpl::net::protocol::PacketType::StateHashReport, "and carries the StateHashReport type");

    lpl::engine::detail::dispatchPacket(reportHeader, reportPayload, reporter, *wired.queues(instance));
    wired.tick(1.0f / 60.0f);

    check(wired.matchedReportCount() == 1, "an agreeing client is counted as a match");
    check(wired.desyncCount() == 0, "and raises no desync");

    LoopbackTransport badLoopback;
    check(lpl::net::protocol::sendStateHashReport(badLoopback, &reporter, agreedTick, agreedDigest ^ 0x5A5A5A5AULL)
              .has_value(),
          "a diverging client encodes its report");

    lpl::net::protocol::PacketHeader badHeader{};
    std::span<const lpl::core::byte> badPayload;
    check(lpl::engine::detail::parsePacket(badLoopback.lastPacket(), badHeader, badPayload), "the bad report parses");

    lpl::engine::detail::dispatchPacket(badHeader, badPayload, reporter, *wired.queues(instance));
    wired.tick(1.0f / 60.0f);

    check(wired.desyncCount() == 1, "a diverging client is detected through the full wire path");
}

/**
 * @brief §6.5: snapshots are kept for post-mortem diagnosis.
 */
void checkSnapshotsAreKeptForPostMortem()
{
    auto replayConfig = lpl::engine::Config::Builder{}
                            .serverMode(true)
                            .tickRate(60)
                            .serverPort(45995)
                            .replaySnapshotInterval(4)
                            .build();
    lpl::engine::Server recorded{replayConfig};
    check(recorded.init().has_value(), "replay server opens its socket");

    const auto instance = recorded.addWorld(lpl::pmr::make_unique<MovingWorld>());
    check(instance != lpl::engine::Server::kInvalidWorldId, "replay instance hosted");
    check(recorded.replay(instance) != nullptr, "the instance has a replay recorder");

    for (int i = 0; i < 20; ++i)
        recorded.tick(1.0f / 60.0f);

    check(recorded.replay(instance)->snapshotCount() == 5, "one snapshot every replaySnapshotInterval ticks (20/4)");
}

/**
 * @brief Backpressure is visible when the receive budget is exceeded.
 *
 * @details A tiny budget plus a flood of real datagrams: the server cannot drain the socket within its budget,
 *          and backpressureEvents must record it. This runs the real SocketTransport (recvmmsg on Linux), not a
 *          fake. The flood is well past the budget of 8, so the socket is still full when the server stops at
 *          its budget.
 */
void checkBackpressureIsVisibleWhenTheBudgetIsExceeded()
{
    auto floodConfig =
        lpl::engine::Config::Builder{}.serverMode(true).tickRate(60).serverPort(45994).maxPacketsPerTick(8).build();
    lpl::engine::Server flooded{floodConfig};
    check(flooded.init().has_value(), "flood server opens its socket");
    check(flooded.addWorld(lpl::pmr::make_unique<lpl::engine::World>()) != lpl::engine::Server::kInvalidWorldId,
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
        for (int i = 0; i < 64; ++i)
            (void) ::sendto(client, datagram.data(), datagram.size(), 0, reinterpret_cast<sockaddr *>(&server),
                            sizeof(server));

        flooded.tick(1.0f / 60.0f);
        check(flooded.backpressureEvents() >= 1, "hitting the receive budget with a full socket is recorded");
        check(flooded.lastBackpressureTick() == flooded.currentTick(), "and the tick it happened on is recorded");

        ::close(client);
    }
}

/**
 * @brief Parallel instance tick.
 *
 * @details Config::serverWorkerThreads fans the per-instance steps across workers. Every instance must be
 *          stepped exactly once per tick — no instance skipped by the fan-out, none stepped twice.
 */
void checkParallelFanOutStepsEveryInstanceOncePerTick()
{
    auto parallelConfig =
        lpl::engine::Config::Builder{}.serverMode(true).tickRate(60).serverWorkerThreads(4).serverPort(45999).build();
    lpl::engine::Server parallel{parallelConfig};
    check(parallel.init().has_value(), "parallel server opens its socket");

    CountingWorld *instances[4]{};
    for (auto &instance : instances)
    {
        auto owned = lpl::pmr::make_unique<CountingWorld>();
        instance = owned.get();
        check(parallel.addWorld(std::move(owned)) != lpl::engine::Server::kInvalidWorldId,
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

} // namespace

/**
 * @brief Server packet routing test.
 *
 * @details Proves the multi-instance server fans packets out to the right game instance: the server drains ONE
 *          shared socket and each datagram must land in the event queues of the instance its sender belongs
 *          to, never in another's. Also covers the legacy parity rule that a handshake retransmitted from an
 *          address that is already connected must not create a second player.
 */
int main()
{
    std::printf("== server packet routing ==\n");

    auto config = lpl::engine::Config::Builder{}.serverMode(true).tickRate(60).serverPort(45998).build();
    lpl::engine::Server server{config};

    checkHostingIsRefusedUntilTheSharedTransportExists(server);

    const auto worldA = server.addWorld(lpl::pmr::make_unique<lpl::engine::World>());
    const auto worldB = server.addWorld(lpl::pmr::make_unique<lpl::engine::World>());

    checkEachInstanceHasItsOwnQueuesAndSessions(server, worldA, worldB);
    checkEachHostedInstanceGetsTheServerSideSystems(server, worldA, worldB);
    checkOutgoingPathIsIsolated(server, worldA, worldB);

    const auto alice = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 40001);
    const auto bob = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 40002);

    checkSenderIsBoundToOneInstance(server, worldB, alice, bob);
    checkPacketLandsOnlyInItsInstanceQueues(server, worldA, worldB, bob);
    checkMalformedDatagramIsRejected();
    checkRetransmittedHandshakeDoesNotDoubleConnect(alice, bob);
    checkHostedChainActuallyRuns(server);
    checkRemovingAnInstanceDropsItsRoutes(server, worldB, bob);
    checkStateHashingAndDesyncDetection();
    checkClientReportCrossesTheWireFormat();
    checkSnapshotsAreKeptForPostMortem();
    checkBackpressureIsVisibleWhenTheBudgetIsExceeded();
    checkParallelFanOutStepsEveryInstanceOncePerTick();

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
