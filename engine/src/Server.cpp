/**
 * @file Server.cpp
 * @brief Multi-instance game server implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-21
 * @copyright MIT License
 */

#include <lpl/engine/Server.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/core/Log.hpp>
#    include <lpl/concurrency/JobSystem.hpp>
#    include <lpl/engine/PacketDispatch.hpp>
#    include <lpl/engine/systems/AoiBroadcastSystem.hpp>
#    include <lpl/engine/systems/BroadcastSystem.hpp>
#    include <lpl/engine/systems/InputProcessingSystem.hpp>
#    include <lpl/engine/systems/MovementSystem.hpp>
#    include <lpl/engine/systems/PhysicsSystem.hpp>
#    include <lpl/engine/systems/ServerMonitorSystem.hpp>
#    include <lpl/engine/systems/SessionSystem.hpp>
#    include <lpl/input/InputManager.hpp>
#    include <lpl/memory/ArenaAllocator.hpp>
#    include <lpl/net/netcode/AuthoritativeStrategy.hpp>
#    include <lpl/physics/CpuPhysicsBackend.hpp>
#    include <lpl/serial/ReplayRecorder.hpp>
#    include <lpl/serial/StateSnapshot.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/KernelTransport.hpp>
#    include <lpl/net/protocol/Protocol.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/SocketTransport.hpp>
#    include <lpl/std/unordered_map.hpp>

#    include <array>
#    include <lpl/platform/linux/LinuxPlatform.hpp>

#    include <atomic>
#    include <memory>

namespace lpl::engine {

struct Server::Impl {
    Config config;

    /// Shared by reference with every hosted World: one load per asset for the
    /// whole server, not one per instance.
    ResourceManager resources;

    /// Instance slots. A removed instance leaves a null hole rather than
    /// compacting, so live ids stay stable for the server's lifetime.
    lpl::pmr::vector<lpl::pmr::unique_ptr<World>> worlds;

    /// One set of queues per instance, parallel to @c worlds: a packet is decoded
    /// into the queues of the instance its sender belongs to, so one game's
    /// inputs can never be seen by another's systems.
    lpl::pmr::vector<lpl::pmr::unique_ptr<EventQueues>> queues;

    /// Sender -> instance. Keyed by the endpoint packed into 48 bits (IPv4 +
    /// port), which is exactly what identifies a UDP peer.
    lpl::pmr::unordered_map<core::u64, WorldId> routing;

    /// Where a sender we have never seen is placed.
    WorldId defaultWorld{kInvalidWorldId};

    /// One set of connected clients per instance, parallel to @c worlds. NOT
    /// shared: broadcastState() walks a whole manager, so one manager for all
    /// instances would leak game A's state to game B's players.
    lpl::pmr::vector<lpl::pmr::unique_ptr<net::session::SessionManager>> sessions;

    /// One input state per instance, parallel to @c worlds. NOT shared either:
    /// InputProcessingSystem keys input by ENTITY id and MovementSystem reads it
    /// back from the same map, but entity ids are minted per registry — so two
    /// instances would both own an entity 1, and game A's steering would drive
    /// game B's player. Same failure mode as a shared SessionManager, on the
    /// inbound path instead of the outbound one.
    lpl::pmr::vector<lpl::pmr::unique_ptr<input::InputManager>> inputs;

    /// One physics backend per instance, parallel to @c worlds: a backend is
    /// bound to the registry it integrates, so it cannot be shared.
    lpl::pmr::vector<lpl::pmr::unique_ptr<physics::IPhysicsBackend>> physicsBackends;

    lpl::pmr::unique_ptr<net::netcode::INetcodeStrategy> netcode;
    std::shared_ptr<net::transport::ITransport> transport;

    /// Rolling digest history per instance, parallel to @c worlds: a ring of
    /// kStateHashHistory entries indexed by (tick % size). A client reports a
    /// digest for a tick we stepped a few frames ago, so the current digest is
    /// useless for the comparison — we need the one it actually refers to.
    struct HashRing {
        core::u64 ticks[Server::kStateHashHistory]{};
        core::u64 digests[Server::kStateHashHistory]{};
        bool filled[Server::kStateHashHistory]{};
        core::u64 lastDigest{0};
    };
    lpl::pmr::vector<lpl::pmr::unique_ptr<HashRing>> hashes;

    /// Ticks stepped since the server started; the timeline digests are keyed by.
    core::u64 tickCounter{0};

    /// One replay recorder per instance, parallel to @c worlds: periodic
    /// snapshots kept so a divergence can be diagnosed after the fact (§6.4.2).
    lpl::pmr::vector<lpl::pmr::unique_ptr<serial::ReplayRecorder>> replays;

    /// Desync telemetry, cumulative over the server's life.
    core::u64 desyncCount{0};
    core::u64 matchedReportCount{0};
    core::u64 staleReportCount{0};

    /// The most recent captured divergence, for post-mortem diagnosis (§6.4.2).
    Server::DesyncReport lastDesync{};
    bool hasDesync{false};

    /// Reused receive scratch: one contiguous block sliced into per-slot buffers,
    /// so draining the socket never allocates on the hot path.
    std::vector<core::byte> receiveStorage;
    std::vector<net::transport::ReceiveSlot> receiveSlots;

    /// How many ticks hit the per-tick receive budget with the socket still
    /// full — i.e. the server did not keep up and the kernel is now shedding
    /// load. The one number that makes an otherwise invisible wall visible.
    core::u64 backpressureEvents{0};
    core::u64 lastBackpressureTick{0};

    /// Fans the per-instance ticks out across threads. Null when
    /// Config::serverWorkerThreads is 0, which ticks instances sequentially.
    lpl::pmr::unique_ptr<concurrency::JobSystem> jobs;

    /// The World hooks need host services; the server has no display or input,
    /// so it presents a headless platform to them.
    platform::linux_host::LinuxPlatform platform;
    memory::ArenaAllocator arena;

    std::atomic<bool> running{false};
    bool initialised{false};

    explicit Impl(Config cfg) : config{std::move(cfg)}, arena{config.arenaSize()} {}
};

namespace {

/// Packs an endpoint into the key the routing table is indexed by.
[[nodiscard]] core::u64 senderKey(const net::Endpoint &endpoint) noexcept
{
    return (static_cast<core::u64>(endpoint.address()) << 16) | static_cast<core::u64>(endpoint.port());
}

} // namespace

Server::Server(Config config) : _impl{lpl::pmr::make_unique<Impl>(std::move(config))} {}

Server::~Server()
{
    if (_impl && _impl->initialised)
        shutdown();
}

core::Expected<void> Server::init()
{
    core::Log::info("Server: opening shared network session");

    auto socketTransport = std::make_shared<net::transport::SocketTransport>(_impl->config.serverPort());
    if (auto res = socketTransport->open(); !res)
    {
        core::Log::error("Server: failed to open the listening socket");
        return res;
    }
    _impl->transport = std::move(socketTransport);

    _impl->netcode = lpl::pmr::make_unique<net::netcode::AuthoritativeStrategy>();

    if (_impl->config.serverWorkerThreads() > 0)
    {
        _impl->jobs = lpl::pmr::make_unique<concurrency::JobSystem>(_impl->config.serverWorkerThreads());
        core::Log::info("Server: hosted instances tick in parallel");
    }

    _impl->initialised = true;

    core::Log::info("Server: ready");
    return {};
}

Server::WorldId Server::addWorld(lpl::pmr::unique_ptr<World> world)
{
    if (!world)
        return kInvalidWorldId;

    // The instance's session and broadcast systems are built around the shared
    // transport, so it has to exist before they are wired: hosting before init()
    // would hand them a null socket that only faults on the first tick, far from
    // the call that caused it.
    if (!_impl->initialised)
    {
        core::Log::error("Server: addWorld called before init(), refused");
        return kInvalidWorldId;
    }

    // No Engine hosts this World — the server does — so the context carries a
    // null engine handle, which WorldContext documents as legitimate.
    WorldContext context{_impl->platform, _impl->resources, _impl->arena, _impl->config, nullptr};

    if (auto res = world->onInit(context); !res)
    {
        core::Log::error("Server: instance failed to initialise, refused");
        return kInvalidWorldId;
    }

    _impl->worlds.push_back(std::move(world));
    _impl->queues.push_back(lpl::pmr::make_unique<EventQueues>());
    _impl->sessions.push_back(lpl::pmr::make_unique<net::session::SessionManager>());
    _impl->inputs.push_back(lpl::pmr::make_unique<input::InputManager>());
    _impl->physicsBackends.push_back(nullptr);
    _impl->hashes.push_back(lpl::pmr::make_unique<Impl::HashRing>());
    _impl->replays.push_back(lpl::pmr::make_unique<serial::ReplayRecorder>(
        _impl->config.replaySnapshotInterval() > 0 ? _impl->config.replaySnapshotInterval() : 144u));
    const auto id = static_cast<WorldId>(_impl->worlds.size() - 1u);

    [[maybe_unused]] auto inputRes = _impl->inputs[id]->init();
    registerInstanceSystems(id);

    // Every system is registered by now (the game's in onInit, the server's just
    // above), so the scheduler can resolve its dependency graph. Without this
    // the instance ticks an empty wave list: nothing it registered would ever
    // run. Engine::init does the same after its own registrations, and the
    // legacy server called core.buildSchedule() at the same point.
    if (auto built = _impl->worlds[id]->build(); !built)
    {
        core::Log::error("Server: instance schedule has a dependency cycle, refused");
        removeWorld(id);
        return kInvalidWorldId;
    }

    // The first instance hosted receives new clients unless told otherwise.
    if (_impl->defaultWorld == kInvalidWorldId)
        _impl->defaultWorld = id;

    core::Log::info("Server: instance hosted");
    return id;
}

void Server::registerInstanceSystems(WorldId id)
{
    World &world = *_impl->worlds[id];
    auto &scheduler = world.scheduler();

    // Every one of these systems indexes entities through the spatial grid, so
    // the instance needs one whether or not its game asked for a broad-phase.
    // fromFloat, NOT Fixed32{10}: the raw-representation constructor would give
    // a cell size of 10/65536 and put every entity in its own cell.
    ecs::WorldPartition *spatial = world.spatialPartition();
    if (spatial == nullptr)
        spatial = &world.enableSpatialPartition(math::Fixed32::fromFloat(10.0f), _impl->config.worldCellCapacity());

    // Legacy order (apps/server/main.cpp): NetworkReceive -> Session ->
    // InputProcessing -> Movement -> Physics, then Broadcast and the monitor
    // after the swap. The scheduler orders by phase, so registration order is
    // free — but the FIRST of those systems is deliberately absent here:
    // Server::pumpNetwork() already drains the shared socket once per tick and
    // routes each datagram to its instance. Registering a NetworkReceiveSystem
    // on a hosted World would put N systems in a race for one socket, each
    // stealing datagrams meant for the others.
    {
        auto session = lpl::pmr::make_unique<systems::SessionSystem>(
            *_impl->sessions[id], *_impl->queues[id], _impl->transport, *_impl->inputs[id], *spatial,
            world.registry(), _impl->config.sessionTimeoutMs());
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(session));
    }
    {
        auto inputProc = lpl::pmr::make_unique<systems::InputProcessingSystem>(*_impl->queues[id], *_impl->inputs[id],
                                                                               _impl->sessions[id].get());
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(inputProc));
    }
    {
        auto movement = lpl::pmr::make_unique<systems::MovementSystem>(*_impl->inputs[id], world.registry());
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(movement));
    }

    // Physics: the legacy server stepped the world every tick (its "Physics"
    // PreSwap system). A hosted World has no Engine to register it, so without
    // this an instance would receive inputs and broadcast a state that never
    // moved.
    if (_impl->config.enablePhysics())
    {
        _impl->physicsBackends[id] = lpl::pmr::make_unique<physics::CpuPhysicsBackend>(world.registry());
        [[maybe_unused]] auto initRes = _impl->physicsBackends[id]->init();

        auto physics = lpl::pmr::make_unique<systems::PhysicsSystem>(*spatial, *_impl->physicsBackends[id],
                                                                     world.registry());
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(physics));
    }

    // Interest-managed broadcast when a radius is set, full broadcast otherwise.
    // A positive radius sends each client only what is near its avatar, as
    // spawn/despawn/delta — the O(clients × N) wall goes away (§2.6). Zero keeps
    // the full-state fallback, unchanged.
    if (_impl->config.interestRadius() > math::Fixed32::zero())
    {
        auto broadcast = lpl::pmr::make_unique<systems::AoiBroadcastSystem>(
            *_impl->sessions[id], _impl->transport, *spatial, world.registry(), _impl->config.interestRadius(),
            _impl->config.keyframeInterval(), _impl->config.bandwidthBudgetBytes());
        broadcast->setNetworkLod(_impl->config.lodNearRadius(), _impl->config.lodFarInterval());
        broadcast->setPrecisionLod(_impl->config.worldExtent(), _impl->config.lodFarPosBits());
        broadcast->setReliableBaseline(_impl->config.reliableBaseline());
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(broadcast));
    }
    else
    {
        auto broadcast = lpl::pmr::make_unique<systems::BroadcastSystem>(*_impl->sessions[id], _impl->transport,
                                                                        *spatial, world.registry());
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(broadcast));
    }
    {
        auto monitor = lpl::pmr::make_unique<systems::ServerMonitorSystem>(*_impl->sessions[id], *spatial, 300u, id);
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(monitor));
    }
}

bool Server::removeWorld(WorldId id)
{
    if (id >= _impl->worlds.size() || !_impl->worlds[id])
        return false;

    _impl->worlds[id]->onShutdown();
    // The World owns the systems, and those systems hold references to the
    // queues, sessions, input state and physics backend below: destroy it first.
    _impl->worlds[id].reset();
    _impl->queues[id].reset();
    _impl->sessions[id].reset();
    if (_impl->inputs[id])
        _impl->inputs[id]->shutdown();
    _impl->inputs[id].reset();
    _impl->physicsBackends[id].reset();
    _impl->hashes[id].reset();
    _impl->replays[id].reset();

    // Senders bound to a dead instance would otherwise route into a null slot.
    for (auto it = _impl->routing.begin(); it != _impl->routing.end();)
    {
        if (it->second == id)
            it = _impl->routing.erase(it);
        else
            ++it;
    }

    if (_impl->defaultWorld == id)
    {
        _impl->defaultWorld = kInvalidWorldId;
        for (core::usize i = 0; i < _impl->worlds.size(); ++i)
        {
            if (_impl->worlds[i])
            {
                _impl->defaultWorld = static_cast<WorldId>(i);
                break;
            }
        }
    }
    return true;
}

World *Server::world(WorldId id) noexcept { return (id < _impl->worlds.size()) ? _impl->worlds[id].get() : nullptr; }

core::usize Server::worldCount() const noexcept
{
    core::usize live = 0;
    for (const auto &world : _impl->worlds)
    {
        if (world)
            ++live;
    }
    return live;
}

EventQueues *Server::queues(WorldId id) noexcept
{
    return (id < _impl->queues.size()) ? _impl->queues[id].get() : nullptr;
}

net::session::SessionManager *Server::sessions(WorldId id) noexcept
{
    return (id < _impl->sessions.size()) ? _impl->sessions[id].get() : nullptr;
}

Server::WorldId Server::defaultWorld() const noexcept { return _impl->defaultWorld; }

void Server::setDefaultWorld(WorldId id) noexcept
{
    if (id < _impl->worlds.size() && _impl->worlds[id])
        _impl->defaultWorld = id;
}

bool Server::routeSenderToWorld(const net::Endpoint &sender, WorldId id)
{
    if (id >= _impl->worlds.size() || !_impl->worlds[id])
        return false;
    _impl->routing[senderKey(sender)] = id;
    return true;
}

Server::WorldId Server::worldForSender(const net::Endpoint &sender) const noexcept
{
    const auto it = _impl->routing.find(senderKey(sender));
    return (it == _impl->routing.end()) ? kInvalidWorldId : it->second;
}

void Server::pumpNetwork()
{
    if (!_impl->transport)
        return;

    constexpr core::u32 kBurst = 64; // one recvmmsg's worth per drain
    static constexpr core::usize kSlotSize =
        sizeof(net::protocol::PacketHeader) + net::session::SessionManager::kMaxPayloadSize;

    // Reused across ticks (cleared, capacity kept) so draining does not allocate.
    auto &storage = _impl->receiveStorage;
    auto &slots = _impl->receiveSlots;
    if (storage.empty())
    {
        storage.resize(static_cast<core::usize>(kBurst) * kSlotSize);
        slots.resize(kBurst);
        for (core::u32 i = 0; i < kBurst; ++i)
            slots[i].buffer = std::span<core::byte>{storage.data() + static_cast<core::usize>(i) * kSlotSize, kSlotSize};
    }

    const core::u32 budget = _impl->config.maxPacketsPerTick();
    core::u32 drained = 0;

    while (drained < budget)
    {
        for (auto &slot : slots)
            slot.length = 0;

        auto result = _impl->transport->receiveBatch(std::span<net::transport::ReceiveSlot>{slots.data(), slots.size()});
        if (!result.has_value())
            break;

        const core::u32 count = result.value();
        if (count == 0)
            break; // socket drained: we kept up this tick

        for (core::u32 i = 0; i < count; ++i)
        {
            const auto &slot = slots[i];
            if (slot.length == 0)
                continue;

            net::protocol::PacketHeader header{};
            std::span<const core::byte> payload;
            if (!detail::parsePacket(std::span<const core::byte>{slot.buffer.data(), slot.length}, header, payload))
                continue;

            // Which instance is this sender playing in? An unknown sender is
            // placed in the default instance — that is the join, and it is
            // remembered so every later packet from it goes to the same game.
            auto worldId = worldForSender(slot.source);
            if (worldId == kInvalidWorldId)
            {
                worldId = _impl->defaultWorld;
                if (worldId == kInvalidWorldId)
                    continue; // nothing hosted yet: drop rather than guess
                _impl->routing[senderKey(slot.source)] = worldId;
            }

            if (auto *destination = queues(worldId))
                detail::dispatchPacket(header, payload, slot.source, *destination);
        }

        drained += count;

        // A full burst means the socket may still have more; a short one means
        // it ran dry. Only keep looping while it stays full.
        if (count < slots.size())
            break;
    }

    // We stopped at the budget with the socket still returning full bursts: we
    // did NOT keep up this tick, and the kernel's RX buffer is now absorbing the
    // overflow (and will silently drop past its limit). This is the metric that
    // makes that visible — the wall is otherwise invisible.
    if (drained >= budget)
    {
        ++_impl->backpressureEvents;
        _impl->lastBackpressureTick = _impl->tickCounter + 1; // recorded before tickCounter is bumped
    }
}

void Server::tick(core::f32 dt)
{
    // Drain the shared socket ONCE and fan each packet out to its instance. The
    // instances never touch the transport: the server routes, a World simulates.
    pumpNetwork();

    // Reports decoded by pumpNetwork are checked before the instances advance,
    // so a verdict always refers to the tick history as the client saw it.
    consumeStateHashReports();

    if (_impl->netcode)
        _impl->netcode->tick(dt);

    // Instances are independent — each owns its registry, scheduler, spatial
    // index, physics backend, queues, sessions and input state — so they can be
    // stepped concurrently. What they share is either thread-safe (the asset
    // cache), read-only for the whole step (the transport's fd; sendto is atomic
    // per datagram), or already done: pumpNetwork() and the netcode ran above,
    // single-threaded, so no worker touches the socket for reading.
    //
    // Checked under ThreadSanitizer (test-server-routing, 4 workers): every
    // report sits on the JobSystem's work-stealing transfer, never on instance
    // state — no Registry, WorldPartition, SessionManager, EventQueues or
    // InputManager appears in any of them. Running the same test with a single
    // worker, where nothing can be stolen, reports nothing at all, which pins
    // those reports to ChaseLevDeque::steal: it orders the handover with
    // std::atomic_thread_fence, which TSan does not model. That is a
    // pre-existing property of the job system, not of this fan-out — and it is
    // why serverWorkerThreads defaults to 0 (sequential) rather than on.
    if (_impl->jobs && _impl->worlds.size() > 1u)
    {
        concurrency::JobHandle handle{};

        for (auto &world : _impl->worlds)
        {
            if (!world)
                continue;
            World *instance = world.get();
            _impl->jobs->kickJob([instance, dt]() { instance->onFixedStep(dt); }, handle);
        }
        _impl->jobs->waitForCounter(handle, 0);
        recordStateHashes();
        return;
    }

    for (auto &world : _impl->worlds)
    {
        if (world)
            world->onFixedStep(dt);
    }
    recordStateHashes();
}

void Server::recordStateHashes()
{
    // Deliberately AFTER the fan-out has joined: every instance is at rest and
    // this walk is single-threaded, so it never races a step in flight.
    ++_impl->tickCounter;

    const auto slot = static_cast<core::usize>(_impl->tickCounter % kStateHashHistory);

    for (core::usize i = 0; i < _impl->worlds.size(); ++i)
    {
        if (!_impl->worlds[i])
            continue;

        auto &ring = *_impl->hashes[i];
        const core::u64 digest = _impl->worlds[i]->stateHash();

        ring.ticks[slot] = _impl->tickCounter;
        ring.digests[slot] = digest;
        ring.filled[slot] = true;
        ring.lastDigest = digest;

        // §6.5: keep a periodic snapshot so a divergence reported later can be
        // diagnosed post-mortem against the state we actually held.
        if (_impl->config.replaySnapshotInterval() > 0 &&
            (_impl->tickCounter % _impl->config.replaySnapshotInterval()) == 0)
        {
            serial::StateSnapshot snapshot;
            snapshot.setTick(_impl->tickCounter);
            snapshot.addEntityBlob(0, reinterpret_cast<const core::byte *>(&digest), sizeof(digest));
            snapshot.rehash();
            _impl->replays[i]->recordSnapshot(std::move(snapshot));
        }
    }
}

void Server::consumeStateHashReports()
{
    // A client's report names a tick it has already simulated, so this is where
    // the digest history earns its keep: checkClientHash looks that tick up
    // rather than comparing against the state we hold right now.
    for (core::usize i = 0; i < _impl->queues.size(); ++i)
    {
        if (!_impl->queues[i])
            continue;

        auto reports = _impl->queues[i]->stateHashReports.drain();
        for (const auto &report : reports)
        {
            const auto verdict = checkClientHash(static_cast<WorldId>(i), report.tick, report.digest);

            switch (verdict)
            {
            case DesyncVerdict::Diverged: {
                // The book's answer is a forced resynchronisation of that client.
                // We report it; acting on it is the netcode strategy's call.
                core::Log::error("Server: client desync detected, its state diverged from ours");
                ++_impl->desyncCount;

                // Capture the divergence for post-mortem (§6.4.2): the tick, both
                // digests, and who reported it. The server's own full state for
                // that tick lives in its ReplayRecorder (replay(id)), so this is
                // enough to reconstruct and compare the two states after the fact.
                DesyncReport rep{};
                rep.instance = static_cast<WorldId>(i);
                rep.tick = report.tick;
                rep.clientDigest = report.digest;
                rep.source = report.source;
                if (const auto &ring = _impl->hashes[i])
                {
                    const auto slot = static_cast<core::usize>(report.tick % kStateHashHistory);
                    rep.serverDigest = ring->digests[slot];
                }
                _impl->lastDesync = rep;
                _impl->hasDesync = true;
                break;
            }

            case DesyncVerdict::TickUnknown:
                // Normal for a very late report, or a client ahead of us.
                ++_impl->staleReportCount;
                break;

            case DesyncVerdict::Match:
                ++_impl->matchedReportCount;
                break;
            }
        }
    }
}

core::u64 Server::desyncCount() const noexcept { return _impl->desyncCount; }

core::u64 Server::matchedReportCount() const noexcept { return _impl->matchedReportCount; }

core::u64 Server::staleReportCount() const noexcept { return _impl->staleReportCount; }

bool Server::lastDesyncReport(DesyncReport &out) const noexcept
{
    if (!_impl->hasDesync)
        return false;
    out = _impl->lastDesync;
    return true;
}

core::u64 Server::backpressureEvents() const noexcept { return _impl->backpressureEvents; }

core::u64 Server::lastBackpressureTick() const noexcept { return _impl->lastBackpressureTick; }

const serial::ReplayRecorder *Server::replay(WorldId id) const noexcept
{
    return (id < _impl->replays.size()) ? _impl->replays[id].get() : nullptr;
}

core::u64 Server::stateHash(WorldId id) const noexcept
{
    if (id >= _impl->hashes.size() || !_impl->hashes[id])
        return 0;
    return _impl->hashes[id]->lastDigest;
}

core::u64 Server::currentTick() const noexcept { return _impl->tickCounter; }

Server::DesyncVerdict Server::checkClientHash(WorldId id, core::u64 tick, core::u64 digest) const noexcept
{
    if (id >= _impl->hashes.size() || !_impl->hashes[id])
        return DesyncVerdict::TickUnknown;

    const auto &ring = *_impl->hashes[id];
    const auto slot = static_cast<core::usize>(tick % kStateHashHistory);

    // The slot is reused every kStateHashHistory ticks, so holding the tick
    // number alongside the digest is what distinguishes "the tick the client
    // means" from "a much older tick that happens to land on the same slot".
    if (!ring.filled[slot] || ring.ticks[slot] != tick)
        return DesyncVerdict::TickUnknown;

    return math::StateHash::match(ring.digests[slot], digest) ? DesyncVerdict::Match : DesyncVerdict::Diverged;
}

void Server::run()
{
    _impl->running.store(true, std::memory_order_relaxed);
    const auto dt = static_cast<core::f32>(1.0 / static_cast<core::f64>(_impl->config.tickRate()));

    while (_impl->running.load(std::memory_order_relaxed))
        tick(dt);
}

void Server::requestShutdown() noexcept { _impl->running.store(false, std::memory_order_relaxed); }

void Server::shutdown()
{
    if (!_impl->initialised)
        return;

    core::Log::info("Server: shutting down");

    // Stop the workers before the instances they step are destroyed.
    _impl->jobs.reset();

    for (auto &world : _impl->worlds)
    {
        if (world)
            world->onShutdown();
    }
    // Same ordering rule as removeWorld: the systems die with their World, and
    // they reference everything cleared afterwards.
    _impl->worlds.clear();
    _impl->queues.clear();
    _impl->sessions.clear();
    for (auto &input : _impl->inputs)
    {
        if (input)
            input->shutdown();
    }
    _impl->inputs.clear();
    _impl->physicsBackends.clear();
    _impl->hashes.clear();
    _impl->replays.clear();
    _impl->routing.clear();
    _impl->defaultWorld = kInvalidWorldId;

    if (_impl->transport)
        _impl->transport->close();

    _impl->initialised = false;
}

ResourceManager &Server::resources() noexcept { return _impl->resources; }

} // namespace lpl::engine

#endif // LPL_HAS_NET
