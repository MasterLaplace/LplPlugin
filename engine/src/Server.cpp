#include <lpl/engine/Server.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/concurrency/JobSystem.hpp>
#    include <lpl/core/Log.hpp>
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
#    include <lpl/net/protocol/Protocol.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/KernelTransport.hpp>
#    include <lpl/net/transport/SocketTransport.hpp>
#    include <lpl/physics/CpuPhysicsBackend.hpp>
#    include <lpl/serial/ReplayRecorder.hpp>
#    include <lpl/serial/StateSnapshot.hpp>
#    include <lpl/std/unordered_map.hpp>

#    include <array>
#    include <lpl/platform/linux/LinuxPlatform.hpp>

#    include <atomic>
#    include <memory>

namespace lpl::engine {

namespace {

/**
 * @brief Packs an endpoint into the key the routing table is indexed by.
 */
[[nodiscard]] core::u64 senderKey(const net::Endpoint &endpoint) noexcept
{
    return (static_cast<core::u64>(endpoint.address()) << 16) | static_cast<core::u64>(endpoint.port());
}

} // namespace

struct Server::Impl {
    Config config;

    /**
     * Shared by reference with every hosted World: one load per asset for the
     * whole server, not one per instance.
     */
    ResourceManager resources;

    /**
     * Instance slots. A removed instance leaves a null hole rather than
     * compacting, so live ids stay stable for the server's lifetime.
     */
    lpl::pmr::vector<lpl::pmr::unique_ptr<World>> worlds;

    /**
     * One set of queues per instance, parallel to @c worlds: a packet is decoded
     * into the queues of the instance its sender belongs to, so one game's
     * inputs can never be seen by another's systems.
     */
    lpl::pmr::vector<lpl::pmr::unique_ptr<EventQueues>> queues;

    /**
     * Sender -> instance. Keyed by the endpoint packed into 48 bits (IPv4 +
     * port), which is exactly what identifies a UDP peer.
     */
    lpl::pmr::unordered_map<core::u64, WorldId> routing;

    WorldId defaultWorld{kInvalidWorldId}; /**< Where a sender we have never seen is placed. */

    /**
     * One set of connected clients per instance, parallel to @c worlds. NOT
     * shared: broadcastState() walks a whole manager, so one manager for all
     * instances would leak game A's state to game B's players.
     */
    lpl::pmr::vector<lpl::pmr::unique_ptr<net::session::SessionManager>> sessions;

    /**
     * One input state per instance, parallel to @c worlds. NOT shared either:
     * InputProcessingSystem keys input by ENTITY id and MovementSystem reads it
     * back from the same map, but entity ids are minted per registry — so two
     * instances would both own an entity 1, and game A's steering would drive
     * game B's player. Same failure mode as a shared SessionManager, on the
     * inbound path instead of the outbound one.
     */
    lpl::pmr::vector<lpl::pmr::unique_ptr<input::InputManager>> inputs;

    /**
     * One physics backend per instance, parallel to @c worlds: a backend is
     * bound to the registry it integrates, so it cannot be shared.
     */
    lpl::pmr::vector<lpl::pmr::unique_ptr<physics::IPhysicsBackend>> physicsBackends;

    lpl::pmr::unique_ptr<net::netcode::INetcodeStrategy> netcode;
    std::shared_ptr<net::transport::ITransport> transport;

    /**
     * Rolling digest history per instance, parallel to @c worlds: a ring of
     * kStateHashHistory entries indexed by (tick % size). A client reports a
     * digest for a tick we stepped a few frames ago, so the current digest is
     * useless for the comparison — we need the one it actually refers to.
     */
    struct HashRing {
        core::u64 ticks[Server::kStateHashHistory]{};
        core::u64 digests[Server::kStateHashHistory]{};
        bool filled[Server::kStateHashHistory]{};
        core::u64 lastDigest{0};
    };
    lpl::pmr::vector<lpl::pmr::unique_ptr<HashRing>> hashes;

    core::u64 tickCounter{0}; /**< Ticks stepped since the server started; the timeline digests are keyed by. */

    /**
     * One replay recorder per instance, parallel to @c worlds: periodic
     * snapshots kept so a divergence can be diagnosed after the fact (§6.4.2).
     */
    lpl::pmr::vector<lpl::pmr::unique_ptr<serial::ReplayRecorder>> replays;

    /**
     * @name Desync telemetry, cumulative over the server's life
     * @{
     */
    core::u64 desyncCount{0};
    core::u64 matchedReportCount{0};
    core::u64 staleReportCount{0};
    /** @} */

    Server::DesyncReport lastDesync{}; /**< The most recent captured divergence, for post-mortem diagnosis (§6.4.2). */
    bool hasDesync{false};

    /**
     * Reused receive scratch: one contiguous block sliced into per-slot buffers,
     * so draining the socket never allocates on the hot path.
     */
    std::vector<core::byte> receiveStorage;
    std::vector<net::transport::ReceiveSlot> receiveSlots;

    /**
     * How many ticks hit the per-tick receive budget with the socket still
     * full — i.e. the server did not keep up and the kernel is now shedding
     * load. The one number that makes an otherwise invisible wall visible.
     */
    core::u64 backpressureEvents{0};
    core::u64 lastBackpressureTick{0};

    /**
     * Fans the per-instance ticks out across threads. Null when
     * Config::serverWorkerThreads is 0, which ticks instances sequentially.
     */
    lpl::pmr::unique_ptr<concurrency::JobSystem> jobs;

    /**
     * The World hooks need host services; the server has no display or input,
     * so it presents a headless platform to them.
     */
    platform::linux_host::LinuxPlatform platform;
    memory::ArenaAllocator arena;

    std::atomic<bool> running{false};
    bool initialised{false};

    explicit Impl(Config cfg) : config{std::move(cfg)}, arena{config.arenaSize()} {}
};

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

    if (!_impl->initialised)
    {
        core::Log::error("Server: addWorld called before init(), refused");
        return kInvalidWorldId;
    }

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

    if (auto built = _impl->worlds[id]->build(); !built)
    {
        core::Log::error("Server: instance schedule has a dependency cycle, refused");
        removeWorld(id);
        return kInvalidWorldId;
    }

    if (_impl->defaultWorld == kInvalidWorldId)
        _impl->defaultWorld = id;

    core::Log::info("Server: instance hosted");
    return id;
}

void Server::registerInstanceSystems(WorldId id)
{
    World &world = *_impl->worlds[id];
    auto &scheduler = world.scheduler();

    ecs::WorldPartition *spatial = world.spatialPartition();
    if (spatial == nullptr)
        spatial = &world.enableSpatialPartition(math::Fixed32::fromFloat(10.0f), _impl->config.worldCellCapacity());

    {
        auto session = lpl::pmr::make_unique<systems::SessionSystem>(
            *_impl->sessions[id], *_impl->queues[id], _impl->transport, *_impl->inputs[id], *spatial, world.registry(),
            _impl->config.sessionTimeoutMs());
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

    if (_impl->config.enablePhysics())
    {
        _impl->physicsBackends[id] = lpl::pmr::make_unique<physics::CpuPhysicsBackend>(world.registry());
        [[maybe_unused]] auto initRes = _impl->physicsBackends[id]->init();

        auto physics =
            lpl::pmr::make_unique<systems::PhysicsSystem>(*spatial, *_impl->physicsBackends[id], world.registry());
        [[maybe_unused]] auto r = scheduler.registerSystem(std::move(physics));
    }

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
    _impl->worlds[id].reset();
    _impl->queues[id].reset();
    _impl->sessions[id].reset();
    if (_impl->inputs[id])
        _impl->inputs[id]->shutdown();
    _impl->inputs[id].reset();
    _impl->physicsBackends[id].reset();
    _impl->hashes[id].reset();
    _impl->replays[id].reset();

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

    constexpr core::u32 kBurst = 64;
    static constexpr core::usize kSlotSize =
        sizeof(net::protocol::PacketHeader) + net::session::SessionManager::kMaxPayloadSize;

    auto &storage = _impl->receiveStorage;
    auto &slots = _impl->receiveSlots;
    if (storage.empty())
    {
        storage.resize(static_cast<core::usize>(kBurst) * kSlotSize);
        slots.resize(kBurst);
        for (core::u32 i = 0; i < kBurst; ++i)
            slots[i].buffer =
                std::span<core::byte>{storage.data() + static_cast<core::usize>(i) * kSlotSize, kSlotSize};
    }

    const core::u32 budget = _impl->config.maxPacketsPerTick();
    core::u32 drained = 0;

    while (drained < budget)
    {
        for (auto &slot : slots)
            slot.length = 0;

        auto result =
            _impl->transport->receiveBatch(std::span<net::transport::ReceiveSlot>{slots.data(), slots.size()});
        if (!result.has_value())
            break;

        const core::u32 count = result.value();
        if (count == 0)
            break;

        for (core::u32 i = 0; i < count; ++i)
        {
            const auto &slot = slots[i];
            if (slot.length == 0)
                continue;

            net::protocol::PacketHeader header{};
            std::span<const core::byte> payload;
            if (!detail::parsePacket(std::span<const core::byte>{slot.buffer.data(), slot.length}, header, payload))
                continue;

            auto worldId = worldForSender(slot.source);
            if (worldId == kInvalidWorldId)
            {
                worldId = _impl->defaultWorld;
                if (worldId == kInvalidWorldId)
                    continue;
                _impl->routing[senderKey(slot.source)] = worldId;
            }

            if (auto *destination = queues(worldId))
                detail::dispatchPacket(header, payload, slot.source, *destination);
        }

        drained += count;

        if (count < slots.size())
            break;
    }

    if (drained >= budget)
    {
        ++_impl->backpressureEvents;
        _impl->lastBackpressureTick = _impl->tickCounter + 1;
    }
}

void Server::tick(core::f32 dt)
{
    pumpNetwork();

    consumeStateHashReports();

    if (_impl->netcode)
        _impl->netcode->tick(dt);

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
                core::Log::error("Server: client desync detected, its state diverged from ours");
                ++_impl->desyncCount;

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

            case DesyncVerdict::TickUnknown: ++_impl->staleReportCount; break;

            case DesyncVerdict::Match: ++_impl->matchedReportCount; break;
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

    _impl->jobs.reset();

    for (auto &world : _impl->worlds)
    {
        if (world)
            world->onShutdown();
    }
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
