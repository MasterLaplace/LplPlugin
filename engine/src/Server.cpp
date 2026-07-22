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
#    include <lpl/memory/ArenaAllocator.hpp>
#    include <lpl/net/netcode/AuthoritativeStrategy.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/KernelTransport.hpp>
#    include <lpl/net/transport/SocketTransport.hpp>
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

    lpl::pmr::unique_ptr<net::session::SessionManager> sessionManager;
    lpl::pmr::unique_ptr<net::netcode::INetcodeStrategy> netcode;
    std::shared_ptr<net::transport::ITransport> transport;

    /// The World hooks need host services; the server has no display or input,
    /// so it presents a headless platform to them.
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

    _impl->sessionManager = lpl::pmr::make_unique<net::session::SessionManager>();
    _impl->netcode = lpl::pmr::make_unique<net::netcode::AuthoritativeStrategy>();
    _impl->initialised = true;

    core::Log::info("Server: ready");
    return {};
}

Server::WorldId Server::addWorld(lpl::pmr::unique_ptr<World> world)
{
    if (!world)
        return kInvalidWorldId;

    // No Engine hosts this World — the server does — so the context carries a
    // null engine handle, which WorldContext documents as legitimate.
    WorldContext context{_impl->platform, _impl->resources, _impl->arena, _impl->config, nullptr};

    if (auto res = world->onInit(context); !res)
    {
        core::Log::error("Server: instance failed to initialise, refused");
        return kInvalidWorldId;
    }

    _impl->worlds.push_back(std::move(world));
    const auto id = static_cast<WorldId>(_impl->worlds.size() - 1u);
    core::Log::info("Server: instance hosted");
    return id;
}

bool Server::removeWorld(WorldId id)
{
    if (id >= _impl->worlds.size() || !_impl->worlds[id])
        return false;

    _impl->worlds[id]->onShutdown();
    _impl->worlds[id].reset();
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

void Server::tick(core::f32 dt)
{
    if (_impl->netcode)
        _impl->netcode->tick(dt);

    // Instances are independent: each owns its registry and scheduler, so this
    // is the loop a thread pool would fan out once instances are ticked in
    // parallel (the shared ResourceManager is already thread-safe for that).
    for (auto &world : _impl->worlds)
    {
        if (world)
            world->onFixedStep(dt);
    }
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

    for (auto &world : _impl->worlds)
    {
        if (world)
            world->onShutdown();
    }
    _impl->worlds.clear();

    if (_impl->transport)
        _impl->transport->close();

    _impl->initialised = false;
}

ResourceManager &Server::resources() noexcept { return _impl->resources; }

} // namespace lpl::engine

#endif // LPL_HAS_NET
