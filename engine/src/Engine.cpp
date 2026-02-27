/**
 * @file Engine.cpp
 * @brief Engine façade implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/engine/Engine.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/engine/GameLoop.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/concurrency/ThreadPool.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/input/InputManager.hpp>

#include <lpl/net/session/SessionManager.hpp>
#include <lpl/net/ServerMesh.hpp>
#include <lpl/net/transport/KernelTransport.hpp>
#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/net/netcode/AuthoritativeStrategy.hpp>
#include <lpl/net/netcode/RollbackStrategy.hpp>

namespace lpl::engine {

struct Engine::Impl
{
    Config config;
    GameLoop loop;

    memory::ArenaAllocator arena;
    concurrency::ThreadPool threadPool;
    ecs::Registry registry;
    ecs::SystemScheduler scheduler;
    input::InputManager inputManager;

    std::unique_ptr<net::session::SessionManager> sessionManager;
    std::unique_ptr<ecs::WorldPartition> world;
    std::unique_ptr<net::netcode::INetcodeStrategy> netcode;
    std::shared_ptr<net::transport::ITransport> transport;

    bool initialised{false};

    explicit Impl(Config cfg)
        : config{std::move(cfg)}
        , loop{config}
        , arena{config.arenaSize()}
        , threadPool{8}
        , registry{}
        , scheduler{threadPool}
        , inputManager{}
    {
    }
};

Engine::Engine(Config config)
    : _impl{std::make_unique<Impl>(std::move(config))}
{
}

Engine::~Engine()
{
    if (_impl && _impl->initialised)
    {
        shutdown();
    }
}

core::Expected<void> Engine::init()
{
    core::Log::info("Engine::init — wiring subsystems");

    auto inputResult = _impl->inputManager.init();
    if (!inputResult)
    {
        return inputResult;
    }

    _impl->world = std::make_unique<ecs::WorldPartition>(math::Fixed32{10});

    if (_impl->config.serverMode())
    {
        core::Log::info("Engine: Booting Server");
        _impl->transport = std::make_shared<net::transport::KernelTransport>("/dev/lpl0");
        if (!_impl->transport->open())
        {
            core::Log::warn("Failed to open /dev/lpl0, server networking disabled");
        }
        _impl->sessionManager = std::make_unique<net::session::SessionManager>();
        _impl->netcode = std::make_unique<net::netcode::AuthoritativeStrategy>();
    }
    else
    {
        core::Log::info("Engine: Booting Client");
        _impl->transport = std::make_shared<net::transport::KernelTransport>("/dev/lpl0");
        if (!_impl->transport->open())
        {
            core::Log::warn("Failed to open kernel transport for client");
        }
        _impl->sessionManager = std::make_unique<net::session::SessionManager>();
        _impl->netcode = std::make_unique<net::netcode::RollbackStrategy>(8);
    }

    _impl->initialised = true;
    core::Log::info("Engine::init — done");
    return {};
}

void Engine::run()
{
    LPL_ASSERT(_impl->initialised);

    LoopCallbacks callbacks;

    callbacks.preFrame = [this]()
    {
        [[maybe_unused]] auto r = _impl->inputManager.poll();
    };

    callbacks.fixedUpdate = [this](core::f64 dt)
    {
        // Transport polling would happen here
        if (_impl->netcode)
        {
            _impl->netcode->tick(static_cast<core::f32>(dt));
        }
        
        // Physics and general simulation
        _impl->scheduler.tick(static_cast<core::f32>(dt));
    };

    callbacks.render = [](core::f64 /*alpha*/)
    {
    };

    callbacks.postFrame = []()
    {
    };

    _impl->loop.run(callbacks);
}

void Engine::requestShutdown() noexcept
{
    _impl->loop.requestStop();
}

void Engine::shutdown()
{
    if (!_impl->initialised)
    {
        return;
    }

    core::Log::info("Engine::shutdown");
    
    if (_impl->transport)
    {
        _impl->transport->close();
    }
    
    _impl->inputManager.shutdown();
    _impl->initialised = false;
}

const Config& Engine::config() const noexcept
{
    return _impl->config;
}

} // namespace lpl::engine
