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

#include <lpl/render/VulkanRenderer.hpp>
#include <GLFW/glfw3.h>

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

    std::unique_ptr<render::IRenderer> renderer;
    GLFWwindow* window{nullptr};

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

    if (_impl->config.enableGpu())
    {
        core::Log::info("Engine: Booting GPU Renderer");

        if (!glfwInit())
        {
            core::Log::error("Failed to initialize GLFW");
            return core::makeError(core::ErrorCode::kGpuInitFailed, "Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        _impl->window = glfwCreateWindow(800, 600, "LplPlugin Client", nullptr, nullptr);

        _impl->renderer = std::make_unique<render::vk::VulkanRenderer>();
        if (auto res = _impl->renderer->init(800, 600); !res)
        {
            core::Log::error("Failed to initialize VulkanRenderer");
            return res;
        }
        
        // Let the Vulkan Renderer bind to the GLFW Window
        auto& vkRenderer = static_cast<render::vk::VulkanRenderer&>(*_impl->renderer);
        vkRenderer.initVulkanContext(_impl->window);
    }

    _impl->initialised = true;
    core::Log::info("Engine::init — done");
    return {};
}

void Engine::run()
{
    LPL_ASSERT(_impl->initialised);

    if (auto res = _impl->scheduler.buildGraph(); !res)
    {
        core::Log::fatal("Failed to build ECS system graph: {}", res.error().message());
        std::abort();
    }

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

    callbacks.render = [this](core::f64 /*alpha*/)
    {
        if (_impl->renderer)
        {
            _impl->renderer->beginFrame();
            _impl->renderer->endFrame(); 
        }

        if (_impl->window && glfwWindowShouldClose(_impl->window))
        {
            requestShutdown();
        }
    };

    callbacks.postFrame = [this]()
    {
        if (_impl->window)
        {
            glfwPollEvents();
        }
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
    
    if (_impl->renderer)
    {
        _impl->renderer->shutdown();
        _impl->renderer.reset();
    }

    if (_impl->window)
    {
        glfwDestroyWindow(_impl->window);
        _impl->window = nullptr;
    }
    glfwTerminate();

    _impl->inputManager.shutdown();
    _impl->initialised = false;
}

const Config& Engine::config() const noexcept
{
    return _impl->config;
}

} // namespace lpl::engine
