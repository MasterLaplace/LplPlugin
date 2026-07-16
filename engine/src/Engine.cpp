/**
 * @file Engine.cpp
 * @brief Engine façade implementation.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/engine/Engine.hpp>
#include <lpl/engine/GameLoop.hpp>

#include <lpl/concurrency/IJobSystem.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/input/InputManager.hpp>
#include <lpl/memory/ArenaAllocator.hpp>

#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/net/ServerMesh.hpp>
#include <lpl/net/netcode/AuthoritativeStrategy.hpp>
#include <lpl/net/netcode/RollbackStrategy.hpp>
#include <lpl/net/session/SessionManager.hpp>
#include <lpl/net/transport/KernelTransport.hpp>
#include <lpl/net/transport/SocketTransport.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/physics/IPhysicsBackend.hpp>
#ifdef LPL_HAS_CUDA
#    include <lpl/gpu/CudaBackend.hpp>
#    include <lpl/physics/GpuPhysicsBackend.hpp>
#endif

#include <lpl/platform/IPlatform.hpp>
#include <lpl/platform/linux/LinuxPlatform.hpp>

#include <lpl/engine/EventQueue.hpp>
#include <lpl/engine/systems/BroadcastSystem.hpp>
#include <lpl/engine/systems/CameraSystem.hpp>
#include <lpl/engine/systems/InputProcessingSystem.hpp>
#include <lpl/engine/systems/InputSendSystem.hpp>
#include <lpl/engine/systems/LocalInputSystem.hpp>
#include <lpl/engine/systems/MovementSystem.hpp>
#include <lpl/engine/systems/NetworkReceiveSystem.hpp>
#include <lpl/engine/systems/PhysicsSystem.hpp>
#include <lpl/engine/systems/RenderSystem.hpp>
#include <lpl/engine/systems/ServerMonitorSystem.hpp>
#include <lpl/engine/systems/SessionSystem.hpp>
#include <lpl/engine/systems/SpawnSystem.hpp>
#include <lpl/engine/systems/StateReconciliationSystem.hpp>
#include <lpl/engine/systems/WelcomeSystem.hpp>
#include <lpl/net/protocol/PacketBuilder.hpp>

#include <arpa/inet.h>

#ifdef LPL_HAS_RENDERER
#    include <GLFW/glfw3.h>
#    include <lpl/render/VulkanRenderer.hpp>
#endif

#include <lpl/bci/BciAdapter.hpp>
#include <lpl/bci/SourceBciDriver.hpp>
#include <lpl/bci/source/SyntheticSource.hpp>

namespace lpl::engine {

struct Engine::Impl {
    Config config;
    std::unique_ptr<platform::IPlatform> platform;
    GameLoop loop;

    memory::ArenaAllocator arena;
    concurrency::InlineJobSystem jobSystem;
    ecs::Registry registry;
    ecs::SystemScheduler scheduler;
    input::InputManager inputManager;
    core::CommandQueue commandQueue;

    std::unique_ptr<net::session::SessionManager> sessionManager;
    std::unique_ptr<ecs::WorldPartition> world;
    std::unique_ptr<net::netcode::INetcodeStrategy> netcode;
    std::shared_ptr<net::transport::ITransport> transport;

    EventQueues eventQueues;
    std::unique_ptr<physics::IPhysicsBackend> physicsBackend;
#ifdef LPL_HAS_CUDA
    std::unique_ptr<gpu::IComputeBackend> computeBackend; // owns the GPU dispatch backend
#endif

    // Client-side state (set by WelcomeSystem)
    core::u32 myEntityId{0};
    bool connected{false};
    systems::CameraData cameraData{};

#ifdef LPL_HAS_RENDERER
    std::unique_ptr<render::IRenderer> renderer;
    GLFWwindow *window{nullptr};
#endif

    bool initialised{false};

    std::unique_ptr<bci::BciAdapter> bciAdapter;

    Impl(Config cfg, std::unique_ptr<platform::IPlatform> plat)
        : config{std::move(cfg)}, platform{std::move(plat)}, loop{config, platform->clock()}, arena{config.arenaSize()},
          jobSystem{}, registry{}, scheduler{jobSystem}, inputManager{}
    {
    }
};

Engine::Engine(Config config)
    : _impl{std::make_unique<Impl>(std::move(config), std::make_unique<platform::linux_host::LinuxPlatform>())}
{
}

Engine::Engine(Config config, std::unique_ptr<platform::IPlatform> platform)
    : _impl{std::make_unique<Impl>(std::move(config), std::move(platform))}
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

    // Probe the injected platform seam (clock / display / input / GPU-memory),
    // the hosted mirror of the kernel's p2_hal_smoke. The engine reaches every
    // host or kernel facility exclusively through these backends.
    {
        platform::IPlatform &plat = *_impl->platform;
        platform::SurfaceDescriptor surface;
        const bool haveSurface = plat.display().querySurface(surface);
        core::Log::info(haveSurface ? "Engine: platform seam up (surface available)" :
                                      "Engine: platform seam up (headless, no surface)");
    }

    auto inputResult = _impl->inputManager.init();
    if (!inputResult)
    {
        return inputResult;
    }

    _impl->world = std::make_unique<ecs::WorldPartition>(math::Fixed32{10});

    if (_impl->config.serverMode())
    {
        core::Log::info("Engine: Booting Server");

        // Try kernel transport first, fallback to socket if unavailable
        auto kernelTransport = std::make_shared<net::transport::KernelTransport>("/dev/lpl0");
        if (auto res = kernelTransport->open(); res)
        {
            core::Log::info("Engine: Using KernelTransport (/dev/lpl0)");
            _impl->transport = std::move(kernelTransport);
        }
        else
        {
            core::Log::warn("Engine", "Kernel module unavailable, falling back to SocketTransport");
            auto socketTransport = std::make_shared<net::transport::SocketTransport>(_impl->config.serverPort());
            if (auto res2 = socketTransport->open(); !res2)
            {
                core::Log::error("Engine: Failed to open SocketTransport fallback");
                return res2;
            }
            _impl->transport = std::move(socketTransport);
        }

        _impl->sessionManager = std::make_unique<net::session::SessionManager>();
        _impl->netcode = std::make_unique<net::netcode::AuthoritativeStrategy>();
    }
    else
    {
        core::Log::info("Engine: Booting Client");

        // Client uses socket transport by default (same-machine dev friendly)
        auto socketTransport = std::make_shared<net::transport::SocketTransport>(0);
        if (auto res = socketTransport->open(); !res)
        {
            core::Log::error("Engine: Failed to open client SocketTransport");
            return res;
        }
        core::Log::info("Engine: Using SocketTransport (client default)");

        // Send connect handshake to server and register default destination
        // so that InputSendSystem can call sendInputs(transport, nullptr, ...)
        // without carrying the server address.
        {
            sockaddr_in serverAddr{};
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_port = htons(_impl->config.serverPort());
            inet_pton(AF_INET, _impl->config.serverAddress().c_str(), &serverAddr.sin_addr);

            if (auto res = net::protocol::sendConnect(*socketTransport, &serverAddr); !res)
            {
                core::Log::warn("Engine", "Connect handshake failed, server may not be running yet");
            }
            else
            {
                core::Log::info("Engine: Connect handshake sent to server");
            }

            // Pre-register server address as default destination for all future sends.
            socketTransport->setDefaultDest(&serverAddr);
        }

        _impl->transport = std::move(socketTransport);

        _impl->sessionManager = std::make_unique<net::session::SessionManager>();
        _impl->netcode = std::make_unique<net::netcode::RollbackStrategy>(8);
    }

#ifdef LPL_HAS_RENDERER
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
        auto &vkRenderer = static_cast<render::vk::VulkanRenderer &>(*_impl->renderer);
        vkRenderer.initVulkanContext(_impl->window);
    }
#else
    if (_impl->config.enableGpu())
    {
        core::Log::warn("Engine: GPU rendering requested but LPL_HAS_RENDERER is not enabled (headless build)");
    }
#endif

    // ------------------------------------------------------------------ //
    //  Register ECS systems                                               //
    // ------------------------------------------------------------------ //

    core::Log::info("Engine: Registering ECS systems");

    // Shared systems (both server and client)
    {
        auto netRecv = std::make_unique<systems::NetworkReceiveSystem>(_impl->transport, _impl->eventQueues);
        [[maybe_unused]] auto r1 = _impl->scheduler.registerSystem(std::move(netRecv));

        // Select the physics backend. The CPU backend is the deterministic
        // reference (and the only one compiled into the freestanding kernel).
        // When the GPU module is enabled (Config::enableGpu) and a CUDA build is
        // available, the GpuPhysicsBackend bridge drives the physics_tick compute
        // kernel instead; collision/sleeping stay on the CPU. The selection is
        // behind LPL_HAS_CUDA so non-CUDA hosts never reference the bridge.
#ifdef LPL_HAS_CUDA
        if (_impl->config.enableGpu())
        {
            _impl->computeBackend = std::make_unique<gpu::CudaBackend>();
            _impl->physicsBackend =
                std::make_unique<physics::GpuPhysicsBackend>(_impl->registry, *_impl->computeBackend);
        }
        else
#endif
        {
            _impl->physicsBackend = std::make_unique<physics::CpuPhysicsBackend>(_impl->registry);
        }
        [[maybe_unused]] auto initRes = _impl->physicsBackend->init();

        auto physics = std::make_unique<systems::PhysicsSystem>(*_impl->world, *_impl->physicsBackend, _impl->registry);
        [[maybe_unused]] auto r2 = _impl->scheduler.registerSystem(std::move(physics));
    }

    if (_impl->config.serverMode())
    {
        auto session =
            std::make_unique<systems::SessionSystem>(*_impl->sessionManager, _impl->eventQueues, _impl->transport,
                                                     _impl->inputManager, *_impl->world, _impl->registry);
        [[maybe_unused]] auto r1 = _impl->scheduler.registerSystem(std::move(session));

        auto inputProc = std::make_unique<systems::InputProcessingSystem>(_impl->eventQueues, _impl->inputManager);
        [[maybe_unused]] auto r2 = _impl->scheduler.registerSystem(std::move(inputProc));

        auto movement = std::make_unique<systems::MovementSystem>(_impl->inputManager, _impl->registry);
        [[maybe_unused]] auto r3 = _impl->scheduler.registerSystem(std::move(movement));

        auto broadcast = std::make_unique<systems::BroadcastSystem>(*_impl->sessionManager, _impl->transport,
                                                                    *_impl->world, _impl->registry);
        [[maybe_unused]] auto r4 = _impl->scheduler.registerSystem(std::move(broadcast));

        auto monitor = std::make_unique<systems::ServerMonitorSystem>(*_impl->sessionManager, *_impl->world);
        [[maybe_unused]] auto r5 = _impl->scheduler.registerSystem(std::move(monitor));
    }
    else
    {
        auto welcome =
            std::make_unique<systems::WelcomeSystem>(_impl->eventQueues, _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r1 = _impl->scheduler.registerSystem(std::move(welcome));

        auto reconcile =
            std::make_unique<systems::StateReconciliationSystem>(_impl->eventQueues, *_impl->world, _impl->registry);
        [[maybe_unused]] auto r2 = _impl->scheduler.registerSystem(std::move(reconcile));

        auto spawn = std::make_unique<systems::SpawnSystem>(_impl->registry, _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r3 = _impl->scheduler.registerSystem(std::move(spawn));

#ifdef LPL_HAS_RENDERER
        auto localInput = std::make_unique<systems::LocalInputSystem>(_impl->inputManager, _impl->window,
                                                                      _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r4 = _impl->scheduler.registerSystem(std::move(localInput));
#endif

        auto movement = std::make_unique<systems::MovementSystem>(_impl->inputManager, _impl->registry);
        [[maybe_unused]] auto r5 = _impl->scheduler.registerSystem(std::move(movement));

        auto inputSend = std::make_unique<systems::InputSendSystem>(_impl->inputManager, _impl->transport,
                                                                    _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r6 = _impl->scheduler.registerSystem(std::move(inputSend));

#ifdef LPL_HAS_RENDERER
        auto camera = std::make_unique<systems::CameraSystem>(_impl->cameraData, _impl->registry, _impl->window,
                                                              _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r7 = _impl->scheduler.registerSystem(std::move(camera));

        auto render = std::make_unique<systems::RenderSystem>(_impl->registry, _impl->renderer.get());
        [[maybe_unused]] auto r8 = _impl->scheduler.registerSystem(std::move(render));
#endif
    }

    // ------------------------------------------------------------------ //
    //  BCI subsystem (client-only, when enableBci is set)               //
    // ------------------------------------------------------------------ //

    if (_impl->config.enableBci() && !_impl->config.serverMode())
    {
        core::Log::info("Engine: Initialising BCI adapter (SyntheticSource)");
        auto source = std::make_unique<bci::source::SyntheticSource>(42, true);
        auto driver = std::make_unique<bci::SourceBciDriver>(std::move(source));
        _impl->bciAdapter = std::make_unique<bci::BciAdapter>(std::move(driver));
        if (auto res = _impl->bciAdapter->start(); !res)
        {
            core::Log::warn("Engine", "BCI adapter failed to start, continuing without BCI");
            _impl->bciAdapter.reset();
        }
    }

    // ------------------------------------------------------------------ //
    //  Spawn initial entities (server: 50 NPCs with deterministic seed)  //
    // ------------------------------------------------------------------ //

    if (_impl->config.serverMode())
    {
        core::Log::info("Engine: Spawning initial NPC entities");

        // Build the NPC archetype: Position, Velocity, Mass, AABB, Health
        ecs::Archetype npcArch;
        npcArch.add(ecs::ComponentId::Position);
        npcArch.add(ecs::ComponentId::Velocity);
        npcArch.add(ecs::ComponentId::Mass);
        npcArch.add(ecs::ComponentId::AABB);
        npcArch.add(ecs::ComponentId::Health);

        // Simple deterministic LCG (seed 42) for reproducible NPC placement.
        // IDs 0–99 are reserved for NPCs; player entities start at 100.
        core::u32 seed = 42;
        auto nextRand = [&seed]() -> float {
            seed = seed * 1103515245u + 12345u;
            return static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
        };

        static constexpr core::u32 kNpcCount = 50;
        for (core::u32 i = 0; i < kNpcCount; ++i)
        {
            // Create entity in Registry with proper SoA component storage
            auto entityResult = _impl->registry.createEntity(npcArch);
            if (!entityResult.has_value())
                continue;

            auto entityId = entityResult.value();
            auto refResult = _impl->registry.resolve(entityId);
            if (!refResult.has_value())
                continue;

            auto ref = refResult.value();
            auto &partition = _impl->registry.getOrCreatePartition(npcArch);
            const auto &chunks = partition.chunks();
            if (ref.chunkIndex >= static_cast<core::u32>(chunks.size()))
                continue;

            auto &chunk = *chunks[ref.chunkIndex];

            float px = (nextRand() - 0.5f) * 200.0f; // [-100, 100]
            float py = nextRand() * 50.0f;           // [0, 50]
            float pz = (nextRand() - 0.5f) * 200.0f; // [-100, 100]

            // Write position to both front and back buffers (authoritative Fixed32)
            math::Vec3<math::Fixed32> pos{math::Fixed32::fromFloat(px), math::Fixed32::fromFloat(py),
                                          math::Fixed32::fromFloat(pz)};
            if (auto *wpos = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position)))
            {
                wpos[ref.localIndex] = pos;
            }
            if (auto *rpos = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::Position))))
            {
                rpos[ref.localIndex] = pos;
            }

            // Write velocity (zero initially)
            math::Vec3<math::Fixed32> vel{math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::zero()};
            if (auto *wvel = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
            {
                wvel[ref.localIndex] = vel;
            }

            // Write mass
            math::Fixed32 mass = math::Fixed32::one();
            if (auto *wmass = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Mass)))
            {
                wmass[ref.localIndex] = mass;
            }
            if (auto *rmass = const_cast<math::Fixed32 *>(
                    static_cast<const math::Fixed32 *>(chunk.readComponent(ecs::ComponentId::Mass))))
            {
                rmass[ref.localIndex] = mass;
            }

            // Write AABB (size)
            math::Vec3<math::Fixed32> size{math::Fixed32::fromInt(1), math::Fixed32::fromInt(2),
                                           math::Fixed32::fromInt(1)};
            if (auto *wsize = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
            {
                wsize[ref.localIndex] = size;
            }
            if (auto *rsize = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::AABB))))
            {
                rsize[ref.localIndex] = size;
            }

            // Write health
            core::i32 hp = 100;
            if (auto *whp = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
            {
                whp[ref.localIndex] = hp;
            }
            if (auto *rhp = const_cast<core::i32 *>(
                    static_cast<const core::i32 *>(chunk.readComponent(ecs::ComponentId::Health))))
            {
                rhp[ref.localIndex] = hp;
            }

            // Update spatial index
            auto fixedPos = math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(px), math::Fixed32::fromFloat(py),
                                                      math::Fixed32::fromFloat(pz)};
            [[maybe_unused]] auto res = _impl->world->insertOrUpdate(entityId, fixedPos);
        }
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
        core::Log::fatal("Failed to build ECS system graph");
        std::abort();
    }

    // Insert buffer swap between Physics and Network phases so that
    // Broadcast/Render systems read the freshly-computed physics data
    // (mirrors legacy PreSwap → swapBuffers → PostSwap pattern).
    _impl->scheduler.setPhaseCallback(ecs::SchedulePhase::Physics, [this]() { _impl->registry.swapAllBuffers(); });

    LoopCallbacks callbacks;

    callbacks.preFrame = [this]() {
        _impl->commandQueue.flush();
        _impl->arena.reset();

        [[maybe_unused]] auto r = _impl->inputManager.poll();

        // Poll BCI adapter and feed neural data into InputManager
        if (_impl->bciAdapter)
        {
            if (auto result = _impl->bciAdapter->update(); result.has_value())
            {
                const auto &neural = result.value();
                _impl->inputManager.setNeural(_impl->myEntityId,
                                              neural.channels[0].toFloat(),       // alpha
                                              neural.channels[1].toFloat(),       // beta
                                              neural.channels[2].toFloat(),       // concentration
                                              neural.channels[3].toFloat() > 0.5f // blink
                );
            }
        }
    };

    callbacks.fixedUpdate = [this](core::f64 dt) {
        if (_impl->netcode)
        {
            _impl->netcode->tick(static_cast<core::f32>(dt));
        }

        // Run all ECS systems (Input → PrePhysics → Physics → [swap] → Network)
        _impl->scheduler.tick(static_cast<core::f32>(dt));
    };

    callbacks.render = [this](core::f64 /*alpha*/) {
#ifdef LPL_HAS_RENDERER
        if (_impl->renderer)
        {
            _impl->renderer->beginFrame();
            _impl->renderer->endFrame();
        }

        if (_impl->window && glfwWindowShouldClose(_impl->window))
        {
            requestShutdown();
        }
#endif
    };

    callbacks.postFrame = [this]() {
#ifdef LPL_HAS_RENDERER
        if (_impl->window)
        {
            glfwPollEvents();
        }
#endif
    };

    _impl->loop.run(callbacks);
}

void Engine::requestShutdown() noexcept { _impl->loop.requestStop(); }

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

    if (_impl->physicsBackend)
    {
        _impl->physicsBackend->shutdown();
    }

#ifdef LPL_HAS_RENDERER
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
#endif

    _impl->inputManager.shutdown();
    _impl->initialised = false;
}

void Engine::submitCommand(std::unique_ptr<core::ICommand> cmd) { _impl->commandQueue.push(std::move(cmd)); }

const Config &Engine::config() const noexcept { return _impl->config; }

} // namespace lpl::engine
