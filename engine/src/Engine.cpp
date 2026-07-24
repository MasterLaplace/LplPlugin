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
#include <lpl/engine/Engine.hpp>
#include <lpl/engine/GameLoop.hpp>
#include <lpl/engine/ResourceManager.hpp>
#include <lpl/engine/World.hpp>
#include <lpl/math/FixedPoint.hpp>

#include <lpl/concurrency/IJobSystem.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/input/InputManager.hpp>
#include <lpl/memory/ArenaAllocator.hpp>

#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/physics/IPhysicsBackend.hpp>
#ifdef LPL_HAS_CUDA
#    include <lpl/gpu/CudaBackend.hpp>
#    include <lpl/physics/GpuPhysicsBackend.hpp>
#endif

#include <lpl/platform/IPlatform.hpp>
#if !LPL_TARGET_KERNEL
#    include <lpl/platform/linux/LinuxPlatform.hpp>
#endif

#include <lpl/engine/EventQueue.hpp>
#include <lpl/engine/systems/MovementSystem.hpp>
#include <lpl/engine/systems/PhysicsSystem.hpp>

// The networked session (transport, sessions, netcode, the systems that feed
// them) is a host feature: it needs a sockets stack. LPL_HAS_NET keeps it out
// of the freestanding kernel build, where net/ is not compiled at all.
#ifdef LPL_HAS_NET
#    include <lpl/engine/systems/AoiBroadcastSystem.hpp>
#    include <lpl/engine/systems/BroadcastSystem.hpp>
#    include <lpl/engine/systems/InputProcessingSystem.hpp>
#    include <lpl/engine/systems/InputSendSystem.hpp>
#    include <lpl/engine/systems/NetworkReceiveSystem.hpp>
#    include <lpl/engine/systems/ServerMonitorSystem.hpp>
#    include <lpl/engine/systems/SessionSystem.hpp>
#    include <lpl/engine/systems/SpawnSystem.hpp>
#    include <lpl/engine/systems/StateHashReportSystem.hpp>
#    include <lpl/engine/systems/StateReconciliationSystem.hpp>
#    include <lpl/engine/systems/WelcomeSystem.hpp>
#    include <lpl/net/ServerMesh.hpp>
#    include <lpl/net/netcode/AuthoritativeStrategy.hpp>
#    include <lpl/net/netcode/RollbackStrategy.hpp>
#    include <lpl/net/protocol/PacketBuilder.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/KernelTransport.hpp>
#    include <lpl/net/transport/SocketTransport.hpp>
#endif

#ifdef LPL_HAS_RENDERER
#    include <GLFW/glfw3.h>
#    include <lpl/engine/systems/CameraSystem.hpp>
#    include <lpl/engine/systems/LocalInputSystem.hpp>
#    include <lpl/engine/systems/RenderSystem.hpp>
#    include <lpl/render/VulkanRenderer.hpp>
#endif

// The BCI stack pulls in eigen / liblsl / brainflow and re-enables exceptions;
// none of that exists freestanding.
#ifdef LPL_HAS_BCI
#    include <lpl/bci/BciAdapter.hpp>
#    include <lpl/bci/SourceBciDriver.hpp>
#    include <lpl/bci/source/SyntheticSource.hpp>
#endif

namespace lpl::engine {

struct Engine::Impl {
    Config config;
    pmr::unique_ptr<platform::IPlatform> platform;
    // The one game World the engine hosts. Injected by the caller (its game), or
    // a default empty World when none is given. The World owns the authoritative
    // Registry, its scheduler and (on demand) its spatial index; the engine holds
    // no game state of its own. A server build owns many Worlds above the engine;
    // a client / solo / kernel build runs exactly this one.
    pmr::unique_ptr<World> world;
    GameLoop loop;

    // The per-frame arena's block is reserved ONCE through the platform seam
    // (malloc on a host, a kernel reservation in ring 0) and then bump-allocated
    // from — so no allocator call happens on a tick's path, which is what the
    // freestanding REAL_TIME mode requires (kmalloc refuses to serve a hot loop).
    void *arenaBlock;
    memory::ArenaAllocator arena;
    // The World's ECS storage arena. Distinct from the frame arena above and
    // NEVER reset while the World lives: chunks allocated from it must survive
    // across frames. Bounding it here is what stops a World from eating the
    // whole kernel heap the way the old un-budgeted spatial index did.
    void *worldArenaBlock;
    memory::ArenaAllocator worldArena;
    ResourceManager resources; ///< Shared asset cache handed to the World's hooks.
    input::InputManager inputManager;
    core::CommandQueue commandQueue;

#ifdef LPL_HAS_NET
    // The built-in physics backend steps the entities the networked server
    // spawns. A client/kernel World is driven by its game instead, so it is not
    // built there — otherwise it would step an empty registry every tick. (The
    // spatial index is NOT here: it belongs to the World, enabled on demand by
    // any game with a large map, networked or solo.)
    pmr::unique_ptr<net::session::SessionManager> sessionManager;
    pmr::unique_ptr<net::netcode::INetcodeStrategy> netcode;
    std::shared_ptr<net::transport::ITransport> transport;
#endif

    EventQueues eventQueues;
    pmr::unique_ptr<physics::IPhysicsBackend> physicsBackend;
#ifdef LPL_HAS_CUDA
    pmr::unique_ptr<gpu::IComputeBackend> computeBackend; // owns the GPU dispatch backend
#endif

    // Client-side state (set by WelcomeSystem)
    // The client's own avatar id, assigned by the server's welcome. kNull (not 0)
    // means "unset": 0 is a valid ECS id — the server's first player — so it must
    // not double as the sentinel, or that player would read as unconnected.
    core::u32 myEntityId{ecs::EntityId::kNull};
    bool connected{false};

#ifdef LPL_HAS_RENDERER
    systems::CameraData cameraData{};
    pmr::unique_ptr<render::IRenderer> renderer;
    GLFWwindow *window{nullptr};
#endif

    bool initialised{false};

    /// Real-time violation counter sampled once the World is up. The counter is
    /// global and kernel smokes bump it before we start, so only the DELTA from
    /// this baseline says anything about our own tick.
    core::u32 realTimeViolationBaseline{0};
    core::u32 realTimeBoundedBaseline{0};
    core::u64 stepsTaken{0};
    bool realTimeReported{false};

#ifdef LPL_HAS_BCI
    pmr::unique_ptr<bci::BciAdapter> bciAdapter;
#endif

    /// Services handed to the World's hooks; refers to members above, so it is
    /// declared after them and lives exactly as long as the engine.
    WorldContext worldContext;

    Impl(Config cfg, pmr::unique_ptr<platform::IPlatform> plat, pmr::unique_ptr<World> game, Engine &owner)
        : config{std::move(cfg)}, platform{std::move(plat)}, world{game ? std::move(game) : pmr::make_unique<World>()},
          loop{config, platform->clock()},
          arenaBlock{platform->memory().reserve(config.arenaSize(), alignof(std::max_align_t))},
          arena{arenaBlock, arenaBlock != nullptr ? config.arenaSize() : core::usize{0}},
          worldArenaBlock{platform->memory().reserve(config.worldArenaSize(), alignof(std::max_align_t))},
          worldArena{worldArenaBlock, worldArenaBlock != nullptr ? config.worldArenaSize() : core::usize{0}},
          resources{}, inputManager{}, worldContext{*platform, resources, arena, config, &owner}
    {
    }

    ~Impl()
    {
        // The arenas never owned these blocks; the platform did.
        platform->memory().release(arenaBlock, config.arenaSize());
        platform->memory().release(worldArenaBlock, config.worldArenaSize());
    }
};

#if !LPL_TARGET_KERNEL
Engine::Engine(Config config)
    : _impl{pmr::make_unique<Impl>(std::move(config), pmr::make_unique<platform::linux_host::LinuxPlatform>(), nullptr,
                                   *this)}
{
}
#endif

#if !LPL_TARGET_KERNEL
Engine::Engine(Config config, pmr::unique_ptr<World> world)
    : _impl{pmr::make_unique<Impl>(std::move(config), pmr::make_unique<platform::linux_host::LinuxPlatform>(),
                                   std::move(world), *this)}
{
}
#endif

Engine::Engine(Config config, pmr::unique_ptr<platform::IPlatform> platform)
    : _impl{pmr::make_unique<Impl>(std::move(config), std::move(platform), nullptr, *this)}
{
}

Engine::Engine(Config config, pmr::unique_ptr<platform::IPlatform> platform, pmr::unique_ptr<World> world)
    : _impl{pmr::make_unique<Impl>(std::move(config), std::move(platform), std::move(world), *this)}
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

#ifdef LPL_HAS_NET
    if (_impl->config.enableNetworking() && _impl->config.serverMode())
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

        _impl->sessionManager = pmr::make_unique<net::session::SessionManager>();
        _impl->netcode = pmr::make_unique<net::netcode::AuthoritativeStrategy>();
    }
    else if (_impl->config.enableNetworking())
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
            net::Endpoint serverAddr{};
            if (!net::Endpoint::parse(_impl->config.serverAddress().c_str(), _impl->config.serverPort(), serverAddr))
            {
                core::Log::error("Engine: Malformed server address in config");
                return core::makeError(core::ErrorCode::InvalidArgument, "Malformed server address");
            }

            if (auto res = net::protocol::sendConnect(*socketTransport, &serverAddr); !res)
            {
                core::Log::warn("Engine", "Connect handshake failed, server may not be running yet");
            }
            else
            {
                core::Log::info("Engine: Connect handshake sent to server");
            }

            // Pre-register server address as default destination for all future sends.
            socketTransport->setDefaultDest(serverAddr);
        }

        _impl->transport = std::move(socketTransport);

        _impl->sessionManager = pmr::make_unique<net::session::SessionManager>();
        _impl->netcode = pmr::make_unique<net::netcode::RollbackStrategy>(8);
    }
#endif // LPL_HAS_NET

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

        _impl->renderer = pmr::make_unique<render::vk::VulkanRenderer>();
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
    //  Built-in system groups (selected by Config, see Config::Builder)    //
    //                                                                      //
    //  These are engine-provided and game-agnostic. Which groups a World    //
    //  gets is declared in the Config; the game's OWN systems and content   //
    //  are registered by World::onInit further down. Nothing here knows     //
    //  about any particular game.                                           //
    // ------------------------------------------------------------------ //

    // Physics: a spatial broad-phase index plus the integrator that steps the
    // World's entities. NOT a networking feature — a solo large-map game wants
    // it just as much as a server, so it is selected by Config::enablePhysics
    // alone. A game that does its own broad-phase (samples::CubePile keeps an
    // octree inside its backend) simply turns it off.
    if (_impl->config.enablePhysics())
    {
        core::Log::info("Engine: Registering built-in physics systems");
        _impl->world->enableSpatialPartition(math::Fixed32::fromFloat(10.0f), _impl->config.worldCellCapacity());

        // The CPU backend is the deterministic reference. When the GPU module is
        // enabled (Config::enableGpu) and a CUDA build is available, the
        // GpuPhysicsBackend bridge drives the physics_tick compute kernel
        // instead; collision/sleeping stay on the CPU. The selection is behind
        // LPL_HAS_CUDA so non-CUDA hosts never reference the bridge.
#ifdef LPL_HAS_CUDA
        if (_impl->config.enableGpu())
        {
            _impl->computeBackend = pmr::make_unique<gpu::CudaBackend>();
            _impl->physicsBackend =
                pmr::make_unique<physics::GpuPhysicsBackend>(_impl->world->registry(), *_impl->computeBackend);
        }
        else
#endif
        {
            _impl->physicsBackend = pmr::make_unique<physics::CpuPhysicsBackend>(_impl->world->registry());
        }
        [[maybe_unused]] auto initRes = _impl->physicsBackend->init();

        auto physics = pmr::make_unique<systems::PhysicsSystem>(*_impl->world->spatialPartition(),
                                                                *_impl->physicsBackend, _impl->world->registry());
        [[maybe_unused]] auto r = _impl->world->scheduler().registerSystem(std::move(physics));
    }

#ifdef LPL_HAS_NET
    if (_impl->config.enableNetworking())
    {
        core::Log::info("Engine: Registering built-in networking systems");
        auto netRecv = pmr::make_unique<systems::NetworkReceiveSystem>(_impl->transport, _impl->eventQueues);
        [[maybe_unused]] auto r = _impl->world->scheduler().registerSystem(std::move(netRecv));
    }

    if (_impl->config.enableNetworking() && _impl->config.serverMode())
    {
        auto session = pmr::make_unique<systems::SessionSystem>(
            *_impl->sessionManager, _impl->eventQueues, _impl->transport, _impl->inputManager,
            *_impl->world->spatialPartition(), _impl->world->registry(), _impl->config.sessionTimeoutMs());
        [[maybe_unused]] auto r1 = _impl->world->scheduler().registerSystem(std::move(session));

        auto inputProc = pmr::make_unique<systems::InputProcessingSystem>(_impl->eventQueues, _impl->inputManager,
                                                                          _impl->sessionManager.get());
        [[maybe_unused]] auto r2 = _impl->world->scheduler().registerSystem(std::move(inputProc));

        auto movement = pmr::make_unique<systems::MovementSystem>(_impl->inputManager, _impl->world->registry());
        [[maybe_unused]] auto r3 = _impl->world->scheduler().registerSystem(std::move(movement));

        // Interest-managed broadcast when a radius is set, full broadcast otherwise
        // (mirrors engine::Server::registerInstanceSystems). Zero keeps the current
        // full-state fallback.
        if (_impl->config.interestRadius() > math::Fixed32::zero())
        {
            auto broadcast = pmr::make_unique<systems::AoiBroadcastSystem>(
                *_impl->sessionManager, _impl->transport, *_impl->world->spatialPartition(), _impl->world->registry(),
                _impl->config.interestRadius(), _impl->config.keyframeInterval(),
                _impl->config.bandwidthBudgetBytes());
            broadcast->setNetworkLod(_impl->config.lodNearRadius(), _impl->config.lodFarInterval());
            broadcast->setPrecisionLod(_impl->config.worldExtent(), _impl->config.lodFarPosBits());
            broadcast->setReliableBaseline(_impl->config.reliableBaseline());
            [[maybe_unused]] auto r4 = _impl->world->scheduler().registerSystem(std::move(broadcast));
        }
        else
        {
            auto broadcast = pmr::make_unique<systems::BroadcastSystem>(
                *_impl->sessionManager, _impl->transport, *_impl->world->spatialPartition(), _impl->world->registry());
            [[maybe_unused]] auto r4 = _impl->world->scheduler().registerSystem(std::move(broadcast));
        }

        auto monitor =
            pmr::make_unique<systems::ServerMonitorSystem>(*_impl->sessionManager, *_impl->world->spatialPartition());
        [[maybe_unused]] auto r5 = _impl->world->scheduler().registerSystem(std::move(monitor));
    }
    else if (_impl->config.enableNetworking())
    {
        auto welcome =
            pmr::make_unique<systems::WelcomeSystem>(_impl->eventQueues, _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r1 = _impl->world->scheduler().registerSystem(std::move(welcome));

        auto reconcile = pmr::make_unique<systems::StateReconciliationSystem>(
            _impl->eventQueues, *_impl->world->spatialPartition(), _impl->world->registry());
        [[maybe_unused]] auto r2 = _impl->world->scheduler().registerSystem(std::move(reconcile));

        auto spawn =
            pmr::make_unique<systems::SpawnSystem>(_impl->world->registry(), _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r3 = _impl->world->scheduler().registerSystem(std::move(spawn));

#    ifdef LPL_HAS_RENDERER
        if (_impl->config.enableRendering())
        {
            auto localInput = pmr::make_unique<systems::LocalInputSystem>(_impl->inputManager, _impl->window,
                                                                          _impl->myEntityId, _impl->connected);
            [[maybe_unused]] auto r4 = _impl->world->scheduler().registerSystem(std::move(localInput));
        }
#    endif

        auto movement = pmr::make_unique<systems::MovementSystem>(_impl->inputManager, _impl->world->registry());
        [[maybe_unused]] auto r5 = _impl->world->scheduler().registerSystem(std::move(movement));

        // §6.4: tell the server what our authoritative state hashed to, so a
        // divergence between its simulation and ours is detected rather than
        // silently drifting.
        auto hashReport = pmr::make_unique<systems::StateHashReportSystem>(*_impl->world, _impl->transport,
                                                                          _impl->connected);
        [[maybe_unused]] auto rHash = _impl->world->scheduler().registerSystem(std::move(hashReport));

        auto inputSend = pmr::make_unique<systems::InputSendSystem>(_impl->inputManager, _impl->transport,
                                                                    _impl->myEntityId, _impl->connected);
        [[maybe_unused]] auto r6 = _impl->world->scheduler().registerSystem(std::move(inputSend));

#    ifdef LPL_HAS_RENDERER
        if (_impl->config.enableRendering())
        {
            auto camera = pmr::make_unique<systems::CameraSystem>(_impl->cameraData, _impl->world->registry(),
                                                                  _impl->window, _impl->myEntityId, _impl->connected);
            [[maybe_unused]] auto r7 = _impl->world->scheduler().registerSystem(std::move(camera));

            auto render = pmr::make_unique<systems::RenderSystem>(_impl->world->registry(), _impl->renderer.get());
            [[maybe_unused]] auto r8 = _impl->world->scheduler().registerSystem(std::move(render));
        }
#    endif
    }
#endif // LPL_HAS_NET

    // ------------------------------------------------------------------ //
    //  BCI subsystem (client-only, when enableBci is set)               //
    // ------------------------------------------------------------------ //

#ifdef LPL_HAS_BCI
    if (_impl->config.enableBci() && !_impl->config.serverMode())
    {
        core::Log::info("Engine: Initialising BCI adapter (SyntheticSource)");
        auto source = pmr::make_unique<bci::source::SyntheticSource>(42, true);
        auto driver = pmr::make_unique<bci::SourceBciDriver>(std::move(source));
        _impl->bciAdapter = pmr::make_unique<bci::BciAdapter>(std::move(driver));
        if (auto res = _impl->bciAdapter->start(); !res)
        {
            core::Log::warn("Engine", "BCI adapter failed to start, continuing without BCI");
            _impl->bciAdapter.reset();
        }
    }
#else
    if (_impl->config.enableBci())
    {
        core::Log::warn("Engine", "BCI requested but LPL_HAS_BCI is not enabled in this build");
    }
#endif // LPL_HAS_BCI

    // ------------------------------------------------------------------ //
    //  Game World setup                                                    //
    // ------------------------------------------------------------------ //

    // Before the World creates a single entity: its chunk storage comes from the
    // bounded persistent arena, not the heap. Chunks built after this point never
    // call the allocator on a tick's path.
    if (_impl->worldArenaBlock != nullptr)
        _impl->world->registry().setAllocator(&_impl->worldArena);

    core::Log::info("Engine: Initialising World");
    if (auto res = _impl->world->onInit(_impl->worldContext); !res)
    {
        core::Log::error("Engine: World failed to initialise");
        return res;
    }

    // Baselines, never absolutes: these counters are global and the kernel's own
    // smoke battery moves them well before the engine ever runs a step.
    _impl->realTimeViolationBaseline = _impl->platform->memory().realTimeViolationCount();
    _impl->realTimeBoundedBaseline = _impl->platform->memory().realTimeBoundedCount();
    _impl->initialised = true;
    core::Log::info("Engine::init — done");
    return {};
}

void Engine::run()
{
    LPL_ASSERT(_impl->initialised);

    // A malformed system graph is not recoverable, and std::abort does not exist
    // freestanding; LPL_VERIFY routes to the kernel halt primitive there.
    LPL_VERIFY(_impl->world->scheduler().buildGraph().has_value());

    // Insert buffer swap between Physics and Network phases so that
    // Broadcast/Render systems read the freshly-computed physics data
    // (mirrors legacy PreSwap → swapBuffers → PostSwap pattern).
    _impl->world->scheduler().setPhaseCallback(ecs::SchedulePhase::Physics,
                                               [this]() { _impl->world->registry().swapAllBuffers(); });

    LoopCallbacks callbacks;

    callbacks.preFrame = [this]() {
        _impl->commandQueue.flush();
        _impl->arena.reset();

        [[maybe_unused]] auto r = _impl->inputManager.poll();

#ifdef LPL_HAS_BCI
        // Poll BCI adapter and feed neural data into InputManager
        if (_impl->bciAdapter)
        {
            if (auto result = _impl->bciAdapter->update(); result.has_value())
            {
                const auto &neural = result.value();
                if (_impl->myEntityId != ecs::EntityId::kNull)
                    _impl->inputManager.setNeural(_impl->myEntityId,
                                              neural.channels[0].toFloat(),       // alpha
                                              neural.channels[1].toFloat(),       // beta
                                              neural.channels[2].toFloat(),       // concentration
                                              neural.channels[3].toFloat() > 0.5f // blink
                );
            }
        }
#endif // LPL_HAS_BCI
    };

    callbacks.fixedUpdate = [this](core::f64 dt) {
#ifdef LPL_HAS_NET
        if (_impl->netcode)
        {
            _impl->netcode->tick(static_cast<core::f32>(dt));
        }
#endif

        // Advance the World one authoritative step, inside a real-time section:
        // on a backend that enforces it the heap refuses to allocate here, so an
        // allocation-free tick is proven rather than assumed. Everything the step
        // needs was reserved up front (the World's persistent arena).
        const bool guarded = _impl->config.enableRealTimeGuard();
        if (guarded)
            _impl->platform->memory().beginRealTimeSection();
        _impl->world->onFixedStep(static_cast<core::f32>(dt));

        if (guarded)
            _impl->platform->memory().endRealTimeSection();

        // Report once, after enough steps that any warm-up allocation would have
        // shown up: did the authoritative tick ever ask the heap for memory?
        constexpr core::u64 kRealTimeProbeSteps = 120;
        if (!_impl->realTimeReported && ++_impl->stepsTaken >= kRealTimeProbeSteps)
        {
            _impl->realTimeReported = true;
            const auto violations =
                _impl->platform->memory().realTimeViolationCount() - _impl->realTimeViolationBaseline;
            const auto bounded = _impl->platform->memory().realTimeBoundedCount() - _impl->realTimeBoundedBaseline;

            // Three outcomes, not two: a tick can hold its deadline (no
            // unbounded path taken) while still doing O(1) heap traffic that is
            // worth removing for throughput.
            core::Log::info(!guarded   ? "Engine: real-time guard off — allocation behaviour unmeasured" :
                            violations ? "Engine: authoritative tick takes an UNBOUNDED allocation path" :
                            bounded    ? "Engine: authoritative tick is deadline-safe but still touches the heap" :
                                         "Engine: authoritative tick is allocation-free");
        }
    };

    callbacks.render = [this](core::f64 alpha) {
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
        _impl->world->onRender(_impl->worldContext, alpha);
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

    if (_impl->platform->memory().realTimeViolationCount() != _impl->realTimeViolationBaseline)
        core::Log::warn("Engine", "allocations were attempted inside the authoritative tick");

    _impl->world->onShutdown();

#ifdef LPL_HAS_NET
    if (_impl->transport)
    {
        _impl->transport->close();
    }
#endif

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

void Engine::submitCommand(pmr::unique_ptr<core::ICommand> cmd) { _impl->commandQueue.push(std::move(cmd)); }

const Config &Engine::config() const noexcept { return _impl->config; }

} // namespace lpl::engine
