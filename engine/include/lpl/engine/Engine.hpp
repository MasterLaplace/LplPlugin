/**
 * @file Engine.hpp
 * @brief Top-level engine façade (Façade pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_ENGINE_ENGINE_HPP
#    define LPL_ENGINE_ENGINE_HPP

#    include <lpl/core/Command.hpp>
#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Platform.hpp>
#    include <lpl/engine/Config.hpp>
#    include <lpl/std/memory.hpp>

namespace lpl::platform {
class IPlatform;
} // namespace lpl::platform

namespace lpl::engine {

class World;

/**
 * @brief Top-level engine façade.
 *
 * Owns all subsystem instances, initialises them in dependency
 * order, runs the game loop, and shuts down cleanly.
 *
 * The engine reaches host/kernel facilities exclusively through an injected
 * platform::IPlatform (clock / display / input / GPU-memory backends). The
 * single-argument constructor defaults to the hosted LinuxPlatform; the
 * two-argument form injects a platform explicitly (the kernel entry point
 * supplies a KernelPlatform).
 *
 * A second seam, World, injects the game itself: the engine hosts exactly one
 * World and drives its lifecycle hooks. The engine holds no game state — the
 * World owns the registry, its systems and its spatial index. Platform says
 * where it runs, World says what it runs. With no World injected the engine
 * hosts a default empty one (used by the built-in networked session).
 */
class Engine {
public:
#    if !LPL_TARGET_KERNEL
    /**
     * @brief Constructs an engine with the given configuration and game World.
     * @param config Immutable engine configuration (defaults to LinuxPlatform).
     * Hosted builds only — the kernel has no default platform to fall back on
     * and must inject a KernelPlatform explicitly.
     */
    explicit Engine(Config config);

    /**
     * @brief Constructs an engine with the given configuration and game World.
     * @param config Immutable engine configuration.
     * @param world Injected game World (ownership transfers).
     * Hosted convenience: defaults the platform to LinuxPlatform.
     */
    Engine(Config config, pmr::unique_ptr<World> world);
#    endif

    /**
     * @brief Constructs an engine with the given configuration, platform, and game World.
     * @param config Immutable engine configuration.
     * @param platform Injected platform seam (ownership transfers).
     */
    Engine(Config config, pmr::unique_ptr<platform::IPlatform> platform);

    /**
     * @brief Constructs an engine with the given configuration, platform, and game World.
     * @param config Immutable engine configuration.
     * @param platform Injected platform seam (ownership transfers).
     * @param world Injected game World (ownership transfers). This is the game.
     */
    Engine(Config config, pmr::unique_ptr<platform::IPlatform> platform, pmr::unique_ptr<World> world);

    ~Engine();

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    /**
     * @brief Initialise all subsystems.
     *
     * It first probes the injected platform seam, the hosted mirror of the engine test
     * `kernel_platform`. With networking on, a server tries the kernel transport
     * (`/dev/lpl0`) first and falls back to a socket when it is unavailable; a client
     * uses a socket transport (same-machine dev friendly), sends the connect handshake,
     * and registers the server as the default destination, so that InputSendSystem can
     * call sendInputs(transport, nullptr, ...) without carrying the server address.
     *
     * The built-in system groups are engine-provided and game-agnostic. Which groups a
     * World gets is declared in the Config; the game's OWN systems and content are
     * registered by World::onInit afterwards. Nothing here knows about any particular
     * game.
     *
     * - Physics: a spatial broad-phase index plus the integrator that steps the World's
     *   entities. NOT a networking feature — a solo large-map game wants it just as much
     *   as a server, so it is selected by Config::enablePhysics alone. A game that does
     *   its own broad-phase simply turns it off. The CPU backend is the deterministic
     *   reference. When the GPU module is enabled (Config::enableGpu) and a CUDA build is
     *   available, the GpuPhysicsBackend bridge drives the physics_tick compute kernel
     *   instead; collision/sleeping stay on the CPU. The selection is behind
     *   LPL_HAS_CUDA so non-CUDA hosts never reference the bridge.
     * - Networking: a server broadcasts interest-managed when a radius is set and the
     *   full state otherwise (mirroring engine::Server::registerInstanceSystems); zero
     *   keeps the full-state fallback. A client tells the server what its authoritative
     *   state hashed to (§6.4), so a divergence between the two simulations is detected
     *   rather than silently drifting.
     * - BCI: client-only, when Config::enableBci is set.
     *
     * Before the World creates a single entity, its chunk storage is pointed at the
     * bounded persistent arena rather than the heap: chunks built after this point never
     * call the allocator on a tick's path.
     *
     * @return Success or the first error encountered.
     */
    [[nodiscard]] core::Expected<void> init();

    /**
     * @brief Run the main loop (blocks until stop is requested).
     *
     * A malformed system graph is not recoverable, and std::abort does not exist
     * freestanding; LPL_VERIFY routes to the kernel halt primitive there.
     *
     * The buffer swap is inserted after the Physics phase, so that the Broadcast/Render
     * systems read the freshly-computed physics data (mirrors the legacy PreSwap →
     * swapBuffers → PostSwap pattern). Each frame polls the BCI adapter, when there is
     * one, and feeds its neural data into the InputManager.
     *
     * The World advances one authoritative step at a time, inside a real-time section
     * when Config::enableRealTimeGuard is on: on a backend that enforces it the heap
     * refuses to allocate there, so an allocation-free tick is proven rather than
     * assumed. Everything the step needs was reserved up front (the World's persistent
     * arena). The loop reports once, after enough steps that any warm-up allocation
     * would have shown up, whether the authoritative tick ever asked the heap for
     * memory — three outcomes, not two: a tick can hold its deadline (no unbounded path
     * taken) while still doing O(1) heap traffic that is worth removing for throughput.
     *
     * Each frame asks the display seam, rather than one windowing library, whether to
     * close: a scanout answers false and pays nothing, a window answers when its close
     * button is pressed. It is asked BEFORE the world draws, so the frame after a close
     * request is not one the host has to keep a surface alive for.
     */
    void run();

    /**
     * @brief Request graceful shutdown.
     */
    void requestShutdown() noexcept;

    /**
     * @brief Shut down all subsystems in reverse init order.
     */
    void shutdown();

    /**
     * @brief Submit a deferred command to run at the start of the next frame.
     */
    void submitCommand(pmr::unique_ptr<core::ICommand> cmd);

    /**
     * @brief Access the active configuration.
     */
    [[nodiscard]] const Config &config() const noexcept;

private:
    struct Impl;
    pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine

#endif // LPL_ENGINE_ENGINE_HPP
