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

class IApplication;

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
 * A second seam, IApplication, injects the game/simulation payload. The engine
 * itself holds no game logic: platform says where it runs, application says
 * what it runs. Both are optional — with no application the engine runs its
 * built-in networked session only.
 */
class Engine {
public:
#    if !LPL_TARGET_KERNEL
    /// @param config Immutable engine configuration (defaults to LinuxPlatform).
    /// Hosted builds only — the kernel has no default platform to fall back on
    /// and must inject a KernelPlatform explicitly.
    explicit Engine(Config config);
#    endif

    /// @param config Immutable engine configuration.
    /// @param platform Injected platform seam; must outlive the engine if a
    ///        non-owning reference is desired — here ownership transfers.
    Engine(Config config, pmr::unique_ptr<platform::IPlatform> platform);

    /// @param config Immutable engine configuration.
    /// @param platform Injected platform seam (ownership transfers).
    /// @param application Injected game/simulation payload (ownership transfers).
    Engine(Config config, pmr::unique_ptr<platform::IPlatform> platform, pmr::unique_ptr<IApplication> application);

    ~Engine();

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    /**
     * @brief Initialise all subsystems.
     * @return Success or the first error encountered.
     */
    [[nodiscard]] core::Expected<void> init();

    /** @brief Run the main loop (blocks until stop is requested). */
    void run();

    /** @brief Request graceful shutdown. */
    void requestShutdown() noexcept;

    /** @brief Shut down all subsystems in reverse init order. */
    void shutdown();

    /** @brief Submit a deferred command to run at the start of the next frame. */
    void submitCommand(pmr::unique_ptr<core::ICommand> cmd);

    /** @brief Access the active configuration. */
    [[nodiscard]] const Config &config() const noexcept;

private:
    struct Impl;
    pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine

#endif // LPL_ENGINE_ENGINE_HPP
