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
#    include <lpl/engine/Config.hpp>
#    include <memory>

namespace lpl::platform {
class IPlatform;
} // namespace lpl::platform

namespace lpl::engine {

/**
 * @brief Top-level engine façade.
 *
 * Owns all subsystem instances, initialises them in dependency
 * order, runs the game loop, and shuts down cleanly.
 *
 * The engine reaches host/kernel facilities exclusively through an injected
 * platform::IPlatform (clock / display / input / GPU-memory backends). The
 * single-argument constructor defaults to the hosted LinuxPlatform; the
 * two-argument form injects a platform explicitly (the kernel boot facade
 * supplies a KernelPlatform).
 */
class Engine {
public:
    /// @param config Immutable engine configuration (defaults to LinuxPlatform).
    explicit Engine(Config config);

    /// @param config Immutable engine configuration.
    /// @param platform Injected platform seam; must outlive the engine if a
    ///        non-owning reference is desired — here ownership transfers.
    Engine(Config config, std::unique_ptr<platform::IPlatform> platform);

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
    void submitCommand(std::unique_ptr<core::ICommand> cmd);

    /** @brief Access the active configuration. */
    [[nodiscard]] const Config &config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine

#endif // LPL_ENGINE_ENGINE_HPP
