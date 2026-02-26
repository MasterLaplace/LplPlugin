// /////////////////////////////////////////////////////////////////////////////
/// @file Engine.hpp
/// @brief Top-level engine façade (Façade pattern).
///
/// Single entry-point that wires all subsystems together:
/// ECS, physics, networking, input, render, audio, haptic, BCI.
// /////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lpl/engine/Config.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <memory>

namespace lpl::engine {

/// @brief Top-level engine façade.
///
/// Owns all subsystem instances, initialises them in dependency
/// order, runs the game loop, and shuts down cleanly.
class Engine
{
public:
    /// @param config Immutable engine configuration.
    explicit Engine(Config config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// @brief Initialise all subsystems.
    /// @return Success or the first error encountered.
    [[nodiscard]] core::Expected<void> init();

    /// @brief Run the main loop (blocks until stop is requested).
    void run();

    /// @brief Request graceful shutdown.
    void requestShutdown() noexcept;

    /// @brief Shut down all subsystems in reverse init order.
    void shutdown();

    /// @brief Access the active configuration.
    [[nodiscard]] const Config& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::engine
