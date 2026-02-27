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
    #define LPL_ENGINE_ENGINE_HPP

#include <lpl/engine/Config.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <memory>

namespace lpl::engine {

/**
 * @brief Top-level engine façade.
 *
 * Owns all subsystem instances, initialises them in dependency
 * order, runs the game loop, and shuts down cleanly.
 */
class Engine
{
public:
    /// @param config Immutable engine configuration.
    explicit Engine(Config config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

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

    /** @brief Access the active configuration. */
    [[nodiscard]] const Config& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine

#endif // LPL_ENGINE_ENGINE_HPP
