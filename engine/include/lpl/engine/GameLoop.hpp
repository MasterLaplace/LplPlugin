/**
 * @file GameLoop.hpp
 * @brief Fixed time-step game loop (144 Hz).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_ENGINE_GAMELOOP_HPP
    #define LPL_ENGINE_GAMELOOP_HPP

#include <lpl/engine/Config.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <functional>

namespace lpl::engine {

/** @brief Callbacks the game loop invokes each frame. */
struct LoopCallbacks
{
    /** @brief Called once per fixed tick (dt = 1/tickRate). */
    std::function<void(core::f64 dt)> fixedUpdate;

    /** @brief Called once per render frame with interpolation alpha [0,1]. */
    std::function<void(core::f64 alpha)> render;

    /** @brief Called once per frame before fixed updates (input poll, etc.). */
    std::function<void()> preFrame;

    /** @brief Called once per frame after render (swap, metrics, etc.). */
    std::function<void()> postFrame;
};

/** @brief Fixed time-step game loop. */
class GameLoop
{
public:
    /// @param config Engine configuration (provides tickRate).
    explicit GameLoop(const Config& config);
    ~GameLoop();

    GameLoop(const GameLoop&) = delete;
    GameLoop& operator=(const GameLoop&) = delete;

    /**
     * @brief Run the loop until requestStop() is called.
     * @param callbacks Tick / render callbacks.
     */
    void run(const LoopCallbacks& callbacks);

    /** @brief Request graceful loop termination. */
    void requestStop() noexcept;

    /** @brief Whether the loop is currently running. */
    [[nodiscard]] bool isRunning() const noexcept;

    /** @brief Total ticks elapsed since run() was called. */
    [[nodiscard]] core::u64 tickCount() const noexcept;

private:
    core::f64 _fixedDt;
    bool _running{false};
    core::u64 _tickCount{0};
};

} // namespace lpl::engine

#endif // LPL_ENGINE_GAMELOOP_HPP
