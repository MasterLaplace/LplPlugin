/**
 * @file GameLoop.cpp
 * @brief GameLoop implementation — fixed time-step with accumulator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/engine/GameLoop.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <chrono>

namespace lpl::engine {

GameLoop::GameLoop(const Config& config)
    : _fixedDt{1.0 / static_cast<core::f64>(config.tickRate())}
{
    LPL_ASSERT(config.tickRate() > 0);
}

GameLoop::~GameLoop() = default;

void GameLoop::run(const LoopCallbacks& callbacks)
{
    LPL_ASSERT(callbacks.fixedUpdate);
    _running = true;
    _tickCount = 0;

    using Clock = std::chrono::steady_clock;
    auto previous = Clock::now();
    core::f64 accumulator = 0.0;

    while (_running)
    {
        const auto current = Clock::now();
        core::f64 frameTime = std::chrono::duration<core::f64>(current - previous).count();
        previous = current;

        constexpr core::f64 kMaxFrameTime = 0.25;
        if (frameTime > kMaxFrameTime)
        {
            frameTime = kMaxFrameTime;
        }

        if (callbacks.preFrame)
        {
            callbacks.preFrame();
        }

        accumulator += frameTime;

        while (accumulator >= _fixedDt)
        {
            callbacks.fixedUpdate(_fixedDt);
            accumulator -= _fixedDt;
            ++_tickCount;
        }

        const core::f64 alpha = accumulator / _fixedDt;

        if (callbacks.render)
        {
            callbacks.render(alpha);
        }

        if (callbacks.postFrame)
        {
            callbacks.postFrame();
        }
    }

    core::Log::info("GameLoop: stopped");
}

void GameLoop::requestStop() noexcept
{
    _running = false;
}

bool GameLoop::isRunning() const noexcept
{
    return _running;
}

core::u64 GameLoop::tickCount() const noexcept
{
    return _tickCount;
}

} // namespace lpl::engine
