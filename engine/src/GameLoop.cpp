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
#include <csignal>

namespace lpl::engine {

// ────────────────────────────────────────────────────────────────────────── //
//  SIGINT handler (mirrors legacy Core::static_sigint_handler)               //
// ────────────────────────────────────────────────────────────────────────── //

static std::atomic<GameLoop*> s_activeLoop{nullptr};

static void sigintHandler(int /*sig*/)
{
    if (auto* loop = s_activeLoop.load(std::memory_order_relaxed))
    {
        loop->requestStop();
    }
}

GameLoop::GameLoop(const Config& config)
    : _fixedDt{1.0 / static_cast<core::f64>(config.tickRate())}
{
    LPL_ASSERT(config.tickRate() > 0);
}

GameLoop::~GameLoop() = default;

void GameLoop::run(const LoopCallbacks& callbacks)
{
    LPL_ASSERT(callbacks.fixedUpdate);
    _running.store(true, std::memory_order_relaxed);
    _tickCount = 0;

    // Install SIGINT handler for graceful shutdown (legacy parity)
    s_activeLoop.store(this, std::memory_order_relaxed);
    struct sigaction sa{};
    sa.sa_handler = sigintHandler;
    sa.sa_flags = 0;    // SA_RESTART=0 so nanosleep/poll are interrupted
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);

    using Clock = std::chrono::steady_clock;
    auto previous = Clock::now();
    core::f64 accumulator = 0.0;

    while (_running.load(std::memory_order_relaxed))
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

    s_activeLoop.store(nullptr, std::memory_order_relaxed);
    core::Log::info("GameLoop: stopped");
}

void GameLoop::requestStop() noexcept
{
    _running.store(false, std::memory_order_relaxed);
}

bool GameLoop::isRunning() const noexcept
{
    return _running.load(std::memory_order_relaxed);
}

core::u64 GameLoop::tickCount() const noexcept
{
    return _tickCount;
}

} // namespace lpl::engine
