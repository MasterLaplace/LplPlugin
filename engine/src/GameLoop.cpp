/**
 * @file GameLoop.cpp
 * @brief GameLoop implementation — fixed time-step with accumulator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/engine/GameLoop.hpp>
#include <lpl/platform/IClockBackend.hpp>

#if !LPL_TARGET_KERNEL
#    include <csignal>
#endif

namespace lpl::engine {

#if !LPL_TARGET_KERNEL
// ────────────────────────────────────────────────────────────────────────── //
//  SIGINT handler (mirrors legacy Core::static_sigint_handler).              //
//  Host-only: the kernel has no POSIX signals; in-kernel shutdown is driven  //
//  by requestStop() from the boot facade / input ring instead.              //
// ────────────────────────────────────────────────────────────────────────── //

static std::atomic<GameLoop *> s_activeLoop{nullptr};

static void sigintHandler(int /*sig*/)
{
    if (auto *loop = s_activeLoop.load(std::memory_order_relaxed))
    {
        loop->requestStop();
    }
}
#endif // !LPL_TARGET_KERNEL

GameLoop::GameLoop(const Config &config, platform::IClockBackend &clock)
    : _clock{clock}, _fixedDt{1.0 / static_cast<core::f64>(config.tickRate())}
{
    LPL_ASSERT(config.tickRate() > 0);
}

GameLoop::~GameLoop() = default;

void GameLoop::run(const LoopCallbacks &callbacks)
{
    LPL_ASSERT(callbacks.fixedUpdate);
    _running.store(true, std::memory_order_relaxed);
    _tickCount = 0;

#if !LPL_TARGET_KERNEL
    // Install SIGINT handler for graceful shutdown (legacy parity)
    s_activeLoop.store(this, std::memory_order_relaxed);
    struct sigaction sa {};
    sa.sa_handler = sigintHandler;
    sa.sa_flags = 0; // SA_RESTART=0 so nanosleep/poll are interrupted
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
#endif

    // Wall-clock pacing comes from the platform clock backend (non-authoritative;
    // it only governs how many fixed steps run, never the fixed dt value). The
    // tick count is u32 and may wrap, so deltas are taken modular-safe.
    const core::f64 tickHz = static_cast<core::f64>(_clock.tickHertz());
    core::u32 previous = _clock.tickCount();
    core::f64 accumulator = 0.0;

    while (_running.load(std::memory_order_relaxed))
    {
        const core::u32 current = _clock.tickCount();
        const core::u32 deltaTicks = current - previous; // wrap-safe modular delta
        previous = current;
        core::f64 frameTime = (tickHz > 0.0) ? static_cast<core::f64>(deltaTicks) / tickHz : 0.0;

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

#if !LPL_TARGET_KERNEL
    s_activeLoop.store(nullptr, std::memory_order_relaxed);
#endif
    core::Log::info("GameLoop: stopped");
}

void GameLoop::requestStop() noexcept { _running.store(false, std::memory_order_relaxed); }

bool GameLoop::isRunning() const noexcept { return _running.load(std::memory_order_relaxed); }

core::u64 GameLoop::tickCount() const noexcept { return _tickCount; }

} // namespace lpl::engine
