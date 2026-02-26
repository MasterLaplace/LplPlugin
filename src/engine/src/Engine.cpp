// /////////////////////////////////////////////////////////////////////////////
/// @file Engine.cpp
/// @brief Engine façade implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/engine/Engine.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/engine/GameLoop.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/concurrency/ThreadPool.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/input/InputManager.hpp>

namespace lpl::engine {

struct Engine::Impl
{
    Config config;
    GameLoop loop;

    memory::ArenaAllocator arena;
    concurrency::ThreadPool threadPool;
    ecs::Registry registry;
    ecs::SystemScheduler scheduler;
    input::InputManager inputManager;

    bool initialised{false};

    explicit Impl(Config cfg)
        : config{std::move(cfg)}
        , loop{config}
        , arena{config.arenaSize()}
        , threadPool{8}
        , registry{}
        , scheduler{threadPool}
        , inputManager{}
    {
    }
};

Engine::Engine(Config config)
    : impl_{std::make_unique<Impl>(std::move(config))}
{
}

Engine::~Engine()
{
    if (impl_ && impl_->initialised)
    {
        shutdown();
    }
}

core::Expected<void> Engine::init()
{
    core::Log::info("Engine::init — wiring subsystems");

    auto inputResult = impl_->inputManager.init();
    if (!inputResult)
    {
        return inputResult;
    }

    impl_->initialised = true;
    core::Log::info("Engine::init — done");
    return {};
}

void Engine::run()
{
    LPL_ASSERT(impl_->initialised);

    LoopCallbacks callbacks;

    callbacks.preFrame = [this]()
    {
        [[maybe_unused]] auto r = impl_->inputManager.poll();
    };

    callbacks.fixedUpdate = [this](core::f64 dt)
    {
        impl_->scheduler.tick(static_cast<core::f32>(dt));
    };

    callbacks.render = [](core::f64 /*alpha*/)
    {
    };

    callbacks.postFrame = []()
    {
    };

    impl_->loop.run(callbacks);
}

void Engine::requestShutdown() noexcept
{
    impl_->loop.requestStop();
}

void Engine::shutdown()
{
    if (!impl_->initialised)
    {
        return;
    }

    core::Log::info("Engine::shutdown");
    impl_->inputManager.shutdown();
    impl_->initialised = false;
}

const Config& Engine::config() const noexcept
{
    return impl_->config;
}

} // namespace lpl::engine
