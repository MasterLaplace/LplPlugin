// /////////////////////////////////////////////////////////////////////////////
/// @file main.cpp
/// @brief LplPlugin dedicated server entry-point.
///
/// Headless server: no render, no audio, no haptic.
/// Runs the authoritative simulation loop at 144 Hz.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/engine/Engine.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/core/Types.hpp>

int main(int /*argc*/, char* /*argv*/[])
{
    lpl::core::Log::info("=== LplPlugin Server ===");

    auto config = lpl::engine::Config::Builder{}
        .tickRate(144)
        .maxEntities(10000)
        .serverMode(true)
        .headless(true)
        .arenaSize(128 * 1024 * 1024)
        .enableGpu(false)
        .enableBci(false)
        .build();

    lpl::engine::Engine engine{config};

    auto result = engine.init();
    if (!result)
    {
        lpl::core::Log::error("Server init failed");
        return 1;
    }

    engine.run();
    engine.shutdown();

    lpl::core::Log::info("Server exited cleanly");
    return 0;
}
