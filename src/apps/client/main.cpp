// /////////////////////////////////////////////////////////////////////////////
/// @file main.cpp
/// @brief LplPlugin client entry-point (desktop / VR).
///
/// Full client: render, audio, haptic, optional BCI.
/// Connects to a remote or local server.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/engine/Engine.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/core/Types.hpp>

int main(int /*argc*/, char* /*argv*/[])
{
    lpl::core::Log::info("=== LplPlugin Client ===");

    auto config = lpl::engine::Config::Builder{}
        .tickRate(144)
        .maxEntities(10000)
        .serverMode(false)
        .headless(false)
        .arenaSize(64 * 1024 * 1024)
        .enableGpu(true)
        .enableBci(false)
        .build();

    lpl::engine::Engine engine{config};

    auto result = engine.init();
    if (!result)
    {
        lpl::core::Log::error("Client init failed");
        return 1;
    }

    engine.run();
    engine.shutdown();

    lpl::core::Log::info("Client exited cleanly");
    return 0;
}
