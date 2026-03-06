/**
 * @file main.cpp
 * @brief LplPlugin dedicated server entry-point.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Log.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/engine/Engine.hpp>

int main(int /*argc*/, char * /*argv*/[])
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
