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
#include <lpl/engine/ConfigValidation.hpp>
#include <lpl/engine/Engine.hpp>
#include <lpl/engine/GameProfile.hpp>
#include <lpl/samples/NetworkDemoWorld.hpp>
#include <lpl/std/memory.hpp>

#include <cstring>

int main(int argc, char *argv[])
{
    lpl::core::Log::info("=== LplPlugin Server ===");

    // Pick the netcode preset from `--game <type>` (default: MMORPG / FullDive).
    auto profile = lpl::engine::GameProfile::Mmorpg;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--game") == 0 && i + 1 < argc)
        {
            if (!lpl::engine::parseGameProfile(argv[i + 1], profile))
                lpl::core::Log::warn("Unknown --game profile; falling back to MMORPG.");
        }
    }
    lpl::core::Log::info("Server game profile: {}", lpl::engine::gameProfileName(profile));

    auto builder = lpl::engine::Config::Builder{};
    builder.tickRate(144)
        .maxEntities(10000)
        .serverMode(true)
        .headless(true)
        .arenaSize(128 * 1024 * 1024)
        .enableGpu(false)
        .enableBci(false);
    lpl::engine::applyGameProfile(builder, profile);
    auto config = builder.build();

    // Surface any contradictory switches before they fail silently at runtime.
    lpl::engine::forEachConfigWarning(config, [](const char *msg) { lpl::core::Log::warn("config: {}", msg); });

    lpl::engine::Engine engine{config, lpl::pmr::make_unique<lpl::samples::NetworkDemoWorld>()};

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
