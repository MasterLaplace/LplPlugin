/**
 * @file main.cpp
 * @brief LplPlugin client entry-point (desktop / VR).
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

#include <cstring>

int main(int argc, char *argv[])
{
    lpl::core::Log::info("=== LplPlugin Client ===");

    // The genre is chosen server-side; the client only notes it (its replication
    // behaviour — prediction, interpolation, reconciliation — is genre-independent
    // today). Parsed for symmetry with the server and future client-side tuning.
    auto profile = lpl::engine::GameProfile::Mmorpg;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--game") == 0 && i + 1 < argc)
            (void) lpl::engine::parseGameProfile(argv[i + 1], profile);
    lpl::core::Log::info("Client game profile: {}", lpl::engine::gameProfileName(profile));

    auto config = lpl::engine::Config::Builder{}
                      .tickRate(144)
                      .maxEntities(10000)
                      .serverMode(false)
                      .headless(false)
                      .arenaSize(64 * 1024 * 1024)
                      .enableGpu(true)
                      .enableBci(false)
                      .serverAddress("127.0.0.1")
                      .serverPort(4242)
                      .build();

    lpl::engine::forEachConfigWarning(config, [](const char *msg) { lpl::core::Log::warn("config: {}", msg); });

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
