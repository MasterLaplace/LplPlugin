/**
 * @file HostProfile.hpp
 * @brief One-call presets per HOST: what the machine can afford, and how it draws.
 *
 * @ref GameProfile.hpp answers "what kind of game is this" and sets the
 * networking spectrum accordingly. This is the other half of the same idea and
 * the question nobody had a place to answer: "what kind of machine is this
 * running on". A 4 MiB kernel heap and a desktop with gigabytes disagree about
 * every budget in @ref Config, and until now that disagreement lived as forty
 * lines of hand-written builder calls inside each entry point — duplicated
 * between the kernel client, the kernel server and the hosted apps, and drifting.
 *
 * Deliberately NOT here: the streaming radii and the per-tick generation budget
 * (procgen::StreamingParams owns them), and the terrain's amplitude, frequency
 * and scatter rules (procgen::WorldRecipe owns them, and a .lplscene already
 * serialises it). A preset that restated those would be a second source of truth
 * for a quantity that has an owner — the failure this engine has already been
 * bitten by twice.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_HOST_PROFILE_HPP
#    define LPL_ENGINE_HOST_PROFILE_HPP

#    include <lpl/engine/Config.hpp>

namespace lpl::engine {

/**
 * @enum HostProfile
 * @brief The machine a build targets.
 */
enum class HostProfile {
    Ring0Client,     ///< Freestanding, 4 MiB heap, HAL display, no network.
    Ring0Server,     ///< Freestanding, headless, deterministic tick, no display.
    DesktopClient,   ///< Hosted, memory to spare, every presentation feature on.
    DedicatedServer, ///< Hosted, headless, networking and interest management on.
};

/**
 * @brief Applies the budgets and presentation defaults for @p profile.
 *
 * Returns the builder for chaining, so an app writes what is genuinely its own
 * around it: `applyHostProfile(Builder{}.tickRate(60), HostProfile::Ring0Client)`.
 */
inline Config::Builder &applyHostProfile(Config::Builder &builder, HostProfile profile)
{
    switch (profile)
    {
    case HostProfile::Ring0Client:
        // Sized for the kernel's 4 MiB heap, not a desktop's. The hosted defaults
        // (a 64 MiB arena, 65536 world cells) exhaust it during Engine::init and
        // starve the simulation of its entity chunks — measured, not assumed.
        //
        // The real-time guard is ON: inside the guarded section the heap still
        // serves its bounded O(1) paths (a slab hit, a TLSF block) and refuses
        // only the unbounded ones (pool growth, a first-fit walk), so a threatened
        // deadline is caught rather than hoped for.
        //
        // Rendering stays OFF because the World rasterizes itself: the engine's
        // built-in render systems assume a renderer the kernel does not have.
        builder.arenaSize(256u * 1024u)
            .worldArenaSize(512u * 1024u)
            .worldCellCapacity(1024u)
            .maxResidentChunks(56u)
            .enablePhysics(true)
            .enableNetworking(false)
            .enableRendering(false)
            .enableGpu(false)
            .enableBci(false)
            .enableRealTimeGuard(true)
            .headless(false)
            .serverMode(false)
            // Presentation: everything the software rasterizer can afford at
            // 480x300, which measurement says is more than expected — the frame is
            // bounded by the simulation, not by the fill.
            .lodRings(3u)
            .viewDistance(70.0f)
            .enableTerrainShadows(true)
            .shadowChunksPerTick(1u)
            .enablePerPixelSurface(true)
            .enablePbrSurface(false)
            .enableWaterReflection(true)
            .skyBlockSize(1u);
        break;

    case HostProfile::Ring0Server:
        // No display at all, so every presentation knob is off rather than merely
        // unused: a headless build that still computed shadows would burn its tick
        // budget on pixels nobody receives.
        builder.arenaSize(256u * 1024u)
            .worldArenaSize(512u * 1024u)
            .worldCellCapacity(1024u)
            .maxResidentChunks(24u)
            .enablePhysics(true)
            .enableNetworking(false)
            .enableRendering(false)
            .enableGpu(false)
            .enableBci(false)
            .enableRealTimeGuard(true)
            .headless(true)
            .serverMode(false)
            .lodRings(1u)
            .viewDistance(0.0f)
            .enableTerrainShadows(false)
            .shadowChunksPerTick(0u)
            .enablePerPixelSurface(false)
            .enablePbrSurface(false)
            .enableWaterReflection(false)
            .skyBlockSize(4u);
        break;

    case HostProfile::DesktopClient:
        builder.arenaSize(64u * 1024u * 1024u)
            .worldArenaSize(64u * 1024u * 1024u)
            .maxResidentChunks(256u)
            .enablePhysics(true)
            .enableNetworking(false)
            .enableRendering(true)
            .enableRealTimeGuard(false)
            .headless(false)
            .serverMode(false)
            .lodRings(4u)
            .viewDistance(220.0f)
            .enableTerrainShadows(true)
            .shadowChunksPerTick(4u)
            .enablePerPixelSurface(true)
            .enablePbrSurface(true)
            .enableWaterReflection(true)
            .skyBlockSize(1u);
        break;

    case HostProfile::DedicatedServer:
        builder.arenaSize(64u * 1024u * 1024u)
            .worldArenaSize(64u * 1024u * 1024u)
            .maxResidentChunks(512u)
            .enablePhysics(true)
            .enableNetworking(true)
            .enableRendering(false)
            .enableRealTimeGuard(false)
            .headless(true)
            .serverMode(true)
            .lodRings(1u)
            .viewDistance(0.0f)
            .enableTerrainShadows(false)
            .shadowChunksPerTick(0u)
            .enablePerPixelSurface(false)
            .enablePbrSurface(false)
            .enableWaterReflection(false)
            .skyBlockSize(4u);
        break;
    }
    return builder;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_HOST_PROFILE_HPP
