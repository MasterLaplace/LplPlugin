/**
 * @file Boot.hpp
 * @brief One call that boots a game: cartridge, budgets, engine, loop.
 *
 * The question this answers is "what if only the engine remains and the engine
 * does the rest". What each entry point actually did was: decode a pack, hand-write
 * forty lines of budgets, construct an Engine, call init/run/shutdown, and log the
 * same four sentences. All of that is host-independent except ONE thing — which
 * World to run — so that is the only thing left to pass in.
 *
 * The World arrives as a FACTORY, not as an object, and that is what keeps the
 * dependency arrow pointing the right way: the engine hands the factory the decoded
 * recipes and gets back a World it knows nothing else about. An engine that named a
 * game type would be an engine that ships with a game.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_BOOT_HPP
#    define LPL_ENGINE_BOOT_HPP

#    include <lpl/core/Log.hpp>
#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/engine/Config.hpp>
#    include <lpl/engine/ConfigValidation.hpp>
#    include <lpl/engine/Engine.hpp>
#    include <lpl/engine/HostProfile.hpp>
#    include <lpl/engine/ViewProfile.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/pack/Cartridge.hpp>
#    include <lpl/platform/IPlatform.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>
#    include <lpl/std/memory.hpp>

#    include <utility>

namespace lpl::engine {

/**
 * @struct BootRequest
 * @brief Everything a host has to say about itself.
 */
struct BootRequest {
    HostProfile host{HostProfile::Ring0Client};
    core::u32 tickRate{60u};

    /**
     * Cartridge bytes, if any.
     *
     * MUTABLE, so the parity section can be used. A reader handed a const pointer can
     * detect damage and nothing more; repairing needs somewhere to put the corrected
     * byte. The kernel copies its GRUB module into memory it owns before pointing
     * this at it.
     */
    core::u8 *packBytes{nullptr};
    core::u32 packSize{0u};
    const void *fallbackPackBytes{nullptr}; ///< Pack compiled into the image, if any.
    core::u32 fallbackPackSize{0u};

    const char *banner{"=== Laplace ==="}; ///< First line in the log.
};

/**
 * @brief What booting produced, for a caller that wants to report or assert on it.
 */
struct BootResult {
    bool initialised{false};
    pack::CartridgeSource source{pack::CartridgeSource::Defaults};
    bool packFailed{false};
    core::u32 configWarnings{0u}; ///< Inconsistencies the config check reported.
    bool viewFromPack{false};     ///< The pack said what the world looks like.
};

/**
 * @brief Decodes the cartridge, sizes the budgets, runs the game to completion.
 *
 * @param request  What kind of host this is and where the game bytes are.
 * @param platform The platform seam (kernel HAL, Linux, …), moved in.
 * @param makeWorld Factory: (WorldRecipe, LivingRecipe, ViewProfile) -> World. The
 *                 view profile is passed even when the pack declared none, holding
 *                 the engine's defaults — a factory that had to test a flag would
 *                 have every game re-implement the same fallback.
 * @param tune     Optional last word on the Config, applied after the profile —
 *                 so a host can override one budget without restating forty.
 */
template <typename WorldFactory, typename Tune>
BootResult bootGame(const BootRequest &request, pmr::unique_ptr<platform::IPlatform> platform, WorldFactory &&makeWorld,
                    Tune &&tune)
{
    BootResult result{};
    core::Log::info(request.banner);

    // Repairing, not plain: a cartridge that carries parity gets one chance to fix
    // itself before the boot falls back to the compiled recipe. Only when it does not
    // already open — a healthy image is never written to.
    pack::EccRepairReport repair{};
    const pack::Cartridge cartridge = pack::loadCartridgeRepairing(
        request.packBytes, request.packSize, request.fallbackPackBytes, request.fallbackPackSize,
        procgen::parityWorldRecipe(), ecology::parityLivingRecipe(), repair);
    result.source = cartridge.source;
    result.packFailed = cartridge.failed;

    if (repair.present && repair.damagedCodewords != 0u)
        core::Log::info("Boot: the cartridge was damaged and its parity section repaired it");
    else if (repair.present && !repair.repaired)
        core::Log::error("Boot: the cartridge is damaged beyond what its parity can correct");

    if (cartridge.failed)
        core::Log::error("Boot: the pack failed to validate — falling back to the compiled recipe");
    else if (cartridge.source == pack::CartridgeSource::Cartridge)
        core::Log::info("Boot: world decoded from the cartridge");
    else if (cartridge.source == pack::CartridgeSource::BuiltIn)
        core::Log::info("Boot: world decoded from the built-in pack");
    if (cartridge.livingFromPack)
        core::Log::info("Boot: ecosystem decoded from the pack");

    // The look is the third thing a document can carry, and the only one that has
    // to be TRANSLATED here rather than in pack/: turning it into a sky needs
    // render types, and pack/ is read by ring 0 precisely because it depends on
    // nothing that draws.
    ViewProfile view{};
    if (cartridge.viewFromPack)
    {
        view = toEngineView(cartridge.view);
        result.viewFromPack = true;
        core::Log::info("Boot: view profile decoded from the pack");
    }

    Config::Builder builder;
    builder.tickRate(request.tickRate);
    applyHostProfile(builder, request.host);
    tune(builder);
    const Config config = builder.build();

    // Every boot validates its own configuration, and this is the reason it happens
    // HERE: forEachConfigWarning existed, was written to be called after build(),
    // and had no caller anywhere in the tree — a checker nobody runs is a comment.
    // Booting is the one moment a Config is finished and not yet used, so it is the
    // only place the check cannot be forgotten.
    result.configWarnings = forEachConfigWarning(config, [](const char *message) { core::Log::warn(message); });

    Engine engine{config, std::move(platform), makeWorld(cartridge.recipe, cartridge.living, view)};

    if (auto initialised = engine.init(); !initialised)
    {
        core::Log::error("Boot: engine init failed");
        return result;
    }
    result.initialised = true;

    engine.run();
    engine.shutdown();
    core::Log::info("Boot: exited cleanly");
    return result;
}

/** @brief @copybrief bootGame — with the profile's budgets left exactly as they are. */
template <typename WorldFactory>
BootResult bootGame(const BootRequest &request, pmr::unique_ptr<platform::IPlatform> platform, WorldFactory &&makeWorld)
{
    return bootGame(request, std::move(platform), makeWorld, [](Config::Builder &) {});
}

} // namespace lpl::engine

#endif // LPL_ENGINE_BOOT_HPP
