#ifndef LPL_ENGINE_BOOT_INL
#define LPL_ENGINE_BOOT_INL

namespace lpl::engine {

namespace detail {

inline void logRefusedField(const char *field)
{
    pmr::string line{"Boot: the pack holds a value this build does not know in "};

    line += field;
    line += " — falling back to the compiled recipe";
    core::Log::error(line);
}

} // namespace detail

template <typename WorldFactory, typename Tune>
BootResult bootGame(const BootRequest &request, pmr::unique_ptr<platform::IPlatform> platform, WorldFactory &&makeWorld,
                    Tune &&tune)
{
    BootResult result{};
    core::Log::info(request.banner);

    pack::EccRepairReport repair{};
    const pack::Cartridge cartridge = pack::loadCartridgeRepairing(
        request.packBytes, request.packSize, request.fallbackPackBytes, request.fallbackPackSize,
        procgen::parityWorldRecipe(), ecology::parityLivingRecipe(), repair);
    result.source = cartridge.source;
    result.packFailed = cartridge.failed;
    result.refusal = cartridge.refusal;

    if (repair.present && repair.damagedCodewords != 0u)
        core::Log::info("Boot: the cartridge was damaged and its parity section repaired it");
    else if (repair.present && !repair.repaired)
        core::Log::error("Boot: the cartridge is damaged beyond what its parity can correct");

    if (cartridge.refusal.field != nullptr)
        detail::logRefusedField(cartridge.refusal.field);
    else if (cartridge.failed)
        core::Log::error("Boot: the pack failed to validate — falling back to the compiled recipe");
    else if (cartridge.source == pack::CartridgeSource::Cartridge)
        core::Log::info("Boot: world decoded from the cartridge");
    else if (cartridge.source == pack::CartridgeSource::BuiltIn)
        core::Log::info("Boot: world decoded from the built-in pack");
    if (cartridge.livingFromPack)
        core::Log::info("Boot: ecosystem decoded from the pack");

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

} // namespace lpl::engine

#endif // LPL_ENGINE_BOOT_INL
