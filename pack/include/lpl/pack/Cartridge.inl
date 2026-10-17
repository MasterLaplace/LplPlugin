#ifndef LPL_PACK_CARTRIDGE_INL
#define LPL_PACK_CARTRIDGE_INL

namespace lpl::pack {

namespace detail {

inline bool decodeRecipes(const View &view, Cartridge &out) noexcept
{
    RecipeV1 wire{};
    LivingV1 livingWire{};

    if (!view.readRecipe(wire) || !toEngineRecipe(wire, out.recipe, out.refusal))
        return false;
    out.livingFromPack = view.readLiving(livingWire);
    return !out.livingFromPack || toEngineLiving(livingWire, out.living, out.refusal);
}

} // namespace detail

inline Cartridge loadCartridge(const void *bytes, core::u32 size, const void *fallbackBytes, core::u32 fallbackSize,
                               const procgen::WorldRecipe &defaults, const ecology::LivingRecipe &defaultLiving)
{
    Cartridge out;
    out.recipe = defaults;
    out.living = defaultLiving;

    const core::u8 *chosen = static_cast<const core::u8 *>(bytes);
    core::u32 chosenSize = size;
    out.source = CartridgeSource::Cartridge;
    if (chosen == nullptr || chosenSize == 0u)
    {
        chosen = static_cast<const core::u8 *>(fallbackBytes);
        chosenSize = fallbackSize;
        out.source = CartridgeSource::BuiltIn;
    }
    if (chosen == nullptr || chosenSize == 0u)
    {
        out.source = CartridgeSource::Defaults;
        return out;
    }

    View view;
    if (!view.open(chosen, chosenSize) || !detail::decodeRecipes(view, out))
    {
        out.recipe = defaults;
        out.living = defaultLiving;
        out.livingFromPack = false;
        out.failed = true;
        out.source = CartridgeSource::Defaults;
        return out;
    }

    ViewV1 viewWire{};
    if (view.readView(viewWire))
    {
        out.view = viewWire;
        out.viewFromPack = true;
    }
    return out;
}

inline Cartridge loadCartridgeRepairing(core::u8 *bytes, core::u32 size, const void *fallbackBytes,
                                        core::u32 fallbackSize, const procgen::WorldRecipe &defaults,
                                        const ecology::LivingRecipe &defaultLiving, EccRepairReport &outRepair)
{
    outRepair = EccRepairReport{};

    if (bytes != nullptr && size != 0u)
    {
        View probe;
        if (!probe.open(bytes, size))
            (void) repairPack(bytes, size, outRepair);
    }

    return loadCartridge(bytes, size, fallbackBytes, fallbackSize, defaults, defaultLiving);
}

} // namespace lpl::pack

#endif // LPL_PACK_CARTRIDGE_INL
