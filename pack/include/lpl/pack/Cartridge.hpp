/**
 * @file Cartridge.hpp
 * @brief Decoding a game pack into the recipes a host runs, with its fallback.
 *
 * Every host does exactly this: try the bytes it was handed, fall back to the
 * pack compiled into the image, and say which one it got. It was written out by
 * hand in the kernel client, and the browser deployment found the interesting
 * half of it the hard way — with no baker on the runner there is no cartridge, so
 * whatever the fallback happens to be IS the published demo.
 *
 * Two rules are encoded here rather than left to each caller:
 *  - a cartridge that fails to validate is NOT silently replaced by the built-in
 *    one. A corrupt game must be reported; papering over it hides the corruption
 *    behind a world that looks fine.
 *  - an absent ecosystem section is legitimate. A pack may describe a world with
 *    nothing declared living on it, and the host keeps its own defaults; only a
 *    wrong-SIZED section is a fault, and the reader already refuses that.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PACK_CARTRIDGE_HPP
#    define LPL_PACK_CARTRIDGE_HPP

#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/pack/GamePack.hpp>
#    include <lpl/pack/RecipeCodec.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

namespace lpl::pack {

/** @brief Where a decoded world came from, which a host wants to report. */
enum class CartridgeSource : core::u8 {
    Cartridge = 0, ///< The bytes the host was handed (a GRUB module, a file).
    BuiltIn,       ///< The pack compiled into the image.
    Defaults       ///< Nothing decoded: the compiled-in recipes are in use.
};

/** @brief What a pack decoded to, plus what happened while decoding it. */
struct Cartridge {
    procgen::WorldRecipe recipe{};
    ecology::LivingRecipe living{};
    CartridgeSource source{CartridgeSource::Defaults};
    bool livingFromPack{false}; ///< False when the pack declared no ecosystem.
    bool failed{false};         ///< A pack was offered and did not validate.

    /**
     * @brief The view profile, still in wire form.
     *
     * Left undecoded here on purpose. Translating it needs render::SkyParams and
     * render::WaterParams, and this module is read by ring 0 precisely because it
     * depends on nothing that draws. engine::toEngineView does the translation, on
     * the side of the fence that already knows what a sky is.
     */
    ViewV1 view{};
    bool viewFromPack{false}; ///< False when the pack said nothing about looks.
};

/**
 * @brief Decodes @p bytes, or @p fallbackBytes, into the recipes to run.
 *
 * @param bytes         Cartridge bytes; may be null.
 * @param size          Cartridge size; may be zero.
 * @param fallbackBytes Pack compiled into the image; may be null.
 * @param fallbackSize  Its size.
 * @param defaults      Recipes to keep when nothing decodes.
 * @param defaultLiving Ecosystem to keep when the pack declares none.
 */
[[nodiscard]] inline Cartridge loadCartridge(const void *bytes, core::u32 size, const void *fallbackBytes,
                                             core::u32 fallbackSize, const procgen::WorldRecipe &defaults,
                                             const ecology::LivingRecipe &defaultLiving)
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
    RecipeV1 wire{};
    if (!view.open(chosen, chosenSize) || !view.readRecipe(wire))
    {
        out.failed = true;
        out.source = CartridgeSource::Defaults;
        return out;
    }

    out.recipe = toEngineRecipe(wire);

    LivingV1 livingWire{};
    if (view.readLiving(livingWire))
    {
        out.living = toEngineLiving(livingWire);
        out.livingFromPack = true;
    }

    ViewV1 viewWire{};
    if (view.readView(viewWire))
    {
        out.view = viewWire;
        out.viewFromPack = true;
    }
    return out;
}

} // namespace lpl::pack

#endif // LPL_PACK_CARTRIDGE_HPP
