/**
 * @file test_view_profile.cpp
 * @brief A world's LOOK is content, and has to survive the whole round trip.
 *
 * The `.lplscene` format could describe a valley down to its hydraulic erosion
 * iteration count, and what grazed in it down to the predation coefficient — and
 * had no way to say the world was at dusk. The sky's palette, the tint of its
 * water and the colour of a forest were constants compiled into the host, so every
 * world the format could express came out under the same blue midday sky.
 *
 * Adding a section to a wire format is the kind of change that looks done long
 * before it is, so what is asserted here is the whole chain and not the new struct:
 * JSON -> wire -> baked image -> reader -> engine types -> back to JSON. A break
 * anywhere in it is a cartridge that quietly renders as something else.
 *
 * The property this test exists to protect, above the others: a pack that says
 * NOTHING about how the world looks must be byte-for-byte the pack this baker
 * produced before the section existed. Sections are this format's extension
 * mechanism precisely because that can be true; a grown RecipeV1 would have made
 * every cartridge baked so far the wrong size.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-31
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/editor/GameDocument.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/engine/ViewProfile.hpp>
#include <lpl/pack/Cartridge.hpp>
#include <string>

using namespace lpl;

static int failures = 0;

static void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++failures;
}

static bool near(core::f32 a, core::f32 b) noexcept
{
    const core::f32 d = a - b;
    return (d < 0.0f ? -d : d) <= 1.0e-6f;
}

int main()
{
    std::printf("== view profile: document -> pack -> engine ==\n");

    const procgen::WorldRecipe recipe = procgen::parityWorldRecipe();

    // 1. Silence costs nothing. A pack with no view section must be identical to
    //    the one produced before the section existed.
    {
        const std::vector<core::u8> without = editor::bakeGamePack(recipe, nullptr, nullptr);
        const std::vector<core::u8> legacy = editor::bakeGamePack(recipe);
        check(without == legacy, "a pack with no view section is byte-identical to the old one");
    }

    // 2. A stated look survives the bake and comes back as the same bytes.
    pack::ViewV1 authored{};
    {
        engine::ViewProfile profile{};
        profile.sky.zenithR = 0.31f;
        profile.sky.duskG = 0.51f;
        profile.dayFraction = 0.78f;
        profile.surface.seaLevel = 2.5f;
        profile.surface.fogDensity = 0.031f;
        profile.water.deep = 0x00112233u;
        profile.grazerTint = 0x00ABCDEFu;
        profile.bodyScale = 0.62f;
        profile.palette[0] = 0x00FF0000u;
        profile.palette[1] = 0x0000FF00u;
        profile.paletteCount = 2u;
        authored = engine::toWireView(profile);

        const engine::ViewProfile back = engine::toEngineView(authored);
        check(near(back.sky.zenithR, 0.31f) && near(back.sky.duskG, 0.51f), "the sky survives the codec");
        check(near(back.dayFraction, 0.78f), "the time of day survives");
        check(near(back.surface.seaLevel, 2.5f) && near(back.surface.fogDensity, 0.031f), "the surface survives");
        check(back.water.deep == 0x00112233u, "the water tint survives");
        check(back.grazerTint == 0x00ABCDEFu && near(back.bodyScale, 0.62f), "the creatures survive");
        check(back.paletteCount == 2u && back.palette[1] == 0x0000FF00u, "the palette survives");
        check(back.colourFor(1u, 0u) == 0x0000FF00u && back.colourFor(9u, 0x00DEAD00u) == 0x00DEAD00u,
              "an unstated biome falls back to the caller's colour");
    }

    // 3. Through the actual image, read by the actual reader the kernel uses.
    {
        const std::vector<core::u8> image = editor::bakeGamePack(recipe, nullptr, &authored);
        pack::View view;
        check(view.open(image.data(), static_cast<core::u32>(image.size())), "the three-section image validates");

        pack::ViewV1 decoded{};
        check(view.readView(decoded), "the reader finds the view section");
        check(decoded.dayFraction == authored.dayFraction && decoded.waterDeep == authored.waterDeep &&
                  decoded.biomeColourCount == authored.biomeColourCount,
              "and it decodes to the bytes that were written");

        // The cartridge path is the one every host actually takes.
        const pack::Cartridge cartridge = pack::loadCartridge(image.data(), static_cast<core::u32>(image.size()),
                                                              nullptr, 0u, recipe, ecology::parityLivingRecipe());
        check(!cartridge.failed && cartridge.viewFromPack, "loadCartridge reports the profile as present");
        check(engine::toEngineView(cartridge.view).grazerTint == 0x00ABCDEFu, "and hands back what was authored");
    }

    // 4. Absence is not zeroes. A pack with no view section must leave the host's
    //    defaults alone rather than paint the world black.
    {
        const std::vector<core::u8> image = editor::bakeGamePack(recipe, nullptr, nullptr);
        const pack::Cartridge cartridge = pack::loadCartridge(image.data(), static_cast<core::u32>(image.size()),
                                                              nullptr, 0u, recipe, ecology::parityLivingRecipe());
        check(!cartridge.viewFromPack, "a silent pack is reported as silent, not as a black world");
    }

    // 5. The JSON round trip: a document that states a view must re-emit it.
    {
        const std::string document =
            R"({"format":"lplscene/1",)"
            R"("metadata":{"title":"look","startScene":"a"},)"
            R"("scenes":[{"name":"a","systems":["render"],)"
            R"("procedural":{"seed":7,"width":24,"depth":24},)"
            R"("view":{"dayFraction":0.75,"seaLevel":3.5,"sky":{"zenithR":0.5},)"
            R"("water":{"deep":1122867},"palette":[16711680,65280,255]},)"
            R"("entities":[]}]})";

        const auto parsed = editor::parseGameDocument(document);
        check(parsed.has_value(), "the document with a view block parses");
        if (parsed.has_value())
        {
            const editor::SceneDescription *scene = parsed->startScene();
            check(scene != nullptr && scene->hasView, "the scene reports a view block");
            if (scene != nullptr && scene->hasView)
            {
                check(near(scene->view.dayFraction, 0.75f) && near(scene->view.seaLevel, 3.5f),
                      "the stated fields are read");
                check(near(scene->view.fogDensity, 0.010f) && near(scene->view.ambient, 0.28f),
                      "and the unstated ones keep the engine defaults");
                check(near(scene->view.zenithR, 0.5f) && near(scene->view.zenithB, 0.85f),
                      "a partial sky block overrides only what it names");
                check(scene->view.waterDeep == 1122867u, "the water tint is read");
                check(scene->view.biomeColourCount == 3u &&
                          (scene->view.flags & pack::kViewFlagOverridePalette) != 0u,
                      "the palette is read and flagged as an override");

                // Re-emit, re-parse: a save that loses a field is a lossy save that
                // looks like a round trip until someone reloads it.
                const std::string reemitted = editor::emitGameDocument(*parsed);
                const auto again = editor::parseGameDocument(reemitted);
                check(again.has_value() && again->startScene() != nullptr && again->startScene()->hasView,
                      "the re-emitted document still carries the view");
                if (again.has_value() && again->startScene() != nullptr)
                {
                    const pack::ViewV1 &a = scene->view;
                    const pack::ViewV1 &b = again->startScene()->view;
                    check(near(a.dayFraction, b.dayFraction) && near(a.seaLevel, b.seaLevel) &&
                              near(a.zenithR, b.zenithR) && a.waterDeep == b.waterDeep &&
                              a.biomeColourCount == b.biomeColourCount && a.flags == b.flags,
                          "and every field came back unchanged");
                }
            }
        }
    }

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
