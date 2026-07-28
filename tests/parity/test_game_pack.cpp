/**
 * @file test_game_pack.cpp
 * @brief The document -> pack -> world chain, and the client/server agreement.
 *
 * Proves the property the whole two-encoding design rests on: a game authored
 * once as a `.lplscene` document produces the SAME world in every target that
 * loads it. Two consumers here stand in for the server and the client — one
 * expands the recipe straight from the document, the other from the baked pack
 * a constrained target would receive — and their worlds must fold identically.
 *
 * Also pins the wire format itself: layout sizes, round-tripping, and the
 * rejection of corrupt or truncated images, since a cartridge is untrusted
 * input and the kernel is the reader least able to survive a bad one.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/ecology/LivingRecipe.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/pack/GamePack.hpp>
#include <lpl/pack/ParityPackBlob.hpp>
#include <lpl/pack/RecipeCodec.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char *label)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++g_failures;
}

/// The authored document: what a human, the editor, or the AI writes and git
/// versions. States exactly the fields parityWorldRecipe() overrides, and no
/// others: both start from the engine defaults, so a default that moves moves
/// them together rather than splitting them apart silently.
const char *kSceneDocument = R"({
  "format": "lplscene/1",
  "procedural": {
    "seed": 1337, "width": 24, "depth": 24, "cellSize": 0.5,
    "terrain":    {"seed": 1337, "frequency": 0.15, "amplitude": 12.0, "octaves": 4,
                   "low": -8.0, "high": 16.0},
    "erosion":    {"enabled": true, "thermalIterations": 8, "hydraulicIterations": 12},
    "rivers":     {"enabled": true},
    "climate":    {"enabled": true},
    "caves":      {"enabled": true, "width": 24, "depth": 24, "seed": 51790, "minRegionSize": 12},
    "settlement": {"enabled": true, "districtSize": 8},
    "roads":      {"enabled": true, "iterations": 3, "stepLength": 2},
    "gate":       {"enabled": true, "minPathLength": 4, "minWalkableCells": 16},
    "scatter":    [{"biome": "grassland", "density": 0.06, "halfExtent": 0.2, "tag": 1}]
  },
  "entities": []
})";

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== game pack: document -> pack -> world ==\n\n");

    // ── The wire format is pinned ────────────────────────────────────────────
    check(sizeof(pack::Header) == 32u, "header layout is 32 bytes");
    check(sizeof(pack::SectionEntry) == 16u, "section entry layout is 16 bytes");
    check(sizeof(pack::ScatterV1) == 52u, "scatter rule layout is 52 bytes");
    check(sizeof(pack::RecipeV1) == 532u, "recipe layout is 532 bytes");

    // ── The document parses into a recipe ───────────────────────────────────
    procgen::WorldRecipe fromDocument{};
    const auto parsed = editor::parseSceneRecipe(kSceneDocument, fromDocument);
    check(parsed.has_value(), "the \"procedural\" block parses");

    // ── The document describes exactly the parity recipe ────────────────────
    // If this drifts, the document and the compiled-in reference disagree, and
    // every downstream signature comparison would be measuring the wrong thing.
    const procgen::WorldRecipe reference = procgen::parityWorldRecipe();
    ecs::Registry referenceWorld;
    const auto referenceBaked = procgen::bakeWorld(referenceWorld, reference);

    ecs::Registry documentWorld;
    const auto documentBaked = procgen::bakeWorld(documentWorld, fromDocument);
    check(documentBaked.stateSignature == referenceBaked.stateSignature,
          "the document describes the same world as parityWorldRecipe()");

    // ── Baking, then reading back ───────────────────────────────────────────
    const std::vector<core::u8> image = editor::bakeGamePack(fromDocument);
    check(!image.empty(), "baking produces a pack image");

    pack::View view;
    check(view.open(image.data(), static_cast<core::u32>(image.size())), "the baked pack opens");
    check(view.sectionCount() == 1u, "the pack carries one section");

    pack::RecipeV1 wire{};
    check(view.readRecipe(wire), "the recipe section reads back");

    // ── Client/server agreement: two consumers, one document ────────────────
    // "Server": expands the recipe straight from the authored document.
    // "Client": expands it from the baked pack, as a constrained target would.
    const procgen::WorldRecipe fromPack = pack::toEngineRecipe(wire);
    ecs::Registry clientWorld;
    const auto clientBaked = procgen::bakeWorld(clientWorld, fromPack);

    check(clientBaked.entityCount == documentBaked.entityCount, "both consumers build the same entity count");
    check(clientBaked.stateSignature == documentBaked.stateSignature,
          "both consumers fold the SAME world (client/server agreement)");
    check(clientBaked.gatePathLength == documentBaked.gatePathLength, "both consumers agree on the playability gate");
    check(clientBaked.heightSignature == documentBaked.heightSignature, "both consumers fold the same terrain");
    check(clientBaked.biomeSignature == documentBaked.biomeSignature, "both consumers fold the same biomes");

    // ── The emitted block round-trips ───────────────────────────────────────
    const std::string emitted =
        "{\"format\":\"lplscene/1\",\"procedural\":" + editor::emitSceneRecipe(fromDocument) + ",\"entities\":[]}";
    procgen::WorldRecipe reparsed{};
    const auto reparsedOk = editor::parseSceneRecipe(emitted, reparsed);
    ecs::Registry reparsedWorld;
    const auto reparsedBaked = procgen::bakeWorld(reparsedWorld, reparsed);
    check(reparsedOk.has_value() && reparsedBaked.stateSignature == documentBaked.stateSignature,
          "an emitted \"procedural\" block re-reads to the same world");

    // ── A cartridge is untrusted input ──────────────────────────────────────
    pack::View rejected;
    check(!rejected.open(nullptr, 0u), "a null image is rejected");
    check(!rejected.open(image.data(), static_cast<core::u32>(image.size()) - 1u), "a truncated image is rejected");

    std::vector<core::u8> corrupt = image;
    corrupt[corrupt.size() - 1u] ^= 0xFFu; // flip bits in the recipe payload
    check(!rejected.open(corrupt.data(), static_cast<core::u32>(corrupt.size())),
          "a corrupt image fails its content hash");

    std::vector<core::u8> wrongMagic = image;
    wrongMagic[0] = 'X';
    check(!rejected.open(wrongMagic.data(), static_cast<core::u32>(wrongMagic.size())),
          "an image with the wrong magic is rejected");

    std::vector<core::u8> wrongVersion = image;
    wrongVersion[pack::kMagicSize] = 0xFEu; // formatVersion low byte
    check(!rejected.open(wrongVersion.data(), static_cast<core::u32>(wrongVersion.size())),
          "an unknown format version is rejected");

    // ── The checked-in reference cartridge is not stale ─────────────────────
    // The kernel exercises kParityPackBytes directly (it has no host tool at
    // build time). Assert here that those bytes are still what the baker emits,
    // so a recipe change cannot leave the kernel validating a dead world.
    const std::vector<core::u8> referenceImage = editor::bakeGamePack(reference);
    bool blobMatches = referenceImage.size() == pack::kParityPackSize;
    for (core::u32 i = 0u; blobMatches && i < pack::kParityPackSize; ++i)
        blobMatches = referenceImage[i] == pack::kParityPackBytes[i];
    check(blobMatches, "the checked-in reference cartridge matches the baker");

    // ── The ecosystem crosses the wire too ──────────────────────────────────
    //
    // The world was authorable down to the erosion iteration count while what
    // lived on it was compiled into the host. These checks are what stop that
    // regressing: a section that round-trips wrong is a cartridge describing one
    // ecosystem while the game runs another, and nothing on screen would say so.
    {
        std::printf("\n-- living section --\n");

        const ecology::LivingRecipe living = ecology::parityLivingRecipe();
        const std::vector<core::u8> withLife = editor::bakeGamePack(reference, &living);

        pack::View lifeView;
        check(lifeView.open(withLife.data(), static_cast<core::u32>(withLife.size())), "a two-section pack opens");
        check(lifeView.sectionCount() == 2u, "and carries both sections");

        pack::RecipeV1 worldWire{};
        check(lifeView.readRecipe(worldWire), "the world section still reads");
        check(worldWire.seed == reference.seed, "and is unchanged by the new neighbour");

        pack::LivingV1 lifeWire{};
        check(lifeView.readLiving(lifeWire), "the living section reads");

        const ecology::LivingRecipe decoded = pack::toEngineLiving(lifeWire);
        check(decoded.seed == living.seed, "the seed survives the round trip");
        check(decoded.ticks == living.ticks, "the tick count survives");
        check(decoded.speciesCount == living.speciesCount, "every species survives");
        check(decoded.stepSeconds.raw() == living.stepSeconds.raw(), "the step is carried as raw Q16.16");
        check(decoded.headPerBody == living.headPerBody, "so does the body ratio");

        bool webMatches = true;
        for (core::u32 i = 0u; i < decoded.speciesCount; ++i)
        {
            webMatches =
                webMatches && decoded.species[i].params.capacity.raw() == living.species[i].params.capacity.raw();
            webMatches = webMatches && decoded.species[i].initial.raw() == living.species[i].initial.raw();
            webMatches = webMatches && decoded.species[i].preyIndex == living.species[i].preyIndex;
            webMatches = webMatches && decoded.species[i].params.level == living.species[i].params.level;
        }
        check(webMatches, "and every species keeps its demography, its prey and its level");

        // The decisive one: the same recipe must FOLD the same. A layout that
        // survives a field-by-field comparison and still changes the simulation
        // is a layout that reorders something the run reads in order.
        const ecology::LivingResult direct = ecology::runLiving(living);
        const ecology::LivingResult viaWire = ecology::runLiving(decoded);
        check(direct.populationSignature == viaWire.populationSignature,
              "the decoded recipe folds the same populations");
        check(direct.genomeSignature == viaWire.genomeSignature, "the same genomes");
        check(direct.stigmergySignature == viaWire.stigmergySignature, "the same field");
        check(direct.socialSignature == viaWire.socialSignature, "the same social state");

        // A one-section pack is still valid, and reports the absence rather than
        // handing back zeroes: a world with nothing declared living on it is a
        // legitimate cartridge, not a corrupt one.
        pack::View barren;
        check(barren.open(image.data(), static_cast<core::u32>(image.size())), "a one-section pack still opens");
        pack::LivingV1 absent{};
        check(!barren.readLiving(absent), "and reports that it declares no ecosystem");

        std::printf("  two-section image = %zu bytes\n", withLife.size());
        std::printf("  species carried   = %u\n", decoded.speciesCount);
    }

    std::printf("\n-- pack --\n");
    std::printf("  image size   = %zu bytes\n", image.size());
    std::printf("  entities     = %u\n", documentBaked.entityCount);
    std::printf("  state_sig    = 0x%08X\n", documentBaked.stateSignature);

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
