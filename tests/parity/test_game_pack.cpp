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
#include <lpl/pack/ViewerPackBlob.hpp>
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
    check(sizeof(pack::RecipeV1) == 996u, "recipe layout is 996 bytes");

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

    // ── The viewer's built-in cartridge is not stale either ────────────────
    //
    // Same concern as the parity blob above, for a sharper reason: this is what a
    // boot with no cartridge slot actually shows, so a stale blob does not fail
    // anything — it publishes an old world to a browser demo and looks entirely
    // healthy doing it.
    //
    // It cannot be compared byte for byte here, because it is baked from a FILE
    // (assets/games/worldview.lplscene) and a test run by `xmake run` has no
    // dependable working directory. validate.sh does that comparison, by
    // regenerating both headers into a scratch copy and diffing. What is asserted
    // here is the part that catches the likelier fault: a blob regenerated from a
    // DIFFERENT scene would still open and still parse.
    {
        pack::View viewerView;
        check(viewerView.open(pack::kViewerPackBytes, pack::kViewerPackSize), "the viewer's built-in cartridge opens");
        check(viewerView.sectionCount() == 3u, "and carries a world, an ecosystem AND a look");

        pack::RecipeV1 viewerWorld{};
        check(viewerView.readRecipe(viewerWorld), "its world section reads");
        // The values worldview.lplscene declares. Hardcoded on purpose: a blob
        // regenerated from a DIFFERENT scene would still open and still parse,
        // and only a statement of what it is supposed to contain catches that.
        check(viewerWorld.width == 64u && viewerWorld.depth == 64u, "it is the 64x64 demo world");
        check(viewerWorld.seed == 20260728u, "with the seed the document names");

        pack::LivingV1 viewerLife{};
        check(viewerView.readLiving(viewerLife), "its ecosystem section reads");
        check(viewerLife.speciesCount == 3u, "and declares the three species the document lists");
        check(viewerLife.headPerBody == 3u, "with the body ratio it asks for");

        pack::ViewV1 viewerLook{};
        check(viewerView.readView(viewerLook), "and its view profile reads");

        // Three things read "where is the sea": the classifier that decides what is
        // Ocean, the walkability mask, and the water plane the renderer draws. They
        // have to be the same number, and in this cartridge they were not — the
        // document said nothing about biomes, so the classifier used the engine
        // default of -4 while the water was drawn at -1. The band between them was
        // land you could not walk on, under water, that the scatter planted trees in.
        //
        // Nothing was broken anywhere: every layer did exactly what it says it does.
        // That is what makes this worth an assertion rather than a fix.
        const procgen::WorldRecipe viewerRecipe = pack::toEngineRecipe(viewerWorld);
        check(viewerRecipe.biomes.seaLevel == viewerLook.seaLevel,
              "and the sea it classifies is the sea it draws");
    }

    // ── Every pass is IN the recipe ───────────────────────────────────────────
    //
    // These passes lived briefly in a `Morphology` section, for one reason: RecipeV1's
    // size was treated as frozen. It is not — there is no released version of this
    // format — so they are back where they belong, and what has to be asserted is no
    // longer "a pack without them is unchanged" but the thing that actually matters:
    // a recipe survives the round trip WHOLE, and a byte from disk cannot index a
    // switch past its end.
    std::printf("\n-- every pass round-trips through one section --\n");
    {
        procgen::WorldRecipe layered = fromDocument;
        layered.caveKind = procgen::CaveKind::Layered;
        layered.caveSystem.width = layered.width;
        layered.caveSystem.depth = layered.depth;
        layered.terraceSteps = 5u;
        layered.partitionRegions = true;
        layered.raiseBuildings = true;
        layered.buildings.maxFloors = 4u;
        layered.groundClearance = 1.5f;
        layered.roadsideLevels = 3u;
        const char pattern[] = "{[A,P]:2,[BL,P]:1}*,[G,P]";
        for (core::u32 i = 0u; i < sizeof(pattern); ++i)
            layered.roadsidePattern[i] = pattern[i];

        // Six scatter rules: one per biome is how vegetation is actually written, and
        // four ran out at the fifth kind of plant. This is the case that could not be
        // expressed at all before, so a viewer that wanted it had to build its world
        // by hand and could not save, bake or replay what it showed.
        layered.scatterCount = 6u;
        for (core::u32 i = 0u; i < 6u; ++i)
        {
            layered.scatter[i].biome = static_cast<procgen::BiomeId>(i + 1u);
            layered.scatter[i].density = 0.02f * static_cast<core::f32>(i + 1u);
            layered.scatter[i].tag = 100u + i;
        }

        const std::vector<core::u8> grown = editor::bakeGamePack(layered);
        pack::View grownView;
        check(grownView.open(grown.data(), static_cast<core::u32>(grown.size())), "the pack opens");
        check(grownView.sectionCount() == 1u, "and carries ONE section, not a recipe split in half");

        pack::RecipeV1 grownWire{};
        check(grownView.readRecipe(grownWire), "the recipe section reads back");
        const procgen::WorldRecipe decoded = pack::toEngineRecipe(grownWire);

        check(decoded.seed == layered.seed, "the terrain survives");
        check(decoded.caveKind == procgen::CaveKind::Layered, "the cave kind round-trips");
        check(decoded.terraceSteps == 5u, "so do the terraces");
        check(decoded.partitionRegions, "and the provinces");
        check(decoded.raiseBuildings && decoded.buildings.maxFloors == 4u, "and the shape grammar");
        check(decoded.groundClearance == 1.5f, "and the ground clearance");
        check(decoded.roadsideLevels == 3u, "and the verge levels");
        bool samePattern = true;
        for (core::u32 i = 0u; i < sizeof(pattern); ++i)
            samePattern = samePattern && decoded.roadsidePattern[i] == pattern[i];
        check(samePattern, "and the L-system arrives character for character");

        check(decoded.scatterCount == 6u, "all six scatter rules survive the wire");
        bool sameRules = true;
        for (core::u32 i = 0u; i < 6u; ++i)
            sameRules = sameRules && decoded.scatter[i].tag == 100u + i &&
                        decoded.scatter[i].biome == static_cast<procgen::BiomeId>(i + 1u);
        check(sameRules, "including the two past the old ceiling of four");

        // A byte from disk naming a fifth generator must be clamped, not indexed into
        // a switch that has four.
        pack::RecipeV1 corrupt = grownWire;
        corrupt.caveKind = 99u;
        check(pack::toEngineRecipe(corrupt).caveKind == procgen::CaveKind::Cellular,
              "an out-of-range cave kind falls back, not through");
        // Same for a count: a cartridge is input, not a promise.
        corrupt = grownWire;
        corrupt.scatterCount = 4000u;
        check(pack::toEngineRecipe(corrupt).scatterCount == pack::kWireScatterRules,
              "an out-of-range scatter count is clamped to what the array holds");

        // And the built world must actually differ, or the passes are decoration.
        ecs::Registry plainWorld;
        procgen::WorldRecipe unjudged = decoded;
        const auto grownBaked = procgen::bakeWorld(plainWorld, unjudged);
        check(grownBaked.heightSignature != documentBaked.heightSignature,
              "and the world it describes is genuinely a different world");
    }

    // ── The viewer's world is now sayable ─────────────────────────────────────
    //
    // This is the shape apps/mapview builds: six scatter rules, a layered cave system,
    // Voronoi provinces, terracing, the shape grammar, a roadside L-system and a lift
    // that puts the ground above the physics floor. Until every one of those had a
    // name, that world existed only as two hundred lines of hand-written WorldBuilder
    // calls inside an app — unsaveable, unbakeable, unreplayable in ring 0, and with
    // no test target of any kind.
    //
    // The assertion that matters is not that the fields survive (the block above says
    // that) but that the DOCUMENT does: a world you can only build in C++ is not a
    // world an editor or an intelligence can direct.
    std::printf("\n-- the viewer's world survives the document --\n");
    {
        procgen::WorldRecipe viewer = fromDocument;
        viewer.width = 48u;
        viewer.depth = 48u;
        viewer.normalizeTerrain = false;
        viewer.groundClearance = 1.5f;
        viewer.terraceSteps = 8u;
        viewer.partitionRegions = true;
        viewer.provinces.cellSize = 8u;
        viewer.provinces.metric = procgen::DistanceMetric::Chebyshev;
        viewer.caveKind = procgen::CaveKind::Layered;
        viewer.caveSystem.width = 48u;
        viewer.caveSystem.depth = 48u;
        viewer.caveSystem.layers = 3u;
        viewer.caveSystem.entrances = 3u;
        viewer.raiseBuildings = true;
        viewer.buildings.minFloors = 1u;
        viewer.buildings.maxFloors = 3u;
        viewer.roadsideLevels = 3u;
        const char verge[] = "{[A,P]:2,[BL,P]:1}*,[G,P]";
        for (core::u32 i = 0u; i < sizeof(verge); ++i)
            viewer.roadsidePattern[i] = verge[i];

        const procgen::BiomeId kBiomes[] = {procgen::BiomeId::Taiga,   procgen::BiomeId::Forest,
                                            procgen::BiomeId::Rainforest, procgen::BiomeId::Savanna,
                                            procgen::BiomeId::Desert,  procgen::BiomeId::Marsh};
        viewer.scatterCount = 6u;
        for (core::u32 i = 0u; i < 6u; ++i)
        {
            viewer.scatter[i] = procgen::ScatterRule{};
            viewer.scatter[i].biome = kBiomes[i];
            viewer.scatter[i].density = 0.03f + 0.02f * static_cast<core::f32>(i);
            viewer.scatter[i].collidable = i < 3u;
        }

        const std::string document =
            "{\"format\":\"lplscene/1\",\"procedural\":" + editor::emitSceneRecipe(viewer) + ",\"entities\":[]}";
        procgen::WorldRecipe reread{};
        const auto ok = editor::parseSceneRecipe(document, reread);
        check(static_cast<bool>(ok), "the viewer's world emits a document that parses");

        check(reread.scatterCount == 6u, "six scatter rules survive the document");
        check(reread.caveKind == procgen::CaveKind::Layered, "so does the layered underground");
        check(reread.caveSystem.entrances == 3u, "including the knobs a half-carried generator dropped");
        check(reread.groundClearance == viewer.groundClearance, "and the ground clearance");
        check(reread.terraceSteps == 8u && reread.partitionRegions && reread.raiseBuildings,
              "and the terraces, the provinces and the grammar");
        check(reread.provinces.metric == procgen::DistanceMetric::Chebyshev,
              "and the province metric, spelled as a word rather than an index");

        // The decisive one. Field equality can hold while the world differs — that is
        // what a pass read in the wrong order looks like.
        ecs::Registry direct;
        ecs::Registry viaDocument;
        const auto builtDirect = procgen::bakeWorld(direct, viewer);
        const auto builtViaDocument = procgen::bakeWorld(viaDocument, reread);
        check(builtDirect.heightSignature == builtViaDocument.heightSignature,
              "and the world it builds folds identically to the one built from the struct");
        check(builtDirect.biomeSignature == builtViaDocument.biomeSignature, "biomes included");
        check(builtDirect.stateSignature == builtViaDocument.stateSignature, "entities included");
        std::printf("  viewer world: entities=%u height=0x%08X\n", builtDirect.entityCount,
                    builtDirect.heightSignature);
    }

    std::printf("\n-- pack --\n");
    std::printf("  image size   = %zu bytes\n", image.size());
    std::printf("  entities     = %u\n", documentBaked.entityCount);
    std::printf("  state_sig    = 0x%08X\n", documentBaked.stateSignature);

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
