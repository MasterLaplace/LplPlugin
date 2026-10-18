#include <lpl/ecology/LivingRecipe.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/engine/ViewProfile.hpp>
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

/**
 * @brief Decodes @p wire as a host does, failing the run when the reader refuses it.
 */
lpl::procgen::WorldRecipe decodeRecipe(const lpl::pack::RecipeV1 &wire)
{
    lpl::procgen::WorldRecipe recipe{};
    lpl::pack::WireRefusal refusal{};

    check(lpl::pack::toEngineRecipe(wire, recipe, refusal), "the reader accepts the recipe");
    return recipe;
}

/**
 * @brief Decodes @p wire as a host does, failing the run when the reader refuses it.
 */
lpl::ecology::LivingRecipe decodeLiving(const lpl::pack::LivingV1 &wire)
{
    lpl::ecology::LivingRecipe living{};
    lpl::pack::WireRefusal refusal{};

    check(lpl::pack::toEngineLiving(wire, living, refusal), "the reader accepts the living recipe");
    return living;
}

/**
 * @brief The authored document: what a human, the editor, or the AI writes and git versions. States exactly
 *        the fields parityWorldRecipe() overrides, and no others: both start from the engine defaults, so a
 *        default that moves moves them together rather than splitting them apart silently.
 */
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

/**
 * @brief The wire format is pinned.
 */
void checkWireFormatIsPinned()
{
    check(sizeof(lpl::pack::Header) == 32u, "header layout is 32 bytes");
    check(sizeof(lpl::pack::SectionEntry) == 16u, "section entry layout is 16 bytes");
    check(sizeof(lpl::pack::ScatterV1) == 52u, "scatter rule layout is 52 bytes");
    check(sizeof(lpl::pack::RecipeV1) == 996u, "recipe layout is 996 bytes");
}

/**
 * @brief The document parses into a recipe.
 *
 * @param fromDocument Receives the recipe the document describes.
 */
void checkDocumentParsesIntoARecipe(lpl::procgen::WorldRecipe &fromDocument)
{
    const auto parsed = lpl::editor::parseSceneRecipe(kSceneDocument, fromDocument);
    check(parsed.has_value(), "the \"procedural\" block parses");
}

/**
 * @brief The document describes exactly the parity recipe.
 *
 * @details If this drifts, the document and the compiled-in reference disagree, and every downstream signature
 *          comparison would be measuring the wrong thing.
 *
 * @param reference     The compiled-in parity recipe.
 * @param fromDocument  The recipe the document describes.
 * @param documentBaked Receives what baking @p fromDocument produced.
 */
void checkDocumentDescribesExactlyTheParityRecipe(const lpl::procgen::WorldRecipe &reference,
                                                  const lpl::procgen::WorldRecipe &fromDocument,
                                                  lpl::procgen::WorldRecipeResult &documentBaked)
{
    lpl::ecs::Registry referenceWorld;
    const auto referenceBaked = lpl::procgen::bakeWorld(referenceWorld, reference);

    lpl::ecs::Registry documentWorld;
    documentBaked = lpl::procgen::bakeWorld(documentWorld, fromDocument);
    check(documentBaked.stateSignature == referenceBaked.stateSignature,
          "the document describes the same world as parityWorldRecipe()");
}

/**
 * @brief Baking, then reading back.
 *
 * @param image The pack baked from the document.
 * @param wire  Receives the recipe section read back from @p image.
 */
void checkBakedPackReadsBack(const std::vector<lpl::core::u8> &image, lpl::pack::RecipeV1 &wire)
{
    check(!image.empty(), "baking produces a pack image");

    lpl::pack::View view;
    check(view.open(image.data(), static_cast<lpl::core::u32>(image.size())), "the baked pack opens");
    check(view.sectionCount() == 1u, "the pack carries one section");

    check(view.readRecipe(wire), "the recipe section reads back");
}

/**
 * @brief Client/server agreement: two consumers, one document.
 *
 * @details The "server" expands the recipe straight from the authored document; the "client" expands it from
 *          the baked pack, as a constrained target would.
 *
 * @param wire          The recipe section read back from the baked pack.
 * @param documentBaked What baking the document's recipe produced.
 */
void checkClientAndServerAgree(const lpl::pack::RecipeV1 &wire, const lpl::procgen::WorldRecipeResult &documentBaked)
{
    const lpl::procgen::WorldRecipe fromPack = decodeRecipe(wire);
    lpl::ecs::Registry clientWorld;
    const auto clientBaked = lpl::procgen::bakeWorld(clientWorld, fromPack);

    check(clientBaked.entityCount == documentBaked.entityCount, "both consumers build the same entity count");
    check(clientBaked.stateSignature == documentBaked.stateSignature,
          "both consumers fold the SAME world (client/server agreement)");
    check(clientBaked.gatePathLength == documentBaked.gatePathLength, "both consumers agree on the playability gate");
    check(clientBaked.heightSignature == documentBaked.heightSignature, "both consumers fold the same terrain");
    check(clientBaked.biomeSignature == documentBaked.biomeSignature, "both consumers fold the same biomes");
}

/**
 * @brief The emitted block round-trips.
 *
 * @param fromDocument  The recipe the document describes.
 * @param documentBaked What baking it produced.
 */
void checkEmittedBlockRoundTrips(const lpl::procgen::WorldRecipe &fromDocument,
                                 const lpl::procgen::WorldRecipeResult &documentBaked)
{
    const std::string emitted =
        "{\"format\":\"lplscene/1\",\"procedural\":" + lpl::editor::emitSceneRecipe(fromDocument) + ",\"entities\":[]}";
    lpl::procgen::WorldRecipe reparsed{};
    const auto reparsedOk = lpl::editor::parseSceneRecipe(emitted, reparsed);
    lpl::ecs::Registry reparsedWorld;
    const auto reparsedBaked = lpl::procgen::bakeWorld(reparsedWorld, reparsed);
    check(reparsedOk.has_value() && reparsedBaked.stateSignature == documentBaked.stateSignature,
          "an emitted \"procedural\" block re-reads to the same world");
}

/**
 * @brief A cartridge is untrusted input.
 *
 * @details The corrupt image has bits flipped in its recipe payload; the wrong version one has the low byte of
 *          its formatVersion changed.
 *
 * @param image The pack baked from the document.
 */
void checkCartridgeIsUntrustedInput(const std::vector<lpl::core::u8> &image)
{
    lpl::pack::View rejected;
    check(!rejected.open(nullptr, 0u), "a null image is rejected");
    check(!rejected.open(image.data(), static_cast<lpl::core::u32>(image.size()) - 1u),
          "a truncated image is rejected");

    std::vector<lpl::core::u8> corrupt = image;
    corrupt[corrupt.size() - 1u] ^= 0xFFu;
    check(!rejected.open(corrupt.data(), static_cast<lpl::core::u32>(corrupt.size())),
          "a corrupt image fails its content hash");

    std::vector<lpl::core::u8> wrongMagic = image;
    wrongMagic[0] = 'X';
    check(!rejected.open(wrongMagic.data(), static_cast<lpl::core::u32>(wrongMagic.size())),
          "an image with the wrong magic is rejected");

    std::vector<lpl::core::u8> wrongVersion = image;
    wrongVersion[lpl::pack::kMagicSize] = 0xFEu;
    check(!rejected.open(wrongVersion.data(), static_cast<lpl::core::u32>(wrongVersion.size())),
          "an unknown format version is rejected");
}

/**
 * @brief The checked-in reference cartridge is not stale.
 *
 * @details The kernel exercises kParityPackBytes directly (it has no host tool at build time). Asserts here that
 *          those bytes are still what the baker emits, so a recipe change cannot leave the kernel validating a
 *          dead world.
 *
 * @param reference The compiled-in parity recipe.
 */
void checkReferenceCartridgeIsNotStale(const lpl::procgen::WorldRecipe &reference)
{
    const std::vector<lpl::core::u8> referenceImage = lpl::editor::bakeGamePack(reference);
    bool blobMatches = referenceImage.size() == lpl::pack::kParityPackSize;
    for (lpl::core::u32 i = 0u; blobMatches && i < lpl::pack::kParityPackSize; ++i)
        blobMatches = referenceImage[i] == lpl::pack::kParityPackBytes[i];
    check(blobMatches, "the checked-in reference cartridge matches the baker");
}

/**
 * @brief The ecosystem crosses the wire too.
 *
 * @details The world was authorable down to the erosion iteration count while what lived on it was compiled into
 *          the host. These checks are what stop that regressing: a section that round-trips wrong is a cartridge
 *          describing one ecosystem while the game runs another, and nothing on screen would say so.
 *
 *          The decisive check: the same recipe must FOLD the same. A layout that survives a field-by-field
 *          comparison and still changes the simulation is a layout that reorders something the run reads in
 *          order.
 *
 *          A one-section pack is still valid, and reports the absence rather than handing back zeroes: a world
 *          with nothing declared living on it is a legitimate cartridge, not a corrupt one.
 *
 * @param reference The compiled-in parity recipe.
 * @param image     The one-section pack baked from the document.
 */
void checkEcosystemCrossesTheWire(const lpl::procgen::WorldRecipe &reference, const std::vector<lpl::core::u8> &image)
{
    std::printf("\n-- living section --\n");

    const lpl::ecology::LivingRecipe living = lpl::ecology::parityLivingRecipe();
    const std::vector<lpl::core::u8> withLife = lpl::editor::bakeGamePack(reference, &living);

    lpl::pack::View lifeView;
    check(lifeView.open(withLife.data(), static_cast<lpl::core::u32>(withLife.size())), "a two-section pack opens");
    check(lifeView.sectionCount() == 2u, "and carries both sections");

    lpl::pack::RecipeV1 worldWire{};
    check(lifeView.readRecipe(worldWire), "the world section still reads");
    check(worldWire.seed == reference.seed, "and is unchanged by the new neighbour");

    lpl::pack::LivingV1 lifeWire{};
    check(lifeView.readLiving(lifeWire), "the living section reads");

    const lpl::ecology::LivingRecipe decoded = decodeLiving(lifeWire);
    check(decoded.seed == living.seed, "the seed survives the round trip");
    check(decoded.ticks == living.ticks, "the tick count survives");
    check(decoded.speciesCount == living.speciesCount, "every species survives");
    check(decoded.stepSeconds.raw() == living.stepSeconds.raw(), "the step is carried as raw Q16.16");
    check(decoded.headPerBody == living.headPerBody, "so does the body ratio");

    bool webMatches = true;
    for (lpl::core::u32 i = 0u; i < decoded.speciesCount; ++i)
    {
        webMatches = webMatches && decoded.species[i].params.capacity.raw() == living.species[i].params.capacity.raw();
        webMatches = webMatches && decoded.species[i].initial.raw() == living.species[i].initial.raw();
        webMatches = webMatches && decoded.species[i].preyIndex == living.species[i].preyIndex;
        webMatches = webMatches && decoded.species[i].params.level == living.species[i].params.level;
    }
    check(webMatches, "and every species keeps its demography, its prey and its level");

    const lpl::ecology::LivingResult direct = lpl::ecology::runLiving(living);
    const lpl::ecology::LivingResult viaWire = lpl::ecology::runLiving(decoded);
    check(direct.populationSignature == viaWire.populationSignature, "the decoded recipe folds the same populations");
    check(direct.genomeSignature == viaWire.genomeSignature, "the same genomes");
    check(direct.stigmergySignature == viaWire.stigmergySignature, "the same field");
    check(direct.socialSignature == viaWire.socialSignature, "the same social state");

    lpl::pack::View barren;
    check(barren.open(image.data(), static_cast<lpl::core::u32>(image.size())), "a one-section pack still opens");
    lpl::pack::LivingV1 absent{};
    check(!barren.readLiving(absent), "and reports that it declares no ecosystem");

    std::printf("  two-section image = %zu bytes\n", withLife.size());
    std::printf("  species carried   = %u\n", decoded.speciesCount);
}

/**
 * @brief The document can name a SEA, not only a water colour.
 *
 * @details Before these fields the format could say what colour water was and how fast it rippled, and every body
 *          of water it could express came out as the same flat rippling mirror: no swell, no surf, no light coming
 *          through a crest. A viewer that wanted a sea had to have it compiled in, which is the state this whole
 *          format exists to get out of.
 *
 *          And it survives the trip out and back. Field-by-field codecs are exactly where a rename on one side
 *          silently drops a value, and a dropped swell is a sea that quietly goes flat with nothing to say why.
 *
 * @param viewerLook The view profile of the viewer's built-in cartridge.
 */
void checkViewerCartridgeNamesASea(const lpl::pack::ViewV1 &viewerLook)
{
    check(viewerLook.swellHeight > 0.0f, "the viewer's cartridge asks for a swell");
    check(viewerLook.foamGain > 0.0f, "and for foam");
    check(viewerLook.scatterStrength > 0.0f, "and for light through the crests");
    check(viewerLook.foamCrest > 0.0f && viewerLook.foamCrest < 1.0f, "with a crest threshold inside the wave");

    const lpl::engine::ViewProfile decodedLook = lpl::engine::toEngineView(viewerLook);
    const lpl::pack::ViewV1 reWired = lpl::engine::toWireView(decodedLook);
    check(reWired.swellHeight == viewerLook.swellHeight && reWired.crestSharpness == viewerLook.crestSharpness &&
              reWired.chopStrength == viewerLook.chopStrength && reWired.foamColour == viewerLook.foamColour &&
              reWired.scatterColour == viewerLook.scatterColour && reWired.foamGain == viewerLook.foamGain &&
              reWired.foamWidth == viewerLook.foamWidth && reWired.foamCrest == viewerLook.foamCrest &&
              reWired.scatterStrength == viewerLook.scatterStrength,
          "and every field of it round-trips through the codec");
}

/**
 * @brief The viewer's built-in cartridge is not stale either.
 *
 * @details Same concern as the parity blob of checkReferenceCartridgeIsNotStale, for a sharper reason: this is
 *          what a boot with no cartridge slot actually shows, so a stale blob does not fail anything — it
 *          publishes an old world to a browser demo and looks entirely healthy doing it.
 *
 *          It cannot be compared byte for byte here, because it is baked from a FILE
 *          (assets/games/worldview.lplscene) and a test run by `xmake run` has no dependable working directory.
 *          The full validation does that comparison, by regenerating both headers into a scratch copy and
 *          diffing. What is asserted here is the part that catches the likelier fault: a blob regenerated from a
 *          DIFFERENT scene would still open and still parse. So the values worldview.lplscene declares are
 *          hardcoded on purpose: only a statement of what it is supposed to contain catches that.
 *
 *          Three things read "where is the sea": the classifier that decides what is Ocean, the walkability mask,
 *          and the water plane the renderer draws. They have to be the same number, and in this cartridge they
 *          were not — the document said nothing about biomes, so the classifier used the engine default of -4
 *          while the water was drawn at -1. The band between them was land you could not walk on, under water,
 *          that the scatter planted trees in. Nothing was broken anywhere: every layer did exactly what it says it
 *          does. That is what makes this worth an assertion rather than a fix.
 */
void checkViewerCartridgeIsNotStale()
{
    lpl::pack::View viewerView;
    check(viewerView.open(lpl::pack::kViewerPackBytes, lpl::pack::kViewerPackSize),
          "the viewer's built-in cartridge opens");
    check(viewerView.sectionCount() == 3u, "and carries a world, an ecosystem AND a look");

    lpl::pack::RecipeV1 viewerWorld{};
    check(viewerView.readRecipe(viewerWorld), "its world section reads");
    check(viewerWorld.width == 64u && viewerWorld.depth == 64u, "it is the 64x64 demo world");
    check(viewerWorld.seed == 20260728u, "with the seed the document names");

    lpl::pack::LivingV1 viewerLife{};
    check(viewerView.readLiving(viewerLife), "its ecosystem section reads");
    check(viewerLife.speciesCount == 3u, "and declares the three species the document lists");
    check(viewerLife.headPerBody == 3u, "with the body ratio it asks for");

    lpl::pack::ViewV1 viewerLook{};
    check(viewerView.readView(viewerLook), "and its view profile reads");

    const lpl::procgen::WorldRecipe viewerRecipe = decodeRecipe(viewerWorld);
    check(viewerRecipe.biomes.seaLevel == viewerLook.seaLevel, "and the sea it classifies is the sea it draws");

    checkViewerCartridgeNamesASea(viewerLook);
}

/**
 * @brief Every pass is IN the recipe.
 *
 * @details These passes lived briefly in a `Morphology` section, for one reason: RecipeV1's size was treated as
 *          frozen. It is not — there is no released version of this format — so they are back where they belong,
 *          and what has to be asserted is no longer "a pack without them is unchanged" but the thing that
 *          actually matters: a recipe survives the round trip WHOLE, and a byte from disk cannot index a switch
 *          past its end.
 *
 *          Six scatter rules: one per biome is how vegetation is actually written, and four ran out at the fifth
 *          kind of plant. This is the case that could not be expressed at all before, so a viewer that wanted it
 *          had to build its world by hand and could not save, bake or replay what it showed.
 *
 *          EVERY cave kind round-trips, not one of them. The bound in the decoder's clamp was spelled out as a
 *          specific enum value and went stale the day another was added, so a document that said "auto" came
 *          back saying "cellular" — and a test that round-trips a single kind cannot see that, because the one it
 *          picked was under the old bound.
 *
 *          A byte from disk naming a generator this build does not have must be refused, naming its field, not
 *          clamped to one it does have. A count is not trusted either — a cartridge is input, not a promise — and
 *          one out of range is clamped to what the array holds. And the built world must actually differ, or the
 *          passes are decoration.
 *
 * @param fromDocument  The recipe the document describes.
 * @param documentBaked What baking it produced.
 */
void checkEveryPassRoundTripsThroughOneSection(const lpl::procgen::WorldRecipe &fromDocument,
                                               const lpl::procgen::WorldRecipeResult &documentBaked)
{
    std::printf("\n-- every pass round-trips through one section --\n");
    lpl::procgen::WorldRecipe layered = fromDocument;
    layered.caveKind = lpl::procgen::CaveKind::Layered;
    layered.caveSystem.width = layered.width;
    layered.caveSystem.depth = layered.depth;
    layered.terraceSteps = 5u;
    layered.partitionRegions = true;
    layered.raiseBuildings = true;
    layered.buildings.maxFloors = 4u;
    layered.groundClearance = 1.5f;
    layered.roadsideLevels = 3u;
    const char pattern[] = "{[A,P]:2,[BL,P]:1}*,[G,P]";
    for (lpl::core::u32 i = 0u; i < sizeof(pattern); ++i)
        layered.roadsidePattern[i] = pattern[i];

    layered.scatterCount = 6u;
    for (lpl::core::u32 i = 0u; i < 6u; ++i)
    {
        layered.scatter[i].biome = static_cast<lpl::procgen::BiomeId>(i + 1u);
        layered.scatter[i].density = 0.02f * static_cast<lpl::core::f32>(i + 1u);
        layered.scatter[i].tag = 100u + i;
    }

    const std::vector<lpl::core::u8> grown = lpl::editor::bakeGamePack(layered);
    lpl::pack::View grownView;
    check(grownView.open(grown.data(), static_cast<lpl::core::u32>(grown.size())), "the pack opens");
    check(grownView.sectionCount() == 1u, "and carries ONE section, not a recipe split in half");

    lpl::pack::RecipeV1 grownWire{};
    check(grownView.readRecipe(grownWire), "the recipe section reads back");
    const lpl::procgen::WorldRecipe decoded = decodeRecipe(grownWire);

    check(decoded.seed == layered.seed, "the terrain survives");
    check(decoded.caveKind == lpl::procgen::CaveKind::Layered, "the cave kind round-trips");
    for (lpl::core::u32 k = 0u; k <= static_cast<lpl::core::u32>(lpl::procgen::CaveKind::Auto); ++k)
    {
        lpl::procgen::WorldRecipe every = layered;
        every.caveKind = static_cast<lpl::procgen::CaveKind>(k);
        check(decodeRecipe(lpl::pack::toWireRecipe(every)).caveKind == every.caveKind,
              lpl::procgen::caveKindName(every.caveKind));
    }
    check(decoded.terraceSteps == 5u, "so do the terraces");
    check(decoded.partitionRegions, "and the provinces");
    check(decoded.raiseBuildings && decoded.buildings.maxFloors == 4u, "and the shape grammar");
    check(decoded.groundClearance == 1.5f, "and the ground clearance");
    check(decoded.roadsideLevels == 3u, "and the verge levels");
    bool samePattern = true;
    for (lpl::core::u32 i = 0u; i < sizeof(pattern); ++i)
        samePattern = samePattern && decoded.roadsidePattern[i] == pattern[i];
    check(samePattern, "and the L-system arrives character for character");

    check(decoded.scatterCount == 6u, "all six scatter rules survive the wire");
    bool sameRules = true;
    for (lpl::core::u32 i = 0u; i < 6u; ++i)
        sameRules = sameRules && decoded.scatter[i].tag == 100u + i &&
                    decoded.scatter[i].biome == static_cast<lpl::procgen::BiomeId>(i + 1u);
    check(sameRules, "including the two past the old ceiling of four");

    lpl::pack::RecipeV1 corrupt = grownWire;
    corrupt.caveKind = 99u;
    lpl::procgen::WorldRecipe refused{};
    lpl::pack::WireRefusal refusal{};
    check(!lpl::pack::toEngineRecipe(corrupt, refused, refusal) && refusal.field != nullptr &&
              std::string{refusal.field} == "caveKind" && refusal.value == 99u,
          "an out-of-range cave kind is refused, naming its field and its value");
    corrupt = grownWire;
    corrupt.scatterCount = 4000u;
    check(decodeRecipe(corrupt).scatterCount == lpl::pack::kWireScatterRules,
          "an out-of-range scatter count is clamped to what the array holds");

    lpl::ecs::Registry plainWorld;
    lpl::procgen::WorldRecipe unjudged = decoded;
    const auto grownBaked = lpl::procgen::bakeWorld(plainWorld, unjudged);
    check(grownBaked.heightSignature != documentBaked.heightSignature,
          "and the world it describes is genuinely a different world");
}

/**
 * @brief The viewer's world is now sayable.
 *
 * @details This is the shape apps/mapview builds: six scatter rules, a layered cave system, Voronoi provinces,
 *          terracing, the shape grammar, a roadside L-system and a lift that puts the ground above the physics
 *          floor. Until every one of those had a name, that world existed only as two hundred lines of
 *          hand-written WorldBuilder calls inside an app — unsaveable, unbakeable, unreplayable in ring 0, and
 *          with no test target of any kind.
 *
 *          The assertion that matters is not that the fields survive (checkEveryPassRoundTripsThroughOneSection
 *          says that) but that the DOCUMENT does: a world you can only build in C++ is not a world an editor or an
 *          intelligence can direct. The decisive check: field equality can hold while the world differs — that is
 *          what a pass read in the wrong order looks like.
 *
 * @param fromDocument The recipe the document describes.
 */
void checkViewerWorldSurvivesTheDocument(const lpl::procgen::WorldRecipe &fromDocument)
{
    std::printf("\n-- the viewer's world survives the document --\n");
    lpl::procgen::WorldRecipe viewer = fromDocument;
    viewer.width = 48u;
    viewer.depth = 48u;
    viewer.normalizeTerrain = false;
    viewer.groundClearance = 1.5f;
    viewer.terraceSteps = 8u;
    viewer.partitionRegions = true;
    viewer.provinces.cellSize = 8u;
    viewer.provinces.metric = lpl::procgen::DistanceMetric::Chebyshev;
    viewer.caveKind = lpl::procgen::CaveKind::Layered;
    viewer.caveSystem.width = 48u;
    viewer.caveSystem.depth = 48u;
    viewer.caveSystem.layers = 3u;
    viewer.caveSystem.entrances = 3u;
    viewer.raiseBuildings = true;
    viewer.buildings.minFloors = 1u;
    viewer.buildings.maxFloors = 3u;
    viewer.roadsideLevels = 3u;
    const char verge[] = "{[A,P]:2,[BL,P]:1}*,[G,P]";
    for (lpl::core::u32 i = 0u; i < sizeof(verge); ++i)
        viewer.roadsidePattern[i] = verge[i];

    const lpl::procgen::BiomeId kBiomes[] = {lpl::procgen::BiomeId::Taiga,      lpl::procgen::BiomeId::Forest,
                                             lpl::procgen::BiomeId::Rainforest, lpl::procgen::BiomeId::Savanna,
                                             lpl::procgen::BiomeId::Desert,     lpl::procgen::BiomeId::Marsh};
    viewer.scatterCount = 6u;
    for (lpl::core::u32 i = 0u; i < 6u; ++i)
    {
        viewer.scatter[i] = lpl::procgen::ScatterRule{};
        viewer.scatter[i].biome = kBiomes[i];
        viewer.scatter[i].density = 0.03f + 0.02f * static_cast<lpl::core::f32>(i);
        viewer.scatter[i].collidable = i < 3u;
    }

    const std::string document =
        "{\"format\":\"lplscene/1\",\"procedural\":" + lpl::editor::emitSceneRecipe(viewer) + ",\"entities\":[]}";
    lpl::procgen::WorldRecipe reread{};
    const auto ok = lpl::editor::parseSceneRecipe(document, reread);
    check(static_cast<bool>(ok), "the viewer's world emits a document that parses");

    check(reread.scatterCount == 6u, "six scatter rules survive the document");
    check(reread.caveKind == lpl::procgen::CaveKind::Layered, "so does the layered underground");
    check(reread.caveSystem.entrances == 3u, "including the knobs a half-carried generator dropped");
    check(reread.groundClearance == viewer.groundClearance, "and the ground clearance");
    check(reread.terraceSteps == 8u && reread.partitionRegions && reread.raiseBuildings,
          "and the terraces, the provinces and the grammar");
    check(reread.provinces.metric == lpl::procgen::DistanceMetric::Chebyshev,
          "and the province metric, spelled as a word rather than an index");

    lpl::ecs::Registry direct;
    lpl::ecs::Registry viaDocument;
    const auto builtDirect = lpl::procgen::bakeWorld(direct, viewer);
    const auto builtViaDocument = lpl::procgen::bakeWorld(viaDocument, reread);
    check(builtDirect.heightSignature == builtViaDocument.heightSignature,
          "and the world it builds folds identically to the one built from the struct");
    check(builtDirect.biomeSignature == builtViaDocument.biomeSignature, "biomes included");
    check(builtDirect.stateSignature == builtViaDocument.stateSignature, "entities included");
    std::printf("  viewer world: entities=%u height=0x%08X\n", builtDirect.entityCount, builtDirect.heightSignature);
}

} // namespace

/**
 * @brief The document -> pack -> world chain, and the client/server agreement.
 *
 * @details Proves the property the whole two-encoding design rests on: a game authored once as a `.lplscene`
 *          document produces the SAME world in every target that loads it. Two consumers here stand in for the
 *          server and the client — one expands the recipe straight from the document, the other from the baked
 *          pack a constrained target would receive — and their worlds must fold identically.
 *
 *          Also pins the wire format itself: layout sizes, round-tripping, and the rejection of corrupt or
 *          truncated images, since a cartridge is untrusted input and the kernel is the reader least able to
 *          survive a bad one.
 */
int main()
{
    std::printf("== game pack: document -> pack -> world ==\n\n");

    checkWireFormatIsPinned();

    lpl::procgen::WorldRecipe fromDocument{};
    checkDocumentParsesIntoARecipe(fromDocument);

    const lpl::procgen::WorldRecipe reference = lpl::procgen::parityWorldRecipe();
    lpl::procgen::WorldRecipeResult documentBaked{};
    checkDocumentDescribesExactlyTheParityRecipe(reference, fromDocument, documentBaked);

    const std::vector<lpl::core::u8> image = lpl::editor::bakeGamePack(fromDocument);
    lpl::pack::RecipeV1 wire{};
    checkBakedPackReadsBack(image, wire);

    checkClientAndServerAgree(wire, documentBaked);
    checkEmittedBlockRoundTrips(fromDocument, documentBaked);
    checkCartridgeIsUntrustedInput(image);
    checkReferenceCartridgeIsNotStale(reference);
    checkEcosystemCrossesTheWire(reference, image);
    checkViewerCartridgeIsNotStale();
    checkEveryPassRoundTripsThroughOneSection(fromDocument, documentBaked);
    checkViewerWorldSurvivesTheDocument(fromDocument);

    std::printf("\n-- pack --\n");
    std::printf("  image size   = %zu bytes\n", image.size());
    std::printf("  entities     = %u\n", documentBaked.entityCount);
    std::printf("  state_sig    = 0x%08X\n", documentBaked.stateSignature);

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
