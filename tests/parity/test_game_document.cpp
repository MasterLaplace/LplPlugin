#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/GameDocument.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const char *label)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++g_failures;
}

/**
 * @brief A full four-stage game: two scenes, mixed resource scopes, ordered systems.
 */
const char *kGame = R"({
  "format": "lplscene/1",
  "metadata": {
    "title": "Reference Game", "version": "0.1.0", "profile": "mmorpg",
    "startScene": "arena", "online": true,
    "minPlayers": 1, "maxPlayers": 64, "maxInstances": 8
  },
  "scenes": [
    {
      "name": "lobby",
      "systems": ["position", "control", "render"],
      "resources": [
        {"name":"floor",  "path":"tex/floor.png",  "kind":"texture", "scope":"client"},
        {"name":"tuning", "path":"cfg/tuning.json","kind":"data",    "scope":"server"},
        {"name":"colmesh","path":"geo/col.bin",    "kind":"data",    "scope":"shared"}
      ],
      "templates": {"marker": {"Health": {"points": 10}}},
      "entities": [{"$use":"marker"}, {"$use":"marker"}]
    },
    {
      "name": "arena",
      "systems": ["position", "handleCollision", "rigid_body", "render"],
      "textures": {"grass": "tex/grass.png", "rock": "tex/rock.png"},
      "procedural": {
        "seed": 1337, "width": 16, "depth": 16, "cellSize": 0.5,
        "terrain": {"seed": 1337, "frequency": 0.15, "amplitude": 12.0, "octaves": 4},
        "caves": {"enabled": false}, "settlement": {"enabled": false}, "gate": {"enabled": false}
      },
      "entities": []
    }
  ]
})";

/**
 * @brief The shape every document had before the four-stage form existed.
 */
const char *kLegacy = R"({
  "format": "lplscene/1",
  "templates": {"grunt": {"Health": {"points": 30}}},
  "entities": [{"$use":"grunt"}, {"$use":"grunt"}, {"$use":"grunt"}]
})";

/**
 * @brief Stage 1: metadata.
 *
 * @param game The parsed four-stage game.
 */
void checkMetadataIsRead(const lpl::editor::GameDocument &game)
{
    check(game.metadata.title == "Reference Game" && game.metadata.version == "0.1.0", "game metadata is read");
    check(game.metadata.online && game.metadata.maxPlayers == 64u && game.metadata.maxInstances == 8u,
          "server-facing metadata is read");
    check(game.metadata.profile == "mmorpg", "the netcode profile is carried by name");
}

/**
 * @brief Stage 4 bis: scene selection.
 *
 * @param game The parsed four-stage game.
 */
void checkScenesAreSelectedByName(const lpl::editor::GameDocument &game)
{
    check(game.scenes.size() == 2u, "both scenes are parsed");
    check(game.findScene("lobby") != nullptr && game.findScene("arena") != nullptr, "scenes are addressable by name");
    check(game.findScene("nope") == nullptr, "an unknown scene name resolves to nothing");
    check(game.startScene() != nullptr && game.startScene()->name == "arena", "startScene selects the declared scene");
}

/**
 * @brief Stage 3: ordered systems.
 *
 * @param lobby The lobby scene.
 * @param arena The arena scene.
 */
void checkSystemListsKeepTheirOrder(const lpl::editor::SceneDescription &lobby,
                                    const lpl::editor::SceneDescription &arena)
{
    check(lobby.systems.size() == 3u && lobby.systems[0] == "position" && lobby.systems[2] == "render",
          "the system list keeps its declaration ORDER");
    check(arena.systems.size() == 4u && arena.systems[1] == "handleCollision", "each scene has its own system list");
}

/**
 * @brief Stage 2: named resources, scoped by consumer.
 *
 * @details Flakkari's per-kind arrays are accepted, and imply a client scope.
 *
 * @param lobby The lobby scene, with a flat resource table.
 * @param arena The arena scene, with a Flakkari per-kind texture map.
 */
void checkResourcesAreScopedByConsumer(const lpl::editor::SceneDescription &lobby,
                                       const lpl::editor::SceneDescription &arena)
{
    check(lobby.resources.size() == 3u, "the flat resource table is read");
    const auto lobbyClient = lpl::editor::resourcesFor(lobby, lpl::editor::ResourceScope::Client);
    const auto lobbyServer = lpl::editor::resourcesFor(lobby, lpl::editor::ResourceScope::Server);
    check(lobbyClient.size() == 2u, "a client gets its own resources plus the shared ones");
    check(lobbyServer.size() == 2u, "a server gets its own resources plus the shared ones");

    bool clientSeesServerOnly = false;
    for (const auto *resource : lobbyClient)
        if (resource->name == "tuning")
            clientSeesServerOnly = true;
    check(!clientSeesServerOnly, "a client is NOT handed the server-only resource");

    bool serverSeesTexture = false;
    for (const auto *resource : lobbyServer)
        if (resource->name == "floor")
            serverSeesTexture = true;
    check(!serverSeesTexture, "a server is NOT handed the texture");

    bool bothSeeShared = false;
    for (const auto *resource : lobbyServer)
        if (resource->name == "colmesh")
            bothSeeShared = true;
    check(bothSeeShared, "the shared resource reaches both");

    check(arena.resources.size() == 2u, "Flakkari's per-kind texture map is read");
    check(lpl::editor::resourcesFor(arena, lpl::editor::ResourceScope::Server).empty(),
          "textures declared the Flakkari way are client-scoped implicitly");
}

/**
 * @brief A scene is a Registry.
 *
 * @param lobby     The lobby scene.
 * @param arena     The arena scene.
 * @param arenaFold Receives the fold of the world the arena instantiates.
 */
void checkSceneIsARegistry(const lpl::editor::SceneDescription &lobby, const lpl::editor::SceneDescription &arena,
                           lpl::core::u32 &arenaFold)
{
    lpl::ecs::Registry lobbyWorld;
    const auto lobbyCount = lpl::editor::instantiateScene(lobby, lobbyWorld);
    check(lobbyCount.has_value() && lobbyCount.value() == 2u, "the lobby instantiates its explicit entities");

    lpl::ecs::Registry arenaWorld;
    const auto arenaCount = lpl::editor::instantiateScene(arena, arenaWorld);
    check(arenaCount.has_value() && arenaCount.value() == 256u, "the arena instantiates its procedural pass");

    arenaFold = lpl::procgen::foldWorldState(arenaWorld);
    check(lpl::procgen::foldWorldState(lobbyWorld) != arenaFold, "two scenes of one document are two different worlds");

    lpl::ecs::Registry arenaAgain;
    (void) lpl::editor::instantiateScene(arena, arenaAgain);
    check(lpl::procgen::foldWorldState(arenaAgain) == arenaFold, "instantiating a scene twice folds identically");
}

/**
 * @brief Round trip.
 *
 * @param game      The parsed four-stage game.
 * @param arena     Its arena scene.
 * @param arenaFold The fold of the world the arena instantiates.
 */
void checkDocumentSurvivesTheRoundTrip(const lpl::editor::GameDocument &game,
                                       const lpl::editor::SceneDescription &arena, lpl::core::u32 arenaFold)
{
    const std::string emitted = lpl::editor::emitGameDocument(game);
    const auto reparsed = lpl::editor::parseGameDocument(emitted);
    check(reparsed.has_value(), "an emitted document parses back");
    check(reparsed->metadata.maxInstances == game.metadata.maxInstances &&
              reparsed->scenes.size() == game.scenes.size(),
          "metadata and scene count survive the round trip");
    check(reparsed->findScene("arena") != nullptr && reparsed->findScene("arena")->systems == arena.systems,
          "system order survives the round trip");
    check(lpl::editor::resourcesFor(*reparsed->findScene("lobby"), lpl::editor::ResourceScope::Server).size() == 2u,
          "resource scopes survive the round trip");

    lpl::ecs::Registry roundTripped;
    (void) lpl::editor::instantiateScene(*reparsed->findScene("arena"), roundTripped);
    check(lpl::procgen::foldWorldState(roundTripped) == arenaFold, "a round-tripped scene rebuilds the SAME world");
}

/**
 * @brief Backward compatibility.
 */
void checkLegacyDocumentsStillLoad()
{
    const auto legacy = lpl::editor::parseGameDocument(kLegacy);
    check(legacy.has_value(), "a document with no scenes array still parses");
    check(legacy->scenes.size() == 1u && legacy->scenes[0].name == "main", "it is treated as one implicit scene");
    lpl::ecs::Registry legacyWorld;
    const auto legacyCount = lpl::editor::instantiateScene(legacy->scenes[0], legacyWorld);
    check(legacyCount.has_value() && legacyCount.value() == 3u, "the implicit scene instantiates its entities");

    check(!lpl::editor::parseGameDocument(R"({"format":"nope"})").has_value(), "an unknown format is refused");
    check(!lpl::editor::parseGameDocument("{ not json").has_value(), "malformed JSON is refused");
}

} // namespace

/**
 * @brief The four-stage document: metadata, resources, systems, scenes.
 *
 * @details Checks the properties the report's §5.2 actually asks for, rather than that the fields survive a
 *          round trip:
 *
 *           - a server and a client reading the SAME document each get their own slice of the resource table,
 *             and neither is handed the other's;
 *           - the system list keeps its ORDER, because the order is the schedule;
 *           - a scene is a Registry: two scenes of one document build two different worlds, and instantiating
 *             the same scene twice folds identically;
 *           - documents written before this stage existed still load unchanged.
 */
int main()
{
    std::printf("== four-stage game document ==\n\n");

    const auto parsed = lpl::editor::parseGameDocument(kGame);
    check(parsed.has_value(), "the four-stage document parses");
    if (!parsed)
    {
        std::printf("\nFAILURES (1 failures)\n");
        return 1;
    }
    const lpl::editor::GameDocument &game = *parsed;

    checkMetadataIsRead(game);
    checkScenesAreSelectedByName(game);

    const lpl::editor::SceneDescription &lobby = *game.findScene("lobby");
    const lpl::editor::SceneDescription &arena = *game.findScene("arena");

    checkSystemListsKeepTheirOrder(lobby, arena);
    checkResourcesAreScopedByConsumer(lobby, arena);
    lpl::core::u32 arenaFold = 0;
    checkSceneIsARegistry(lobby, arena, arenaFold);
    checkDocumentSurvivesTheRoundTrip(game, arena, arenaFold);
    checkLegacyDocumentsStillLoad();

    std::printf("\n-- document --\n");
    std::printf("  title      = %s\n", game.metadata.title.c_str());
    std::printf("  scenes     = %zu\n", game.scenes.size());
    std::printf("  arena fold = 0x%08X\n", arenaFold);

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
