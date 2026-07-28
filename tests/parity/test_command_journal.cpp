/**
 * @file test_command_journal.cpp
 * @brief Undo/redo and deterministic replay, as the report's §5.3 asks for them.
 *
 * The claim under test is that a journal of serialisable commands buys undo,
 * redo and replay for free — without any command having to know how to reverse
 * itself. So the checks are all about world signatures: after an undo the world
 * must fold exactly what it folded before the undone command, and a journal
 * serialised and replayed into a fresh registry must fold what the original
 * did. Anything less and the journal would be bookkeeping rather than a
 * reconstruction.
 *
 * Also covers the inspection commands, which the report calls the half every
 * design forgets.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandJournal.hpp>
#include <lpl/editor/CommandProcessor.hpp>
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

/// Does @p haystack contain @p needle?
bool contains(const std::string &haystack, const char *needle) { return haystack.find(needle) != std::string::npos; }

/// Ground only: the underground, the town and the verdict are switched off, so
/// this command is the terrain and nothing else.
const char *kTerrain = R"({"cmd":"generate_world","seed":1337,"width":16,"depth":16,"cellSize":0.5,
                           "terrain":{"seed":1337,"frequency":0.15,"amplitude":12.0,"octaves":4},
                           "caves":{"enabled":false},"settlement":{"enabled":false},
                           "gate":{"enabled":false}})";
/// Props only: same pipeline, `materializeGround` off, so it adds trees to the
/// world the previous command laid down instead of a second ground.
const char *kScatter = R"({"cmd":"generate_world","seed":4242,"width":16,"depth":16,"cellSize":0.5,
                           "materializeGround":false,
                           "caves":{"enabled":false},"settlement":{"enabled":false},
                           "gate":{"enabled":false},
                           "scatter":[{"biome":"grassland","density":0.1,"halfExtent":0.2}]})";

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== command journal: undo, redo, deterministic replay ==\n\n");

    ecs::Registry world;
    editor::CommandJournal journal{world};

    // ── Recording ───────────────────────────────────────────────────────────
    check(journal.execute(kTerrain).has_value(), "generate_world (ground) runs");
    const core::u32 afterTerrainCount = procgen::foldWorldState(world) != 0u ? 1u : 0u;
    (void) afterTerrainCount;
    const core::u32 terrainFold = procgen::foldWorldState(world);
    check(journal.size() == 1u, "a mutating command is recorded");

    check(journal.execute(kScatter).has_value(), "generate_world (props) runs");
    const core::u32 scatterFold = procgen::foldWorldState(world);
    check(journal.size() == 2u, "the second mutating command is recorded");
    check(terrainFold != scatterFold, "the second command changed the world");

    // ── Inspection does not pollute the journal ─────────────────────────────
    const auto stats = journal.execute(R"({"cmd":"get_world_stats"})");
    check(stats.has_value(), "get_world_stats runs");
    check(journal.size() == 2u, "an inspection command is NOT recorded");
    check(contains(*stats, "\"entities\":") && contains(*stats, "\"archetypes\":") &&
              contains(*stats, "\"stateSignature\":") && contains(*stats, "\"boundsMinRaw\":"),
          "world stats report counts, bounds and a signature");
    check(contains(*stats, "\"Position\":"), "world stats break down per component");

    const auto queryAll = journal.execute(R"({"cmd":"query_entities","limit":4})");
    check(queryAll.has_value() && contains(*queryAll, "\"matched\":"), "query_entities reports a match count");
    check(contains(*queryAll, "\"truncated\":true"), "an over-limit query says so instead of dumping everything");

    const auto queryBox =
        journal.execute(R"({"cmd":"query_entities","with":"Position","minX":-1.0,"maxX":1.0,"limit":64})");
    check(queryBox.has_value(), "query_entities accepts a component + box filter");
    check(!contains(*queryBox, "\"matched\":0,"), "the box filter still matches entities near the origin");

    const auto badFilter = journal.execute(R"({"cmd":"query_entities","with":"NotAComponent"})");
    check(badFilter.has_value() && contains(*badFilter, "\"ok\":false"), "an unknown component is refused, not ignored");

    // ── Undo is a rebuild, not an inverse ───────────────────────────────────
    check(journal.undo(), "undo reports success");
    check(journal.size() == 1u, "undo drops the last entry");
    check(procgen::foldWorldState(world) == terrainFold, "undo restores the EXACT previous world");

    check(journal.redo(), "redo reports success");
    check(journal.size() == 2u, "redo restores the entry");
    check(procgen::foldWorldState(world) == scatterFold, "redo restores the EXACT undone world");

    check(journal.undo() && journal.undo(), "undo unwinds to the start");
    check(procgen::foldWorldState(world) != terrainFold, "the emptied world folds differently");
    check(!journal.undo(), "undo on an empty journal reports failure");

    check(journal.redo() && journal.redo(), "redo replays back to the top");
    check(procgen::foldWorldState(world) == scatterFold, "the world is back where it was");
    check(!journal.redo(), "redo past the top reports failure");

    // ── A new action prunes the redo branch ─────────────────────────────────
    check(journal.undo(), "undo once more");
    check(journal.redoSize() == 1u, "one command is available to redo");
    check(journal.execute(kScatter).has_value(), "a new command runs");
    check(journal.redoSize() == 0u, "a new action discards the redo branch");

    // ── spawn_from_template: the pointwise placement command ────────────────
    {
        const char *spawn = R"({"cmd":"spawn_from_template",
            "templates":{"grunt":{"Health":{"points":30}},
                         "elite":{"$use":"grunt","Health":{"points":90}}},
            "name":"elite","count":3})";
        const core::u32 before = editor::entityCount(world);
        const auto report = journal.execute(spawn);
        check(report.has_value() && contains(*report, "\"created\":3"), "spawn_from_template places N instances");
        check(editor::entityCount(world) == before + 3u, "the world grew by exactly N");
        check(journal.size() == 3u, "spawning is recorded as a mutation");
    }

    // ── The journal IS the recipe ───────────────────────────────────────────
    const std::string document = journal.toJson();
    check(contains(document, "\"format\":\"lplcommands/1\""), "the journal serialises with a versioned format");

    const core::u32 originalFold = procgen::foldWorldState(world);

    ecs::Registry replayed;
    editor::CommandJournal replayJournal{replayed};
    const auto count = replayJournal.replay(document);
    check(count.has_value() && count.value() == journal.size(), "replay restores every command");
    check(procgen::foldWorldState(replayed) == originalFold,
          "a replayed journal rebuilds a bit-identical world (the parity slice)");

    // ── Malformed input never enters the journal ────────────────────────────
    const core::u32 before = journal.size();
    check(!journal.execute("{ this is not json").has_value(), "malformed JSON is rejected");
    check(journal.size() == before, "a rejected command is not recorded");
    check(!replayJournal.replay(R"({"format":"nope","commands":[]})").has_value(),
          "an unknown journal format is refused");

    std::printf("\n-- world --\n");
    std::printf("  commands   = %u\n", journal.size());
    std::printf("  state_sig  = 0x%08X\n", originalFold);

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
