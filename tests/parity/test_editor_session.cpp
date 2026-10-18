#include <cmath>
#include <cstdio>

#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/editor/EditorSession.hpp>

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

/**
 * @brief Builds a small world through the same command path the UI would use.
 */
static void checkACommandBuildsTheWorld(lpl::editor::EditorSession &session)
{
    const auto gen = session.command(R"({"cmd":"generate_world","seed":3,"width":8,"depth":8,
                                         "caves":{"enabled":false},"settlement":{"enabled":false},
                                         "gate":{"enabled":false}})");
    check(gen.has_value(), "generate command runs");
    check(session.entityCount() == 64u, "world has 64 entities");
}

/**
 * @brief Selects entity 10 and reads its Position.y (a Fixed32 lane, human units).
 */
static void checkTheSelectedEntityPositionReads(lpl::editor::EditorSession &session)
{
    session.select(10u);
    check(session.hasSelection(), "selection is valid");

    double y = 0.0;
    const bool got = session.getField(10u, lpl::ecs::ComponentId::Position, "value", 1u, y);
    check(got, "read Position.y of the selected entity");
    std::printf("  entity 10 Position.y = %.4f\n", y);
}

/**
 * @brief Edits Position.y in human units, as an inspector drag would, and reads it back.
 */
static void checkAnEditInHumanUnitsReadsBack(lpl::editor::EditorSession &session, double newY)
{
    check(session.setField(10u, lpl::ecs::ComponentId::Position, "value", 1u, newY), "write Position.y");

    double back = 0.0;
    (void) session.getField(10u, lpl::ecs::ComponentId::Position, "value", 1u, back);
    check(std::fabs(back - newY) < 1e-3, "edited value reads back (quantised)");
}

/**
 * @brief The edit must survive a save -> load round trip (JSON = source of truth).
 */
static void checkTheEditSurvivesASaveLoadRoundTrip(const lpl::editor::EditorSession &session, double newY)
{
    const std::string doc = session.save();
    lpl::editor::EditorSession reloaded;
    const auto n = reloaded.load(doc);
    check(n.has_value() && n.value() == 64u, "saved scene reloads");
    double reY = 0.0;
    (void) reloaded.getField(10u, lpl::ecs::ComponentId::Position, "value", 1u, reY);
    check(std::fabs(reY - newY) < 1e-3, "edit persists through save->load");
}

/**
 * @brief Enumerates the selected entity's fields the way the inspector will.
 */
static void printTheInspectorViewOfTheSelectedEntity(const lpl::editor::EditorSession &session)
{
    std::printf("\n  -- inspector view of entity 10 --\n");
    const lpl::editor::EntityLocation loc = session.locate(10u);
    for (const lpl::ecs::ComponentSchema &schema : lpl::ecs::allSchemas())
    {
        if (!loc.valid() || !loc.chunk->archetype().has(schema.id))
            continue;
        for (const lpl::ecs::FieldDesc &f : schema.fields)
        {
            std::printf("    %.*s.%.*s = [", static_cast<int>(schema.name.size()), schema.name.data(),
                        static_cast<int>(f.name.size()), f.name.data());
            const lpl::core::u32 lanes = lpl::editor::EditorSession::laneCount(f.type);
            for (lpl::core::u32 l = 0; l < lanes; ++l)
            {
                double v = 0.0;
                (void) session.getField(10u, schema.id, f.name, l, v);
                std::printf("%s%.4f", l ? ", " : "", v);
            }
            std::printf("]\n");
        }
    }
}

/**
 * @brief The session has a history, not just a world.
 *
 * @details The journal existed with its own test and no caller: a session executed and forgot, so nothing a user
 *          did through the editor could be undone. What this pins is the wiring — that a session edit is recorded,
 *          and that undo restores the exact previous world rather than an approximation of it.
 *
 * Inspection must not enter the history: an undo that depends on how often somebody looked at the world is not an
 * undo.
 */
static void checkTheSessionHasAHistory(lpl::editor::EditorSession &session)
{
    std::printf("\n  -- history --\n");
    const lpl::core::u32 before = session.entityCount();
    check(session.historySize() == 1u, "the generate command was recorded");

    const auto spawn = session.command(R"({"cmd":"spawn_from_template",
            "templates":{"crate":{"Health":{"points":5}}},"name":"crate","count":4})");
    check(spawn.has_value(), "a second command runs");
    check(session.entityCount() == before + 4u, "the world grew");
    check(session.historySize() == 2u, "the second command was recorded");

    check(session.undo(), "undo reports success");
    check(session.entityCount() == before, "undo restores the previous entity count");
    check(session.historySize() == 1u, "undo drops the entry");

    check(session.redo(), "redo reports success");
    check(session.entityCount() == before + 4u, "redo restores the undone world");

    (void) session.command(R"({"cmd":"count"})");
    check(session.historySize() == 2u, "looking at the world does not record anything");

    check(session.history().find("lplcommands/1") != std::string::npos,
          "the history serialises as a replayable document");
}

/**
 * @brief Headless proof of the reflection-driven editing model.
 *
 * Drives an EditorSession exactly as an imgui inspector would — generate a world
 * through a command, select an entity, read a component field lane, edit it in
 * human units, read it back, and confirm the edit survives a save→load round
 * trip. Because every access goes through the component reflection registry, the
 * test writes no per-component code; it walks the schema like the UI will.
 *
 * Host-only. Build via xmake: `xmake run test-editor-session`.
 */
int main()
{
    std::printf("== editor session (reflection-driven editing) ==\n\n");

    lpl::editor::EditorSession session;

    checkACommandBuildsTheWorld(session);

    checkTheSelectedEntityPositionReads(session);

    const double newY = 12.5;
    checkAnEditInHumanUnitsReadsBack(session, newY);

    checkTheEditSurvivesASaveLoadRoundTrip(session, newY);

    printTheInspectorViewOfTheSelectedEntity(session);

    checkTheSessionHasAHistory(session);

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
