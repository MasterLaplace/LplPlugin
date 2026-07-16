/**
 * @file test_editor_session.cpp
 * @brief Headless proof of the reflection-driven editing model.
 *
 * Drives an EditorSession exactly as an imgui inspector would — generate a world
 * through a command, select an entity, read a component field lane, edit it in
 * human units, read it back, and confirm the edit survives a save→load round
 * trip. Because every access goes through the component reflection registry, the
 * test writes no per-component code; it walks the schema like the UI will.
 *
 * Host-only. Build via xmake: `xmake run test-editor-session`.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <cmath>
#include <cstdio>

#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/editor/EditorSession.hpp>

using namespace lpl;

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

int main()
{
    std::printf("== editor session (reflection-driven editing) ==\n\n");

    editor::EditorSession session;

    // Build a small world through the same command path the UI would use.
    const auto gen = session.command(R"({"cmd":"generate_heightfield","seed":3,"cols":8,"rows":8})");
    check(gen.has_value(), "generate command runs");
    check(session.entityCount() == 64u, "world has 64 entities");

    // Select entity 10 and read its Position.y (a Fixed32 lane, human units).
    session.select(10u);
    check(session.hasSelection(), "selection is valid");

    double y = 0.0;
    const bool got = session.getField(10u, ecs::ComponentId::Position, "value", 1u, y);
    check(got, "read Position.y of the selected entity");
    std::printf("  entity 10 Position.y = %.4f\n", y);

    // Edit it in human units, as an inspector drag would.
    const double newY = 12.5;
    check(session.setField(10u, ecs::ComponentId::Position, "value", 1u, newY), "write Position.y");

    double back = 0.0;
    (void) session.getField(10u, ecs::ComponentId::Position, "value", 1u, back);
    check(std::fabs(back - newY) < 1e-3, "edited value reads back (quantised)");

    // The edit must survive a save -> load round trip (JSON = source of truth).
    const std::string doc = session.save();
    editor::EditorSession reloaded;
    const auto n = reloaded.load(doc);
    check(n.has_value() && n.value() == 64u, "saved scene reloads");
    double reY = 0.0;
    (void) reloaded.getField(10u, ecs::ComponentId::Position, "value", 1u, reY);
    check(std::fabs(reY - newY) < 1e-3, "edit persists through save->load");

    // Enumerate the selected entity's fields the way the inspector will.
    std::printf("\n  -- inspector view of entity 10 --\n");
    const editor::EntityLocation loc = session.locate(10u);
    for (const ecs::ComponentSchema &schema : ecs::allSchemas())
    {
        if (!loc.valid() || !loc.chunk->archetype().has(schema.id))
            continue;
        for (const ecs::FieldDesc &f : schema.fields)
        {
            std::printf("    %.*s.%.*s = [", static_cast<int>(schema.name.size()), schema.name.data(),
                        static_cast<int>(f.name.size()), f.name.data());
            const core::u32 lanes = editor::EditorSession::laneCount(f.type);
            for (core::u32 l = 0; l < lanes; ++l)
            {
                double v = 0.0;
                (void) session.getField(10u, schema.id, f.name, l, v);
                std::printf("%s%.4f", l ? ", " : "", v);
            }
            std::printf("]\n");
        }
    }

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
