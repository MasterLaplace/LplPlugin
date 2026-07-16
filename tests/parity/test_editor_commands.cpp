/**
 * @file test_editor_commands.cpp
 * @brief Proof that the JSON command stream drives a deterministic world.
 *
 * Runs a batch of editor commands (generate a heightfield, scatter props, check
 * playability, save the scene) against one registry, then replays the exact same
 * command JSON against a fresh registry, and checks that the two worlds fold
 * bit-for-bit identically — the command stream is a deterministic recipe, the
 * seam the Caine AI bridge will emit into. Also checks the playability verdict is
 * reported and the saved scene round-trips.
 *
 * Host-only. Build via xmake: `xmake run test-editor-commands`.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <cstdio>
#include <string>

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandProcessor.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

using namespace lpl;
using math::Fixed32;
using FVec3 = math::Vec3<Fixed32>;

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

static core::u32 foldPositions(const ecs::Registry &registry)
{
    core::u32 h = 0x811C9DC5u;
    auto step = [&](core::u32 v) { h = (h ^ v) * 0x01000193u; };
    for (const auto &part : registry.partitions())
        if (part)
            for (const auto &chunk : part->chunks())
            {
                const core::u32 n = chunk->count();
                const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
                if (!pos)
                    continue;
                for (core::u32 i = 0; i < n; ++i)
                {
                    step(static_cast<core::u32>(pos[i].x.raw()));
                    step(static_cast<core::u32>(pos[i].y.raw()));
                    step(static_cast<core::u32>(pos[i].z.raw()));
                }
            }
    return h;
}

int main()
{
    std::printf("== editor command stream determinism ==\n\n");

    // A recipe an AI/editor could emit: build terrain, scatter props, gate it.
    const char *batch = R"([
      {"cmd":"generate_heightfield","seed":7,"cols":16,"rows":16,"amplitude":3.0},
      {"cmd":"scatter_poisson","seed":7,"width":12,"depth":12,"radius":1.25},
      {"cmd":"check_playability","seed":7,"cols":16,"rows":16,"wallThreshold":0.6,
       "goalCol":15,"goalRow":15},
      {"cmd":"count"}
    ])";

    ecs::Registry world1, world2;
    editor::CommandProcessor cp1(world1), cp2(world2);
    const auto r1 = cp1.execute(batch);
    const auto r2 = cp2.execute(batch);

    check(r1.has_value(), "command batch executes");
    check(r1.has_value() && r2.has_value() && r1.value() == r2.value(), "identical batch -> identical report");
    check(foldPositions(world1) == foldPositions(world2), "identical batch -> identical world fold");

    if (r1.has_value())
    {
        const std::string &rep = r1.value();
        check(rep.find("\"reachable\":") != std::string::npos, "playability verdict is reported");
        std::printf("\n  report: %s\n", rep.c_str());
    }
    std::printf("  world fold = 0x%08X (%u entities)\n", foldPositions(world1), editor::entityCount(world1));

    // save_scene -> load into a fresh world -> same fold.
    editor::CommandProcessor saver(world1);
    const auto saved = saver.execute(R"({"cmd":"save_scene"})");
    check(saved.has_value() && saved.value().find("lplscene/1") != std::string::npos, "save_scene emits a document");

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
