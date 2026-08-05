/**
 * @file test_world_recipe.cpp
 * @brief Linux oracle for the procedural world parity gate.
 *
 * Bakes lpl::procgen::parityWorldRecipe() and prints the signatures the i686
 * kernel must reproduce bit for bit (libengine/src/smoke/p7_procgen_smoke.cpp,
 * reported on serial at boot). A world generated from a seed is authoritative
 * state, so it falls under the same HARD determinism contract as the CubePile
 * simulation: same recipe, same fold, on both targets.
 *
 * Also asserts the properties that make the gate meaningful rather than merely
 * stable — that the recipe is reproducible, that a different seed diverges, and
 * that the world passes its own playability check.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/ecs/Registry.hpp>
#include <lpl/procgen/WorldAtlas.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool condition, const char *label)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++g_failures;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== procedural world recipe parity (Linux oracle) ==\n\n");

    const procgen::WorldRecipe recipe = procgen::parityWorldRecipe();

    ecs::Registry world;
    const procgen::WorldRecipeResult baked = procgen::bakeWorld(world, recipe);

    // ── Reproducibility: the same recipe rebuilds the same bits ─────────────
    ecs::Registry twin;
    const procgen::WorldRecipeResult rebaked = procgen::bakeWorld(twin, recipe);
    check(baked.entityCount == rebaked.entityCount, "same recipe yields the same entity count");
    check(baked.stateSignature == rebaked.stateSignature, "same recipe folds bit-for-bit identically");
    check(baked.heightSignature == rebaked.heightSignature, "the terrain grid folds identically");
    check(baked.biomeSignature == rebaked.biomeSignature, "the biome map folds identically");
    check(baked.gatePathLength == rebaked.gatePathLength, "playability verdict is reproducible");

    // ── Sensitivity: the seed actually drives the world ─────────────────────
    procgen::WorldRecipe other = recipe;
    other.seed = 2024u;
    other.terrain.seed = 2024u;
    ecs::Registry variant;
    const procgen::WorldRecipeResult varied = procgen::bakeWorld(variant, other);
    check(varied.stateSignature != baked.stateSignature, "a different seed yields a different world");
    check(varied.heightSignature != baked.heightSignature, "a different seed yields different terrain");

    // ── The passes the recipe asks for actually ran ─────────────────────────
    // A recipe that silently skipped erosion or the underground would still fold
    // stably, and stably folding nothing is exactly the failure the gate exists
    // to catch. So each pass has to show a trace in the result.
    check(baked.entityCount > 0u, "the recipe materialises entities");
    check(baked.riverCells > 0u, "the drainage network carved rivers");
    check(baked.dungeonFloor > 0u, "the underground layer has open cells");
    check(baked.roadCells > 0u, "the road network was grown");
    check(baked.gateReachable == 1u, "the generated world is playable (goal reachable)");
    check(baked.ok == 1u, "recipe passes every gate");

    std::printf("\n-- signatures the kernel must reproduce --\n");
    std::printf("  entities   = %u\n", baked.entityCount);
    std::printf("  state_sig  = 0x%08X\n", baked.stateSignature);
    std::printf("  height_sig = 0x%08X\n", baked.heightSignature);
    std::printf("  biome_sig  = 0x%08X\n", baked.biomeSignature);
    std::printf("  rivers     = %u\n", baked.riverCells);
    std::printf("  roads      = %u\n", baked.roadCells);
    std::printf("  lakes      = %u\n", baked.lakeCells);
    std::printf("  cave_floor = %u\n", baked.dungeonFloor);
    std::printf("  plots      = %u\n", baked.settlementPlots);
    std::printf("  reachable  = %u\n", baked.gateReachable);
    std::printf("  visited    = %u\n", baked.gateVisited);
    std::printf("  path_len   = %u\n", baked.gatePathLength);

    // ── The passes a recipe could not name until now ──────────────────────────
    //
    // Four underground generators existed and a recipe could ask for exactly one of
    // them; provinces, terraces, the building grammar and the roadside decoration
    // could not be asked for at all. They were reachable only by writing
    // WorldBuilder calls by hand, which is what a viewer did — so its world could
    // not be saved, baked, replayed in ring 0, or asked for by an intelligence.
    //
    // Each check below is the anti-orphan proof for one of them: a switch that does
    // not change the world it claims to change is a field, not a feature.
    std::printf("\n-- the passes a recipe could not name --\n");

    const auto bakeWith = [](const procgen::WorldRecipe &recipe) {
        ecs::Registry registry;
        return procgen::bakeWorld(registry, recipe);
    };

    {
        procgen::WorldRecipe stepped = procgen::parityWorldRecipe();
        stepped.terraceSteps = 6u;
        const procgen::WorldRecipeResult result = bakeWith(stepped);
        check(result.heightSignature != baked.heightSignature, "terracing reshapes the terrain");
    }
    {
        procgen::WorldRecipe divided = procgen::parityWorldRecipe();
        divided.partitionRegions = true;
        divided.provinces.width = divided.width;
        divided.provinces.depth = divided.depth;
        divided.provinces.cellSize = 6u;
        const procgen::WorldRecipeResult result = bakeWith(divided);
        // Provinces partition the surface without moving it, so the HEIGHT must not
        // budge — that is the claim, and a pass that moved it would be a bug.
        check(result.heightSignature == baked.heightSignature, "provinces do not disturb the terrain");
        check(result.ok == 1u, "and the world still passes its gate");
    }
    {
        // Every alternative underground must actually DIG something, and the check has
        // to read the grid each generator fills rather than one summary field.
        //
        // The first version of this asked bakeWorld for `dungeonFloor` and passed for
        // the wrong reason: the layered system fills its own CaveSystem and leaves
        // `_dungeon` empty, so it reported ZERO open cells and still counted as
        // "different from the default". A check that cannot fail for the right reason
        // is worse than no check.
        const auto atlasFor = [](procgen::CaveKind kind) {
            procgen::WorldRecipe underground = procgen::parityWorldRecipe();
            underground.caveKind = kind;
            underground.rooms.width = underground.width;
            underground.rooms.depth = underground.depth;
            underground.aggregation.width = underground.width;
            underground.aggregation.depth = underground.depth;
            underground.aggregation.particles = underground.width * 8u;
            underground.caveSystem.width = underground.width;
            underground.caveSystem.depth = underground.depth;
            // The gate judges the FLAT plan, so it cannot speak about a layered system
            // at all — see the note on CaveKind::Layered. Leaving it on here would test
            // the gate rather than the generator.
            underground.checkPlayability = false;
            return procgen::buildAtlas(underground, nullptr, nullptr);
        };

        core::u32 openFor[3] = {0u, 0u, 0u};
        const procgen::CaveKind kinds[3] = {procgen::CaveKind::Bsp, procgen::CaveKind::Dla,
                                           procgen::CaveKind::Layered};
        for (core::u32 i = 0u; i < 3u; ++i)
        {
            const procgen::WorldAtlas atlas = atlasFor(kinds[i]);
            if (kinds[i] == procgen::CaveKind::Layered)
                openFor[i] = atlas.caveSystem.hollowCells;
            else
                for (core::u32 z = 0u; z < atlas.dungeon.depth(); ++z)
                    for (core::u32 x = 0u; x < atlas.dungeon.width(); ++x)
                        if (procgen::isWalkable(atlas.dungeon.at(x, z)))
                            ++openFor[i];
            std::printf("  %-9s open cells = %u\n", procgen::caveKindName(kinds[i]), openFor[i]);
        }
        check(openFor[0] > 0u, "the room partition digs rooms");
        check(openFor[1] > 0u, "the aggregation digs a branching cave");
        check(openFor[2] > 0u, "the layered system digs layers");
        check(openFor[0] != openFor[1], "and they are not the same cave");
    }
    {
        procgen::WorldRecipe raised = procgen::parityWorldRecipe();
        raised.raiseBuildings = true;
        const procgen::WorldRecipeResult result = bakeWith(raised);
        // The grammar raises voxels; it does not move the ground or the plots.
        check(result.heightSignature == baked.heightSignature, "raising a town does not move the ground");
        check(result.settlementPlots == baked.settlementPlots, "nor change how many plots there are");
    }

    // ── A word, never an index ────────────────────────────────────────────────
    std::printf("\n-- named, so a reordering cannot reinterpret a document --\n");
    for (core::u32 i = 0u; i <= static_cast<core::u32>(procgen::CaveKind::Layered); ++i)
    {
        const procgen::CaveKind kind = static_cast<procgen::CaveKind>(i);
        procgen::CaveKind round = procgen::CaveKind::Cellular;
        if (!procgen::caveKindByName(procgen::caveKindName(kind), round) || round != kind)
        {
            check(false, "every cave kind round-trips through its own name");
            break;
        }
        if (i == static_cast<core::u32>(procgen::CaveKind::Layered))
            check(true, "every cave kind round-trips through its own name");
    }
    procgen::CaveKind rejected = procgen::CaveKind::Layered;
    check(!procgen::caveKindByName("spelunk", rejected), "an unknown word is refused, not defaulted");
    check(rejected == procgen::CaveKind::Layered, "and the caller's value is left alone");
    procgen::DistanceMetric metric = procgen::DistanceMetric::Euclidean;
    check(procgen::distanceMetricByName("chebyshev", metric) && metric == procgen::DistanceMetric::Chebyshev,
          "province metrics are named too");

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
