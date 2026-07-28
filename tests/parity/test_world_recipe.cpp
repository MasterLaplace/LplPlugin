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

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
