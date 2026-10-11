#include <lpl/ecs/Registry.hpp>
#include <lpl/pack/GamePack.hpp>
#include <lpl/pack/ParityPackBlob.hpp>
#include <lpl/pack/RecipeCodec.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/std/memory.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(world_recipe);

namespace {

/**
 * @brief Bakes @p recipe into a registry of its own, on the heap: a kernel stack cannot hold one,
 *        and a bake folds everything its registry holds.
 */
[[nodiscard]] lpl::procgen::WorldRecipeResult bake(const lpl::procgen::WorldRecipe &recipe)
{
    const auto registry = lpl::pmr::make_unique<lpl::ecs::Registry>();

    return lpl::procgen::bakeWorld(*registry, recipe);
}

} // namespace

/**
 * @brief Gate P7 world: the recipe read from the reference game pack, through the reader that
 *        ships, bakes the parity world, the same on both targets.
 *
 * @details Every pass the recipe asks for has to leave a trace: a recipe that silently skipped
 *          erosion or the underground would fold just as stably.
 */
LPL_TEST(reference_pack_bakes_the_parity_world)
{
    lpl::pack::View view;
    lpl::pack::RecipeV1 wire{};
    lpl::procgen::WorldRecipe recipe{};
    lpl::pack::WireRefusal refusal{};

    if (!test.check(view.open(lpl::pack::kParityPackBytes, lpl::pack::kParityPackSize) && view.readRecipe(wire) &&
                        lpl::pack::toEngineRecipe(wire, recipe, refusal),
                    "the reference pack opens and its recipe decodes"))
        return;

    const lpl::procgen::WorldRecipeResult baked = bake(recipe);

    test.check(baked == bake(lpl::procgen::parityWorldRecipe()),
               "it bakes the world the parity recipe bakes, field for field");
    test.check(baked.entityCount > 0u, "the world has entities");
    test.check(baked.riverCells > 0u, "the drainage carved rivers");
    test.check(baked.dungeonFloor > 0u, "the underground has open cells");
    test.check(baked.roadCells > 0u, "roads were grown");
    test.check(baked.gateReachable == 1u, "the goal is reachable");
    test.check(baked.ok == 1u, "and the world passes its own gate");

    test.measure("entities", baked.entityCount);
    test.measureHexadecimal("state_signature", baked.stateSignature);
    test.measureHexadecimal("height_signature", baked.heightSignature);
    test.measureHexadecimal("biome_signature", baked.biomeSignature);
    test.measure("river_cells", baked.riverCells);
    test.measure("road_cells", baked.roadCells);
    test.measure("lake_cells", baked.lakeCells);
    test.measure("cave_floor_cells", baked.dungeonFloor);
    test.measure("settlement_plots", baked.settlementPlots);
    test.measure("cells_visited", baked.gateVisited);
    test.measure("path_length", baked.gatePathLength);
}

LPL_TEST(same_recipe_same_world_other_seed_other_world)
{
    const lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    const lpl::procgen::WorldRecipeResult baked = bake(recipe);
    const lpl::procgen::WorldRecipeResult again = bake(recipe);
    lpl::procgen::WorldRecipe other = recipe;

    other.seed = 2024u;
    other.terrain.seed = 2024u;

    const lpl::procgen::WorldRecipeResult varied = bake(other);

    test.check(again == baked, "the same recipe bakes the same world, field for field");
    test.check(varied.stateSignature != baked.stateSignature, "another seed bakes another world");
    test.check(varied.heightSignature != baked.heightSignature, "on other terrain");
}
