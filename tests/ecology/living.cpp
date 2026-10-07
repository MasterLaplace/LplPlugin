#include <lpl/ecology/LivingRecipe.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(living);

/**
 * @brief Gate P8 living: populations, genomes, a pheromone field and a social layer, run for the
 *        parity recipe's ticks, fold the same on both targets.
 *
 * @details A fold proves that two runs agree, not that the run was worth agreeing on: the run
 *          must also move away from its initial conditions, and follow its seed and its ticks.
 */
LPL_TEST(the_parity_world_lives_the_same_way)
{
    const lpl::ecology::LivingRecipe recipe = lpl::ecology::parityLivingRecipe();
    const lpl::ecology::LivingResult run = lpl::ecology::runLiving(recipe);

    test.check(run.ok == 1u, "the recipe produces a usable run");
    test.check(run.trailCells != 0u, "the field holds a trail above its evaporation floor");
    test.check(run.realisedRooms != 0u && run.realisedRooms <= recipe.budget.maxRealisedRooms,
               "the budget realises rooms, and no more than it allows");
    test.check(run.migrations != 0u, "creatures migrate between rooms");

    test.measureHexadecimal("population_signature", run.populationSignature);
    test.measureHexadecimal("genome_signature", run.genomeSignature);
    test.measureHexadecimal("stigmergy_signature", run.stigmergySignature);
    test.measureHexadecimal("social_signature", run.socialSignature);
    test.measure("extinctions", run.extinctions);
    test.measure("anomalies", run.anomalies);
    test.measure("realised_rooms", run.realisedRooms);
    test.measure("migrations", run.migrations);
    test.measure("alpha_changes", run.alphaChanges);
    test.measure("trail_cells", run.trailCells);
}

LPL_TEST(the_run_moves_and_follows_its_seed_and_ticks)
{
    const lpl::ecology::LivingRecipe recipe = lpl::ecology::parityLivingRecipe();
    const lpl::ecology::LivingResult run = lpl::ecology::runLiving(recipe);
    const lpl::ecology::LivingResult twin = lpl::ecology::runLiving(recipe);
    lpl::ecology::LivingRecipe still = recipe;
    lpl::ecology::LivingRecipe otherSeed = recipe;
    lpl::ecology::LivingRecipe halfway = recipe;

    still.ticks = 0u;
    otherSeed.seed = recipe.seed + 1u;
    halfway.ticks = recipe.ticks / 2u;

    const lpl::ecology::LivingResult initial = lpl::ecology::runLiving(still);
    const lpl::ecology::LivingResult elsewhere = lpl::ecology::runLiving(otherSeed);
    const lpl::ecology::LivingResult shorter = lpl::ecology::runLiving(halfway);

    test.check(run.populationSignature != initial.populationSignature, "the populations evolve");
    test.check(run.genomeSignature != initial.genomeSignature, "the genomes drift from the founders");
    test.check(run.stigmergySignature != initial.stigmergySignature, "the pheromone field changes");
    test.check(run.socialSignature != initial.socialSignature, "the social layer reorganises");
    test.check(twin.populationSignature == run.populationSignature && twin.genomeSignature == run.genomeSignature &&
                   twin.stigmergySignature == run.stigmergySignature && twin.socialSignature == run.socialSignature,
               "a second run folds the same state");
    test.check(elsewhere.stigmergySignature != run.stigmergySignature, "another seed gives another field");
    test.check(elsewhere.socialSignature != run.socialSignature, "and another social state");
    test.check(shorter.populationSignature != run.populationSignature, "half the ticks is another world");
}
