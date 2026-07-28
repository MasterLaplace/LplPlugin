/**
 * @file test_living_parity.cpp
 * @brief The Linux oracle for the living simulation gate.
 *
 * Runs `ecology::parityLivingRecipe()` and prints the four signatures the kernel
 * must reproduce bit for bit. The kernel side is
 * `libengine/src/smoke/p8_living_smoke.cpp`, which runs the SAME recipe from the
 * same constexpr definition — the recipe lives in one place precisely so the two
 * callers cannot drift apart by editing their own copy.
 *
 * The checks below are the ones a fold cannot make on its own. A signature proves
 * two runs agree; it says nothing about whether the run was worth agreeing on. So
 * before printing anything, this asserts that the simulation actually moved: that
 * the populations changed, that the genomes are no longer the founder's, that the
 * field holds a trail, and that a second run with a different seed lands
 * somewhere else. A gate over a simulation that stood still is a gate over its
 * initial conditions.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecology/LivingRecipe.hpp>

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool condition, const char *label)
{
    if (!condition)
        ++g_failures;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", label);
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== living simulation parity (Linux oracle) ==\n\n");

    const ecology::LivingRecipe recipe = ecology::parityLivingRecipe();
    const ecology::LivingResult run = ecology::runLiving(recipe);

    std::printf("-- the run is well formed --\n");
    check(run.ok == 1u, "the recipe produced a usable run");
    check(run.trailCells != 0u, "the field holds a trail above its evaporation floor");
    check(run.realisedRooms != 0u, "the budget realised at least one room");
    check(run.realisedRooms <= recipe.budget.maxRealisedRooms, "and never more than the budget allows");
    check(run.migrations != 0u, "abstract creatures migrated between rooms");

    std::printf("\n-- the simulation actually moved --\n");
    // The state after zero ticks is the initial condition. If the folds match it,
    // every subsystem is a very expensive constant.
    ecology::LivingRecipe still = recipe;
    still.ticks = 0u;
    const ecology::LivingResult initial = ecology::runLiving(still);
    check(run.populationSignature != initial.populationSignature, "the populations evolved");
    check(run.genomeSignature != initial.genomeSignature, "the genomes drifted from the founders");
    check(run.stigmergySignature != initial.stigmergySignature, "the pheromone field changed");
    check(run.socialSignature != initial.socialSignature, "the social layer reorganised");

    std::printf("\n-- and it is reproducible --\n");
    const ecology::LivingResult twin = ecology::runLiving(recipe);
    check(twin.populationSignature == run.populationSignature, "a second run folds the same populations");
    check(twin.genomeSignature == run.genomeSignature, "the same genomes");
    check(twin.stigmergySignature == run.stigmergySignature, "the same field");
    check(twin.socialSignature == run.socialSignature, "the same social state");

    ecology::LivingRecipe other = recipe;
    other.seed = recipe.seed + 1u;
    const ecology::LivingResult elsewhere = ecology::runLiving(other);
    // Not a formality: three of the four subsystems seed their streams from the
    // master seed, and a subsystem that ignored it would fold identically here
    // while looking perfectly deterministic.
    check(elsewhere.stigmergySignature != run.stigmergySignature, "a different seed gives a different field");
    check(elsewhere.socialSignature != run.socialSignature, "and a different social state");

    std::printf("\n-- the tick count is part of the state --\n");
    ecology::LivingRecipe shorter = recipe;
    shorter.ticks = recipe.ticks / 2u;
    const ecology::LivingResult halfway = ecology::runLiving(shorter);
    check(halfway.populationSignature != run.populationSignature, "half the ticks is a different world");

    std::printf("\n-- signatures the kernel must reproduce --\n");
    std::printf("  population_sig = 0x%08X\n", run.populationSignature);
    std::printf("  genome_sig     = 0x%08X\n", run.genomeSignature);
    std::printf("  stigmergy_sig  = 0x%08X\n", run.stigmergySignature);
    std::printf("  social_sig     = 0x%08X\n", run.socialSignature);
    std::printf("  extinctions    = %u\n", run.extinctions);
    std::printf("  anomalies      = %u\n", run.anomalies);
    std::printf("  realised_rooms = %u\n", run.realisedRooms);
    std::printf("  migrations     = %u\n", run.migrations);
    std::printf("  alpha_changes  = %u\n", run.alphaChanges);
    std::printf("  trail_cells    = %u\n", run.trailCells);
    std::printf("  living_ok      = %u\n", run.ok);

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
