#include <lpl/ecology/Genome.hpp>
#include <lpl/ecology/Populations.hpp>
#include <lpl/ecology/Society.hpp>

#include <cstdio>

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    if (!condition)
    {
        ++gFailures;
        std::printf("  FAIL %s\n", what);
    }
}

/**
 * @brief A three-level web: grass, herbivores, carnivores.
 */
lpl::ecology::TrophicWeb makeWeb()
{
    lpl::ecology::TrophicWeb web;

    lpl::ecology::SpeciesParams grass;
    grass.level = lpl::ecology::TrophicLevel::Producer;
    grass.growth = lpl::math::Fixed32::fromFloat(0.20f);
    grass.capacity = lpl::math::Fixed32::fromInt(2000);
    grass.refuge = lpl::math::Fixed32::fromInt(50);
    const lpl::core::u32 g = web.add(grass, lpl::math::Fixed32::fromInt(1200), lpl::ecology::Species::kNoPrey);

    lpl::ecology::SpeciesParams deer;
    deer.level = lpl::ecology::TrophicLevel::Primary;
    deer.predation = lpl::math::Fixed32::fromFloat(0.00020f);
    deer.conversion = lpl::math::Fixed32::fromFloat(0.40f);
    deer.mortality = lpl::math::Fixed32::fromFloat(0.05f);
    deer.capacity = lpl::math::Fixed32::fromInt(400);
    deer.refuge = lpl::math::Fixed32::fromInt(8);
    const lpl::core::u32 d = web.add(deer, lpl::math::Fixed32::fromInt(120), g);

    lpl::ecology::SpeciesParams wolf;
    wolf.level = lpl::ecology::TrophicLevel::Secondary;
    wolf.predation = lpl::math::Fixed32::fromFloat(0.0016f);
    wolf.conversion = lpl::math::Fixed32::fromFloat(0.30f);
    wolf.mortality = lpl::math::Fixed32::fromFloat(0.06f);
    wolf.capacity = lpl::math::Fixed32::fromInt(80);
    wolf.refuge = lpl::math::Fixed32::fromInt(2);
    (void) web.add(wolf, lpl::math::Fixed32::fromInt(20), d);

    return web;
}

/**
 * @brief Populations oscillate without spiralling out.
 *
 * @details The web settles, then the amplitude is measured in an early window and a late one. A neutrally stable
 *          model integrated explicitly grows: the late window would be visibly wider. That is the failure being
 *          looked for.
 *
 * The minimum of a window is seeded with 30000: Fixed32 tops out around 32767; seeding a minimum with 100000 wraps
 * it negative, and every comparison against it then reads as a new record.
 */
void testOscillationsStayBounded()
{
    std::printf("populations oscillate without spiralling out\n");

    lpl::ecology::TrophicWeb web = makeWeb();

    web.step(500u);

    const auto amplitude = [&](lpl::core::u32 steps) {
        lpl::math::Fixed32 low = lpl::math::Fixed32::fromInt(30000);
        lpl::math::Fixed32 high{};
        for (lpl::core::u32 i = 0u; i < steps; ++i)
        {
            web.step(1u);
            const lpl::math::Fixed32 n = web.populationOf(1u);
            if (n < low)
                low = n;
            if (n > high)
                high = n;
        }
        return high - low;
    };

    const lpl::math::Fixed32 early = amplitude(2000u);
    web.step(20000u);
    const lpl::math::Fixed32 late = amplitude(2000u);

    std::printf("    herbivore swing: early %.1f, after 20000 more steps %.1f\n", early.toFloat(), late.toFloat());
    check(late < early * lpl::math::Fixed32::fromInt(2), "the oscillation does not grow without bound");

    bool anyExtinct = false;
    for (lpl::core::u32 i = 0u; i < 50000u; ++i)
    {
        web.step(1u);
        for (lpl::core::u32 s = 0u; s < 3u; ++s)
            if (web.populationOf(s).raw() <= 0)
                anyExtinct = true;
    }
    check(!anyExtinct, "no species pseudo-extincts over 50000 steps");
}

/**
 * @brief Removing the apex predator cascades, unscripted.
 *
 * @details The wolves are removed as a trophy hunt would. The checks are the two halves of a cascade, in order,
 *          with no code naming either.
 */
void testTrophicCascade()
{
    std::printf("removing the apex predator cascades, unscripted\n");

    lpl::ecology::TrophicWeb web = makeWeb();
    web.step(3000u);

    const lpl::math::Fixed32 grassBefore = web.populationOf(0u);
    const lpl::math::Fixed32 deerBefore = web.populationOf(1u);
    const lpl::math::Fixed32 wolfBefore = web.populationOf(2u);

    web.extirpate(2u);
    web.step(2000u);

    const lpl::math::Fixed32 grassAfter = web.populationOf(0u);
    const lpl::math::Fixed32 deerAfter = web.populationOf(1u);

    std::printf("    before: grass %.0f deer %.0f wolves %.0f\n", grassBefore.toFloat(), deerBefore.toFloat(),
                wolfBefore.toFloat());
    std::printf("    after removing the wolves: grass %.0f deer %.0f\n", grassAfter.toFloat(), deerAfter.toFloat());

    check(deerAfter > deerBefore, "the herbivores are released once their predator is gone");
    check(grassAfter < grassBefore, "and the vegetation is grazed down as a consequence");
    check(web.populationOf(2u).raw() == 0, "the extirpated species does not return");
}

/**
 * @brief Anomalies emerge from collapse, sometimes and never on demand.
 *
 * @details Each trial breeds a small population among itself. Drift, not a spawn table: it must not fire on every
 *          collapse either, or it is a scripted event wearing a genetics costume.
 *
 * The threshold is a share, so the same collapse reads the same at any scale.
 */
void testBossesEmergeFromCollapse()
{
    std::printf("anomalies emerge from collapse, sometimes and never on demand\n");

    lpl::ecology::HeredityParams params;
    lpl::ecology::Genome base;

    const auto run = [&](lpl::math::Fixed32 local, lpl::math::Fixed32 capacity) {
        lpl::core::u32 anomalies = 0u;
        for (lpl::core::u32 trial = 0u; trial < 400u; ++trial)
        {
            lpl::core::u32 stream = 0xA11CEu + trial * 7919u;

            lpl::pmr::vector<lpl::ecology::Genome> pop;
            for (lpl::core::u32 i = 0u; i < 8u; ++i)
                pop.push_back(lpl::ecology::mutate(base, 4u, 0.10f, stream));

            for (lpl::core::u32 generation = 0u; generation < 12u; ++generation)
            {
                const lpl::ecology::PopulationStats stats = lpl::ecology::strengthStats(&pop[0], 8u);
                lpl::pmr::vector<lpl::ecology::Genome> next;
                for (lpl::core::u32 i = 0u; i < 8u; ++i)
                {
                    const lpl::ecology::Genome child =
                        lpl::ecology::breed(pop[i], pop[(i + 1u) % 8u], local, capacity, params, stream);
                    if (lpl::ecology::isAnomaly(child, stats, params))
                        ++anomalies;
                    next.push_back(child);
                }
                pop = next;
            }
        }
        return anomalies;
    };

    const lpl::core::u32 healthy = run(lpl::math::Fixed32::fromInt(900), lpl::math::Fixed32::fromInt(1000));
    const lpl::core::u32 collapsed = run(lpl::math::Fixed32::fromInt(30), lpl::math::Fixed32::fromInt(1000));

    std::printf("    anomalies over 400 trials: %u in a healthy population, %u in a collapsed one\n", healthy,
                collapsed);
    check(collapsed > healthy, "collapse produces anomalies a healthy population does not");
    check(collapsed > 0u, "the mechanism actually fires");

    check(collapsed < 400u * 12u * 8u / 4u, "an anomaly stays rare even under collapse");

    check(!lpl::ecology::inMutationalMeltdown(lpl::math::Fixed32::fromInt(900), lpl::math::Fixed32::fromInt(1000),
                                              params),
          "a full habitat is not in meltdown");
    check(
        lpl::ecology::inMutationalMeltdown(lpl::math::Fixed32::fromInt(30), lpl::math::Fixed32::fromInt(1000), params),
        "a collapsed one is");

    check(
        lpl::ecology::inMutationalMeltdown(lpl::math::Fixed32::fromInt(3), lpl::math::Fixed32::fromInt(100), params) ==
            lpl::ecology::inMutationalMeltdown(lpl::math::Fixed32::fromInt(300), lpl::math::Fixed32::fromInt(10000),
                                               params),
        "the collapse threshold is relative, so habitat size does not change its meaning");
}

/**
 * @brief Culling the slow makes the species fast.
 *
 * @details The guild's tactic: kill the slowest half. Nothing about this says "make them faster" — that is what
 *          selection does with it.
 */
void testSelectionMovesASpecies()
{
    std::printf("culling the slow makes the species fast\n");

    lpl::core::u32 stream = 0x5EED0u;
    lpl::pmr::vector<lpl::ecology::Genome> population;
    for (lpl::core::u32 i = 0u; i < 64u; ++i)
        population.push_back(lpl::ecology::mutate(lpl::ecology::Genome{}, 8u, 0.25f, stream));

    const auto meanSpeed = [](const lpl::pmr::vector<lpl::ecology::Genome> &pop) {
        lpl::math::Fixed32 total{};
        for (lpl::core::u32 i = 0u; i < pop.size(); ++i)
            total = total + pop[i].maxSpeed;
        return total / lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(pop.size()));
    };

    const lpl::math::Fixed32 initial = meanSpeed(population);

    lpl::ecology::HeredityParams params;
    for (lpl::core::u32 generation = 0u; generation < 40u; ++generation)
    {
        for (lpl::core::u32 i = 0u; i < population.size(); ++i)
            for (lpl::core::u32 j = i + 1u; j < population.size(); ++j)
                if (population[j].maxSpeed > population[i].maxSpeed)
                {
                    const lpl::ecology::Genome swap = population[i];
                    population[i] = population[j];
                    population[j] = swap;
                }

        lpl::pmr::vector<lpl::ecology::Genome> survivors;
        for (lpl::core::u32 i = 0u; i < 32u; ++i)
            survivors.push_back(population[i]);

        lpl::pmr::vector<lpl::ecology::Genome> next;
        for (lpl::core::u32 i = 0u; i < 64u; ++i)
            next.push_back(lpl::ecology::breed(survivors[i % 32u], survivors[(i * 7u + 3u) % 32u],
                                               lpl::math::Fixed32::fromInt(900), lpl::math::Fixed32::fromInt(1000),
                                               params, stream));
        population = next;
    }

    const lpl::math::Fixed32 evolved = meanSpeed(population);
    std::printf("    mean speed %.3f -> %.3f over 40 generations of culling the slow\n", initial.toFloat(),
                evolved.toFloat());
    check(evolved > initial, "the population evolved in the direction the player selected for");
}

/**
 * @brief Isolation drives size in opposite directions.
 *
 * @details Not isolated, nothing happens: the rule is about isolation, not time. Size drags strength and speed with
 *          it, or it would be a cosmetic gene.
 */
void testIslandRule()
{
    std::printf("isolation drives size in opposite directions\n");

    lpl::ecology::IslandParams params;

    lpl::ecology::Genome mouse;
    mouse.size = lpl::math::Fixed32::fromFloat(0.5f);
    lpl::ecology::Genome elephant;
    elephant.size = lpl::math::Fixed32::fromFloat(3.0f);

    const lpl::math::Fixed32 mouseAncestor = mouse.size;
    const lpl::math::Fixed32 elephantAncestor = elephant.size;
    for (lpl::core::u32 generation = 0u; generation < 200u; ++generation)
    {
        mouse = lpl::ecology::applyIslandRule(mouse, mouseAncestor, true, params);
        elephant = lpl::ecology::applyIslandRule(elephant, elephantAncestor, true, params);
    }

    std::printf("    isolated: small species 0.500 -> %.3f, large species 3.000 -> %.3f\n", mouse.size.toFloat(),
                elephant.size.toFloat());
    check(mouse.size > lpl::math::Fixed32::one(), "the small species grew toward gigantism");
    check(elephant.size < lpl::math::Fixed32::fromInt(2), "the large species shrank toward dwarfism");

    lpl::ecology::Genome mainland;
    mainland.size = lpl::math::Fixed32::fromFloat(0.5f);
    for (lpl::core::u32 generation = 0u; generation < 200u; ++generation)
        mainland = lpl::ecology::applyIslandRule(mainland, lpl::math::Fixed32::fromFloat(0.5f), false, params);
    check(mainland.size.raw() == lpl::math::Fixed32::fromFloat(0.5f).raw(),
          "a connected population does not drift in size");

    check(mouse.strength > lpl::ecology::Genome{}.strength, "a giant hits harder");
    check(mouse.maxSpeed < lpl::ecology::Genome{}.maxSpeed, "and moves less nimbly");
}

/**
 * @brief Killing an alpha has social consequences.
 *
 * @details Dissolution is made certain, so the consequence is observable. The alpha is the fittest, with the lower
 *          id winning a tie. Without that tie-break the leader would be whichever member the loop reached first.
 */
void testPackLifeCycle()
{
    std::printf("killing an alpha has social consequences\n");

    lpl::pmr::vector<lpl::ecology::PackMember> members;
    for (lpl::core::u32 i = 0u; i < 12u; ++i)
    {
        lpl::ecology::PackMember member;
        member.id = 100u + i;
        member.lineage = 7u;
        member.fitness = lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(i));
        members.push_back(member);
    }

    lpl::ecology::PackParams params;
    params.dissolutionChance16 = 16u;
    lpl::core::u32 stream = 0xB055u;

    lpl::ecology::PackEvents events = lpl::ecology::stepPacks(&members[0], 12u, params, stream);
    (void) lpl::ecology::stepPacks(&members[0], 12u, params, stream);
    std::printf("    formation: %u formed, %u adopted, %u budded\n", events.formed, events.adopted, events.budded);

    lpl::core::u32 packed = 0u;
    lpl::core::u32 alphas = 0u;
    lpl::core::u32 alphaId = 0u;
    for (lpl::core::u32 i = 0u; i < 12u; ++i)
    {
        packed += members[i].pack != lpl::ecology::kSolitary ? 1u : 0u;
        if (members[i].alpha)
        {
            ++alphas;
            alphaId = members[i].id;
        }
    }
    check(packed > 1u, "kin band together");
    check(alphas >= 1u, "a pack has a leader");

    check(alphaId == 111u, "the fittest member leads");

    const lpl::ecology::PackEvents death = lpl::ecology::killMember(&members[0], 12u, alphaId, params, stream);
    std::printf("    killing the alpha: %u dissolved, %u scattered\n", death.dissolved, death.scattered);
    check(death.dissolved > 0u, "the pack shattered");
    check(death.scattered > 0u, "its members are now solitary");
}

/**
 * @brief Hunger overrides fear, and an invasion is genetically uniform.
 *
 * @details Founder effect: an invasion is numerous and nearly identical.
 */
void testOverflowAndInvasion()
{
    std::printf("hunger overrides fear, and an invasion is genetically uniform\n");

    lpl::ecology::OverflowParams params;

    lpl::math::Fixed32 wellFed[8];
    lpl::math::Fixed32 starving[8];
    for (lpl::core::u32 i = 0u; i < 8u; ++i)
    {
        wellFed[i] = lpl::math::Fixed32::fromFloat(0.8f);
        starving[i] = lpl::math::Fixed32::fromFloat(0.1f);
    }

    const lpl::ecology::OverflowState calm = lpl::ecology::evaluateOverflow(
        lpl::math::Fixed32::fromInt(50), lpl::math::Fixed32::fromInt(100), wellFed, 8u, params);
    check(!calm.overcrowded && !calm.raiding, "a healthy region does not raid");

    const lpl::ecology::OverflowState crowdedButFed = lpl::ecology::evaluateOverflow(
        lpl::math::Fixed32::fromInt(98), lpl::math::Fixed32::fromInt(100), wellFed, 8u, params);
    check(crowdedButFed.overcrowded && !crowdedButFed.raiding, "crowding alone is not enough — they have to be hungry");

    const lpl::ecology::OverflowState spill = lpl::ecology::evaluateOverflow(
        lpl::math::Fixed32::fromInt(98), lpl::math::Fixed32::fromInt(100), starving, 8u, params);
    std::printf("    pressure %.2f, %u starving, raiding: %s\n", spill.pressure.toFloat(), spill.starving,
                spill.raiding ? "yes" : "no");
    check(spill.raiding, "an overcrowded, starving region spills — with nothing deciding to attack");

    lpl::core::u32 stream = 0x11AAu;
    lpl::pmr::vector<lpl::ecology::Genome> natives;
    for (lpl::core::u32 i = 0u; i < 64u; ++i)
        natives.push_back(lpl::ecology::mutate(lpl::ecology::Genome{}, 12u, 0.40f, stream));

    lpl::pmr::vector<lpl::ecology::Genome> invaders;
    lpl::ecology::seedInvasion(lpl::ecology::Genome{}, 64u, 0.02f, stream, invaders);

    const lpl::math::Fixed32 nativeSpread = lpl::ecology::geneticDiversity(&natives[0], 64u);
    const lpl::math::Fixed32 invaderSpread = lpl::ecology::geneticDiversity(&invaders[0], 64u);
    std::printf("    genetic diversity: natives %.4f, invaders %.4f\n", nativeSpread.toFloat(),
                invaderSpread.toFloat());
    check(invaderSpread < nativeSpread, "the invader carries the founder effect's uniformity");
}

} // namespace

/**
 * @brief Probes for the population layer.
 *
 * The claims here are the ones the design rests on, and every one of them is a
 * place the naive implementation fails:
 *
 *  - Oscillations stay BOUNDED. Classical Lotka-Volterra integrated explicitly
 *    spirals outward until a trough passes below one individual and the species
 *    is gone forever. Measuring the amplitude over a long run is the only way to
 *    know which version was built.
 *  - Removing the apex predator produces a cascade — mesopredator release, then
 *    collapse of the herbivores — with no code that names any of it.
 *  - A collapsed population produces anomalies SOMETIMES, never on demand.
 *  - Isolation drives size in opposite directions depending on the species.
 *  - Killing an alpha can shatter its pack.
 */
int main()
{
    std::printf("== ecology ==\n");
    testOscillationsStayBounded();
    testTrophicCascade();
    testBossesEmergeFromCollapse();
    testSelectionMovesASpecies();
    testIslandRule();
    testPackLifeCycle();
    testOverflowAndInvasion();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
