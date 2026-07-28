/**
 * @file test_ecology.cpp
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
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecology/Genome.hpp>
#include <lpl/ecology/Populations.hpp>
#include <lpl/ecology/Society.hpp>

#include <cstdio>

namespace {

using namespace lpl;

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

/// A three-level web: grass, herbivores, carnivores.
ecology::TrophicWeb makeWeb()
{
    ecology::TrophicWeb web;

    ecology::SpeciesParams grass;
    grass.level = ecology::TrophicLevel::Producer;
    grass.growth = math::Fixed32::fromFloat(0.20f);
    grass.capacity = math::Fixed32::fromInt(2000);
    grass.refuge = math::Fixed32::fromInt(50);
    const core::u32 g = web.add(grass, math::Fixed32::fromInt(1200), ecology::Species::kNoPrey);

    ecology::SpeciesParams deer;
    deer.level = ecology::TrophicLevel::Primary;
    deer.predation = math::Fixed32::fromFloat(0.00020f);
    deer.conversion = math::Fixed32::fromFloat(0.40f);
    deer.mortality = math::Fixed32::fromFloat(0.05f);
    deer.capacity = math::Fixed32::fromInt(400);
    deer.refuge = math::Fixed32::fromInt(8);
    const core::u32 d = web.add(deer, math::Fixed32::fromInt(120), g);

    ecology::SpeciesParams wolf;
    wolf.level = ecology::TrophicLevel::Secondary;
    wolf.predation = math::Fixed32::fromFloat(0.0016f);
    wolf.conversion = math::Fixed32::fromFloat(0.30f);
    wolf.mortality = math::Fixed32::fromFloat(0.06f);
    wolf.capacity = math::Fixed32::fromInt(80);
    wolf.refuge = math::Fixed32::fromInt(2);
    (void) web.add(wolf, math::Fixed32::fromInt(20), d);

    return web;
}

void testOscillationsStayBounded()
{
    std::printf("populations oscillate without spiralling out\n");

    ecology::TrophicWeb web = makeWeb();

    // Let it settle, then measure the amplitude in an early window and a late one.
    // A neutrally stable model integrated explicitly grows: the late window would
    // be visibly wider. That is the failure being looked for.
    web.step(500u);

    const auto amplitude = [&](core::u32 steps) {
        // Fixed32 tops out around 32767; seeding a minimum with 100000 wraps it
        // negative, and every comparison against it then reads as a new record.
        math::Fixed32 low = math::Fixed32::fromInt(30000);
        math::Fixed32 high{};
        for (core::u32 i = 0u; i < steps; ++i)
        {
            web.step(1u);
            const math::Fixed32 n = web.populationOf(1u);
            if (n < low)
                low = n;
            if (n > high)
                high = n;
        }
        return high - low;
    };

    const math::Fixed32 early = amplitude(2000u);
    web.step(20000u);
    const math::Fixed32 late = amplitude(2000u);

    std::printf("    herbivore swing: early %.1f, after 20000 more steps %.1f\n", early.toFloat(), late.toFloat());
    check(late < early * math::Fixed32::fromInt(2), "the oscillation does not grow without bound");

    // And nothing reached zero, at any point, ever.
    bool anyExtinct = false;
    for (core::u32 i = 0u; i < 50000u; ++i)
    {
        web.step(1u);
        for (core::u32 s = 0u; s < 3u; ++s)
            if (web.populationOf(s).raw() <= 0)
                anyExtinct = true;
    }
    check(!anyExtinct, "no species pseudo-extincts over 50000 steps");
}

void testTrophicCascade()
{
    std::printf("removing the apex predator cascades, unscripted\n");

    ecology::TrophicWeb web = makeWeb();
    web.step(3000u);

    const math::Fixed32 grassBefore = web.populationOf(0u);
    const math::Fixed32 deerBefore = web.populationOf(1u);
    const math::Fixed32 wolfBefore = web.populationOf(2u);

    // The trophy hunt.
    web.extirpate(2u);
    web.step(2000u);

    const math::Fixed32 grassAfter = web.populationOf(0u);
    const math::Fixed32 deerAfter = web.populationOf(1u);

    std::printf("    before: grass %.0f deer %.0f wolves %.0f\n", grassBefore.toFloat(), deerBefore.toFloat(),
                wolfBefore.toFloat());
    std::printf("    after removing the wolves: grass %.0f deer %.0f\n", grassAfter.toFloat(), deerAfter.toFloat());

    // The two halves of a cascade, in order, with no code naming either.
    check(deerAfter > deerBefore, "the herbivores are released once their predator is gone");
    check(grassAfter < grassBefore, "and the vegetation is grazed down as a consequence");
    check(web.populationOf(2u).raw() == 0, "the extirpated species does not return");
}

void testBossesEmergeFromCollapse()
{
    std::printf("anomalies emerge from collapse, sometimes and never on demand\n");

    ecology::HeredityParams params;
    ecology::Genome base;

    const auto run = [&](math::Fixed32 local, math::Fixed32 capacity) {
        core::u32 anomalies = 0u;
        for (core::u32 trial = 0u; trial < 400u; ++trial)
        {
            core::u32 stream = 0xA11CEu + trial * 7919u;

            // A small population, bred among itself.
            lpl::pmr::vector<ecology::Genome> pop;
            for (core::u32 i = 0u; i < 8u; ++i)
                pop.push_back(ecology::mutate(base, 4u, 0.10f, stream));

            for (core::u32 generation = 0u; generation < 12u; ++generation)
            {
                const ecology::PopulationStats stats = ecology::strengthStats(&pop[0], 8u);
                lpl::pmr::vector<ecology::Genome> next;
                for (core::u32 i = 0u; i < 8u; ++i)
                {
                    const ecology::Genome child =
                        ecology::breed(pop[i], pop[(i + 1u) % 8u], local, capacity, params, stream);
                    if (ecology::isAnomaly(child, stats, params))
                        ++anomalies;
                    next.push_back(child);
                }
                pop = next;
            }
        }
        return anomalies;
    };

    const core::u32 healthy = run(math::Fixed32::fromInt(900), math::Fixed32::fromInt(1000));
    const core::u32 collapsed = run(math::Fixed32::fromInt(30), math::Fixed32::fromInt(1000));

    std::printf("    anomalies over 400 trials: %u in a healthy population, %u in a collapsed one\n", healthy,
                collapsed);
    check(collapsed > healthy, "collapse produces anomalies a healthy population does not");
    check(collapsed > 0u, "the mechanism actually fires");

    // Drift, not a spawn table: it must not fire on every collapse either, or it
    // is a scripted event wearing a genetics costume.
    check(collapsed < 400u * 12u * 8u / 4u, "an anomaly stays rare even under collapse");

    check(!ecology::inMutationalMeltdown(math::Fixed32::fromInt(900), math::Fixed32::fromInt(1000), params),
          "a full habitat is not in meltdown");
    check(ecology::inMutationalMeltdown(math::Fixed32::fromInt(30), math::Fixed32::fromInt(1000), params),
          "a collapsed one is");

    // The threshold is a share, so the same collapse reads the same at any scale.
    check(ecology::inMutationalMeltdown(math::Fixed32::fromInt(3), math::Fixed32::fromInt(100), params) ==
              ecology::inMutationalMeltdown(math::Fixed32::fromInt(300), math::Fixed32::fromInt(10000), params),
          "the collapse threshold is relative, so habitat size does not change its meaning");
}

void testSelectionMovesASpecies()
{
    std::printf("culling the slow makes the species fast\n");

    core::u32 stream = 0x5EED0u;
    lpl::pmr::vector<ecology::Genome> population;
    for (core::u32 i = 0u; i < 64u; ++i)
        population.push_back(ecology::mutate(ecology::Genome{}, 8u, 0.25f, stream));

    const auto meanSpeed = [](const lpl::pmr::vector<ecology::Genome> &pop) {
        math::Fixed32 total{};
        for (core::u32 i = 0u; i < pop.size(); ++i)
            total = total + pop[i].maxSpeed;
        return total / math::Fixed32::fromInt(static_cast<core::i32>(pop.size()));
    };

    const math::Fixed32 initial = meanSpeed(population);

    ecology::HeredityParams params;
    for (core::u32 generation = 0u; generation < 40u; ++generation)
    {
        // The guild's tactic: kill the slowest half. Nothing about this says
        // "make them faster" — that is what selection does with it.
        for (core::u32 i = 0u; i < population.size(); ++i)
            for (core::u32 j = i + 1u; j < population.size(); ++j)
                if (population[j].maxSpeed > population[i].maxSpeed)
                {
                    const ecology::Genome swap = population[i];
                    population[i] = population[j];
                    population[j] = swap;
                }

        lpl::pmr::vector<ecology::Genome> survivors;
        for (core::u32 i = 0u; i < 32u; ++i)
            survivors.push_back(population[i]);

        lpl::pmr::vector<ecology::Genome> next;
        for (core::u32 i = 0u; i < 64u; ++i)
            next.push_back(ecology::breed(survivors[i % 32u], survivors[(i * 7u + 3u) % 32u],
                                          math::Fixed32::fromInt(900), math::Fixed32::fromInt(1000), params, stream));
        population = next;
    }

    const math::Fixed32 evolved = meanSpeed(population);
    std::printf("    mean speed %.3f -> %.3f over 40 generations of culling the slow\n", initial.toFloat(),
                evolved.toFloat());
    check(evolved > initial, "the population evolved in the direction the player selected for");
}

void testIslandRule()
{
    std::printf("isolation drives size in opposite directions\n");

    ecology::IslandParams params;

    ecology::Genome mouse;
    mouse.size = math::Fixed32::fromFloat(0.5f);
    ecology::Genome elephant;
    elephant.size = math::Fixed32::fromFloat(3.0f);

    const math::Fixed32 mouseAncestor = mouse.size;
    const math::Fixed32 elephantAncestor = elephant.size;
    for (core::u32 generation = 0u; generation < 200u; ++generation)
    {
        mouse = ecology::applyIslandRule(mouse, mouseAncestor, true, params);
        elephant = ecology::applyIslandRule(elephant, elephantAncestor, true, params);
    }

    std::printf("    isolated: small species 0.500 -> %.3f, large species 3.000 -> %.3f\n", mouse.size.toFloat(),
                elephant.size.toFloat());
    check(mouse.size > math::Fixed32::one(), "the small species grew toward gigantism");
    check(elephant.size < math::Fixed32::fromInt(2), "the large species shrank toward dwarfism");

    // Not isolated: nothing happens. The rule is about isolation, not time.
    ecology::Genome mainland;
    mainland.size = math::Fixed32::fromFloat(0.5f);
    for (core::u32 generation = 0u; generation < 200u; ++generation)
        mainland = ecology::applyIslandRule(mainland, math::Fixed32::fromFloat(0.5f), false, params);
    check(mainland.size.raw() == math::Fixed32::fromFloat(0.5f).raw(), "a connected population does not drift in size");

    // Size drags strength and speed with it, or it would be a cosmetic gene.
    check(mouse.strength > ecology::Genome{}.strength, "a giant hits harder");
    check(mouse.maxSpeed < ecology::Genome{}.maxSpeed, "and moves less nimbly");
}

void testPackLifeCycle()
{
    std::printf("killing an alpha has social consequences\n");

    lpl::pmr::vector<ecology::PackMember> members;
    for (core::u32 i = 0u; i < 12u; ++i)
    {
        ecology::PackMember member;
        member.id = 100u + i;
        member.lineage = 7u;
        member.fitness = math::Fixed32::fromInt(static_cast<core::i32>(i));
        members.push_back(member);
    }

    ecology::PackParams params;
    params.dissolutionChance16 = 16u; // certain, so the consequence is observable
    core::u32 stream = 0xB055u;

    ecology::PackEvents events = ecology::stepPacks(&members[0], 12u, params, stream);
    (void) ecology::stepPacks(&members[0], 12u, params, stream);
    std::printf("    formation: %u formed, %u adopted, %u budded\n", events.formed, events.adopted, events.budded);

    core::u32 packed = 0u;
    core::u32 alphas = 0u;
    core::u32 alphaId = 0u;
    for (core::u32 i = 0u; i < 12u; ++i)
    {
        packed += members[i].pack != ecology::kSolitary ? 1u : 0u;
        if (members[i].alpha)
        {
            ++alphas;
            alphaId = members[i].id;
        }
    }
    check(packed > 1u, "kin band together");
    check(alphas >= 1u, "a pack has a leader");

    // The alpha is the fittest, with the lower id winning a tie. Without that
    // tie-break the leader would be whichever member the loop reached first.
    check(alphaId == 111u, "the fittest member leads");

    const ecology::PackEvents death = ecology::killMember(&members[0], 12u, alphaId, params, stream);
    std::printf("    killing the alpha: %u dissolved, %u scattered\n", death.dissolved, death.scattered);
    check(death.dissolved > 0u, "the pack shattered");
    check(death.scattered > 0u, "its members are now solitary");
}

void testOverflowAndInvasion()
{
    std::printf("hunger overrides fear, and an invasion is genetically uniform\n");

    ecology::OverflowParams params;

    math::Fixed32 wellFed[8];
    math::Fixed32 starving[8];
    for (core::u32 i = 0u; i < 8u; ++i)
    {
        wellFed[i] = math::Fixed32::fromFloat(0.8f);
        starving[i] = math::Fixed32::fromFloat(0.1f);
    }

    const ecology::OverflowState calm =
        ecology::evaluateOverflow(math::Fixed32::fromInt(50), math::Fixed32::fromInt(100), wellFed, 8u, params);
    check(!calm.overcrowded && !calm.raiding, "a healthy region does not raid");

    const ecology::OverflowState crowdedButFed =
        ecology::evaluateOverflow(math::Fixed32::fromInt(98), math::Fixed32::fromInt(100), wellFed, 8u, params);
    check(crowdedButFed.overcrowded && !crowdedButFed.raiding, "crowding alone is not enough — they have to be hungry");

    const ecology::OverflowState spill =
        ecology::evaluateOverflow(math::Fixed32::fromInt(98), math::Fixed32::fromInt(100), starving, 8u, params);
    std::printf("    pressure %.2f, %u starving, raiding: %s\n", spill.pressure.toFloat(), spill.starving,
                spill.raiding ? "yes" : "no");
    check(spill.raiding, "an overcrowded, starving region spills — with nothing deciding to attack");

    // Founder effect: an invasion is numerous and nearly identical.
    core::u32 stream = 0x11AAu;
    lpl::pmr::vector<ecology::Genome> natives;
    for (core::u32 i = 0u; i < 64u; ++i)
        natives.push_back(ecology::mutate(ecology::Genome{}, 12u, 0.40f, stream));

    lpl::pmr::vector<ecology::Genome> invaders;
    ecology::seedInvasion(ecology::Genome{}, 64u, 0.02f, stream, invaders);

    const math::Fixed32 nativeSpread = ecology::geneticDiversity(&natives[0], 64u);
    const math::Fixed32 invaderSpread = ecology::geneticDiversity(&invaders[0], 64u);
    std::printf("    genetic diversity: natives %.4f, invaders %.4f\n", nativeSpread.toFloat(),
                invaderSpread.toFloat());
    check(invaderSpread < nativeSpread, "the invader carries the founder effect's uniformity");
}

} // namespace

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
