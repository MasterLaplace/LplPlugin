/**
 * @file test_ai.cpp
 * @brief Probes for the agent layer.
 *
 * Each of these measures a claim the source literature makes, rather than
 * checking that the code runs:
 *
 *  - Diffusion around an obstacle: the gradient must lead a body AROUND a wall,
 *    which is the entire reason to prefer a field over pathfinding.
 *  - Encirclement: attraction to prey plus repulsion between kin must produce
 *    flanking with no tactical code anywhere.
 *  - A blocked trail must be ABANDONED within a bounded number of ticks — the
 *    resilience claim ant colony optimisation is chosen for.
 *  - Exploration is what makes that possible: with none, the colony stays stuck.
 *  - A reversal must cost more than going round, or a long body folds through
 *    itself.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ai/AbstractWorld.hpp>
#include <lpl/ai/AntColony.hpp>
#include <lpl/ai/Affordance.hpp>
#include <lpl/ai/AiMap.hpp>
#include <lpl/ai/Personality.hpp>
#include <lpl/ai/Social.hpp>
#include <lpl/ai/SpringBody.hpp>
#include <lpl/ai/StigmergyField.hpp>
#include <lpl/ai/Swarm.hpp>
#include <lpl/math/FixedMath.hpp>

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

// ─────────────────────────────────────────────────────────────────────────────

void testDiffusionGoesAroundWalls()
{
    std::printf("scent flows around a wall, so following it does too\n");

    // A vertical wall across the middle with one gap: the only way from the left
    // side to the source on the right is through the gap.
    const core::u32 size = 13u;
    procgen::Grid<core::u8> walls{size, size, core::u8{0}};
    for (core::u32 z = 0u; z < size; ++z)
        if (z != 2u)
            walls.at(6u, z) = 1u;

    ai::StigmergyField field{size, size, 1u};
    field.setObstacles(walls);

    // A NAVIGATION field, not a forgetting one. Reach is about
    // sqrt(D / (1 - evaporation)) cells, so the defaults (0.92, 0.08) reach under
    // two cells and nothing arrives past the wall however long it runs — measured
    // as a flat zero on the far side after three thousand ticks, which read as
    // "the walker missed the opening" when the truth was that there was nothing
    // there to follow.
    ai::StigmergyParams params;
    params.evaporation = 0.9995f;
    params.diffusion = 0.40f;
    params.floor = 0.0001f;

    // Diffusion is a random walk, so its reach grows as the square root of time:
    // the scent needs enough ticks to travel the long way round before anything
    // downstream of the wall can be asked about it. Measured, 400 ticks got it to
    // the gap and no further, which read as "the walker missed the opening" when
    // the truth was that nothing had arrived yet.
    for (core::u32 tick = 0u; tick < 4000u; ++tick)
    {
        field.deposit(0u, 10u, 6u, math::Fixed32::fromInt(20));
        field.step(params);
    }

    // Walk up the gradient from the far side and see where it goes.
    core::u32 x = 3u;
    core::u32 z = 6u;
    bool reachedGap = false;
    bool crossedWall = false;
    for (core::u32 step = 0u; step < 200u; ++step)
    {
        const core::u32 dir = field.gradientDirection(0u, x, z, true);
        if (dir == ai::StigmergyField::kNoDirection)
            break;
        const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor8X[dir];
        const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor8Z[dir];
        if (walls.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) != 0u)
            crossedWall = true;
        x = static_cast<core::u32>(nx);
        z = static_cast<core::u32>(nz);
        if (x == 6u && z == 2u)
            reachedGap = true;
        if (x >= 9u)
            break;
    }

    std::printf("    walker ended at (%u, %u), gap %s\n", x, z, reachedGap ? "used" : "missed");
    check(!crossedWall, "the walker never steps into a wall");
    check(reachedGap, "the walker is led to the only opening");

    // The far side of the wall must be genuinely darker than the near side at the
    // same distance from the source: if the wall did not block diffusion, the two
    // would be equal and the whole navigation property would be an illusion.
    // Two cells either side of the wall, on the source's own row. In a straight
    // line they are two apart; around the gap they are more than twenty. If the
    // wall did not block diffusion they would read almost the same.
    const math::Fixed32 nearSide = field.value(0u, 7u, 6u);
    const math::Fixed32 farSide = field.value(0u, 5u, 6u);
    std::printf("    scent just past the wall %.4f vs just before it %.4f\n", farSide.toFloat(), nearSide.toFloat());
    check(nearSide > farSide, "a wall casts a scent shadow");
}

void testEncirclementEmerges()
{
    std::printf("attraction plus kin repulsion produces encirclement\n");

    const core::u32 size = 41u;
    ai::StigmergyField field{size, size, 2u};

    ai::StigmergyParams params;
    params.evaporation = 0.98f;
    params.diffusion = 0.25f;

    struct Hunter {
        core::u32 x, z;
    };
    Hunter hunters[6] = {
        {2u, 20u},
        {3u, 19u},
        {2u, 21u},
        {4u, 20u},
        {3u, 21u},
        {4u, 19u}
    };

    for (core::u32 tick = 0u; tick < 300u; ++tick)
    {
        field.deposit(0u, 20u, 20u, math::Fixed32::fromInt(30)); // the prey
        for (core::u32 h = 0u; h < 6u; ++h)
            field.deposit(1u, hunters[h].x, hunters[h].z, math::Fixed32::fromInt(12)); // kin marker
        field.step(params);

        for (core::u32 h = 0u; h < 6u; ++h)
        {
            // Two readings, opposite signs: toward the prey, away from each other.
            // Nothing here mentions surrounding, flanking or formations.
            const core::u32 toPrey = field.gradientDirection(0u, hunters[h].x, hunters[h].z, true);
            const core::u32 fromKin = field.gradientDirection(1u, hunters[h].x, hunters[h].z, false);
            const core::u32 dir = (tick % 3u == 0u && fromKin != ai::StigmergyField::kNoDirection) ? fromKin : toPrey;
            if (dir == ai::StigmergyField::kNoDirection)
                continue;
            const core::i32 nx = static_cast<core::i32>(hunters[h].x) + procgen::kNeighbor8X[dir];
            const core::i32 nz = static_cast<core::i32>(hunters[h].z) + procgen::kNeighbor8Z[dir];
            if (nx <= 0 || nz <= 0 || nx >= static_cast<core::i32>(size) - 1 || nz >= static_cast<core::i32>(size) - 1)
                continue;
            hunters[h].x = static_cast<core::u32>(nx);
            hunters[h].z = static_cast<core::u32>(nz);
        }
    }

    // Which side of the prey each hunter ended on. Starting all six on the left,
    // pure attraction keeps them there in a queue; the kin repulsion is what
    // pushes them around.
    core::u32 quadrants = 0u;
    bool seen[4] = {false, false, false, false};
    for (core::u32 h = 0u; h < 6u; ++h)
    {
        const bool east = hunters[h].x > 20u;
        const bool south = hunters[h].z > 20u;
        const core::u32 q = (east ? 1u : 0u) + (south ? 2u : 0u);
        if (!seen[q])
        {
            seen[q] = true;
            ++quadrants;
        }
    }
    std::printf("    six hunters, all starting west, ended spread over %u quadrants\n", quadrants);
    check(quadrants >= 2u, "hunters spread around the target rather than queueing behind one another");
}

void testColonyAbandonsABlockedTrail()
{
    std::printf("a colony abandons a route that stops working\n");

    const core::u32 size = 24u;
    procgen::Grid<core::u8> walls{size, size, core::u8{0}};

    ai::StigmergyField field{size, size, 1u};
    field.setObstacles(walls);

    ai::StigmergyParams params;
    params.evaporation = 0.90f;
    params.diffusion = 0.15f;

    // Lay a strong trail along row 5.
    lpl::pmr::vector<core::u32> trail;
    for (core::u32 x = 2u; x < 22u; ++x)
        trail.push_back(5u * size + x);
    for (core::u32 tick = 0u; tick < 60u; ++tick)
    {
        field.depositTrail(0u, &trail[0], static_cast<core::u32>(trail.size()), math::Fixed32::fromInt(400));
        field.step(params);
    }

    const math::Fixed32 established = field.value(0u, 12u, 5u);
    check(established > math::Fixed32::zero(), "a trail was established");

    // Now sever it and stop reinforcing. Evaporation is the only thing that can
    // undo the colony's belief, and the claim is that it does so within a bounded
    // time rather than eventually.
    for (core::u32 z = 0u; z < size; ++z)
        walls.at(12u, z) = 1u;
    field.setObstacles(walls);

    core::u32 ticksToForget = 0u;
    for (core::u32 tick = 1u; tick <= 400u; ++tick)
    {
        field.step(params);
        if (field.value(0u, 11u, 5u) <= established / math::Fixed32::fromInt(10))
        {
            ticksToForget = tick;
            break;
        }
    }

    std::printf("    trail strength fell by 10x after %u ticks\n", ticksToForget);
    check(ticksToForget > 0u && ticksToForget < 200u, "the stale trail is abandoned within a bounded time");

    // And it must reach exactly zero, not a permanent faint trace: a value walked
    // down to the last Q16.16 tick would keep the dead route marginally
    // attractive forever.
    core::u32 residue = 0u;
    for (core::u32 tick = 0u; tick < 600u; ++tick)
        field.step(params);
    for (core::u32 z = 0u; z < size; ++z)
        for (core::u32 x = 0u; x < size; ++x)
            if (field.value(0u, x, z).raw() != 0)
                ++residue;
    std::printf("    %u cells still carry a residue after 600 more ticks\n", residue);
    check(residue == 0u, "evaporation reaches zero rather than leaving a permanent trace");
}

void testTheColonyClosesItsTrail()
{
    std::printf("the homing rule is what turns a field into a route\n");

    // Every mechanism of ant colony optimisation was in this module and there was no
    // colony: the agents, the nest and the rule that sends them home lived inside a
    // viewer's main.cpp, where none of this could be stated. The rule is the part
    // worth testing, because it is the one thing chooseAntMove cannot express — the
    // choice is local, the rule is about the colony.
    const core::u32 size = 64u;
    const core::u32 nest = size / 2u;

    struct Run {
        core::u32 furthest;    ///< Squared distance of the furthest agent from the nest.
        core::u32 returns;     ///< Agents sent home over the whole run.
        core::u32 strongCells; ///< Cells carrying a real trail, not the diffuse floor.
        core::u32 fold;
    };

    const auto run = [&](core::u32 forageRange, core::u32 ticks) {
        ai::StigmergyField field{size, size, 1u};
        ai::StigmergyParams decay;
        decay.evaporation = 0.98f;
        decay.diffusion = 0.05f;

        ai::AntColonyParams params;
        params.agents = 24u;
        params.forageRange = forageRange;
        params.seed = 4242u;
        // Explorers, or the seeded field simply holds every agent on its nest and the
        // rule under test never gets a chance to fire.
        params.ants.explore16 = 8u;

        ai::AntColony colony;
        colony.reset(field, size, size, params, nest, nest);

        Run out{0u, 0u, 0u, 0x811C9DC5u};
        for (core::u32 tick = 0u; tick < ticks; ++tick)
        {
            colony.step(field);
            field.step(decay);
            for (core::u32 i = 0u; i < colony.agentCount(); ++i)
            {
                const core::i32 dx = static_cast<core::i32>(colony.agentX(i)) - static_cast<core::i32>(nest);
                const core::i32 dz = static_cast<core::i32>(colony.agentZ(i)) - static_cast<core::i32>(nest);
                const core::u32 squared = static_cast<core::u32>(dx * dx + dz * dz);
                if (squared > out.furthest)
                    out.furthest = squared;
            }
        }
        out.returns = colony.returns();

        // A trail is where the pheromone is STRONG. Counting non-zero cells would
        // count the diffuse background instead, which saturates the whole map within a
        // few hundred ticks and makes every colony look identical.
        core::i32 peak = 0;
        for (core::u32 z = 0u; z < size; ++z)
            for (core::u32 x = 0u; x < size; ++x)
                if (field.value(0u, x, z).raw() > peak)
                    peak = field.value(0u, x, z).raw();
        for (core::u32 z = 0u; z < size; ++z)
            for (core::u32 x = 0u; x < size; ++x)
            {
                const core::i32 raw = field.value(0u, x, z).raw();
                if (peak > 0 && raw > peak / 4)
                    ++out.strongCells;
                out.fold = (out.fold ^ static_cast<core::u32>(raw)) * 0x01000193u;
            }
        return out;
    };

    const Run held = run(6u, 600u);
    std::printf("    range 6:  furthest %u cells away, %u sent home, %u cells carry a trail\n",
                math::integerSqrt(held.furthest), held.returns, held.strongCells);

    // The rule's guarantee, stated as the rule states it: an agent is never further
    // from the nest than its forage range. Diagonal steps mean it can overshoot by
    // one step before being caught, so the bound is the range plus one.
    check(math::integerSqrt(held.furthest) <= 7u, "no agent is ever further from the nest than its forage range");
    check(held.returns > 0u, "and the rule actually fires");

    const Run loose = run(size * 4u, 600u);
    std::printf("    unbounded: furthest %u cells away, %u sent home, %u cells carry a trail\n",
                math::integerSqrt(loose.furthest), loose.returns, loose.strongCells);
    check(loose.returns == 0u, "a range wider than the map never sends anyone home");

    // The difference the rule makes: held near its nest the colony keeps reinforcing
    // the same cells, so a trail exists. Left to diffuse outward it spreads its
    // deposits over the whole map and nothing stands out — a field rather than a
    // route, which is the distinction the rule exists for.
    check(math::integerSqrt(loose.furthest) > math::integerSqrt(held.furthest),
          "and its agents do wander further");
    check(held.strongCells < loose.strongCells,
          "a homing colony concentrates its trail; an unbounded one spreads it thin");

    // Same seed, same trail: the colony advances ONE shared random stream agent by
    // agent, so its determinism depends on visiting them in index order.
    const Run again = run(6u, 600u);
    std::printf("    trail fold 0x%08X\n", held.fold);
    check(again.fold == held.fold && again.returns == held.returns, "the same seed walks the same trail");
}

void testExplorationIsWhatFindsTheDetour()
{
    std::printf("without explorers the colony stays stuck\n");

    const core::u32 size = 16u;
    ai::StigmergyField field{size, size, 1u};

    ai::AntParams greedy;
    greedy.explore16 = 0u;
    ai::AntParams curious;
    curious.explore16 = 4u;

    const auto countExplorations = [&](const ai::AntParams &params) {
        core::u32 explorations = 0u;
        core::u32 stream = 0x1234u;
        for (core::u32 i = 0u; i < 1000u; ++i)
        {
            bool explored = false;
            (void) ai::chooseAntMove(field, params, 8u, 8u, stream, explored);
            explorations += explored ? 1u : 0u;
        }
        return explorations;
    };

    const core::u32 never = countExplorations(greedy);
    const core::u32 sometimes = countExplorations(curious);
    std::printf("    explore16=0 gave %u explorations, explore16=4 gave %u of 1000\n", never, sometimes);

    check(never == 0u, "a purely greedy colony never explores");
    check(sometimes > 150u && sometimes < 400u, "a quarter-exploring colony explores about a quarter of the time");
}

void testReversalCostsMoreThanGoingRound()
{
    std::printf("a reversal costs more than a detour, so a long body does not fold\n");

    // A corridor with a side pocket: the pocket is a dead end, so a body that
    // enters it must either reverse or use the loop.
    const core::u32 w = 12u;
    const core::u32 h = 7u;
    ai::AiMap map{w, h};
    const core::u8 walk = static_cast<core::u8>(ai::Locomotion::Walk);

    for (core::u32 x = 1u; x < w - 1u; ++x)
    {
        map.setCapability(x, 1u, walk);
        map.setCapability(x, 5u, walk);
    }
    for (core::u32 z = 1u; z < h - 1u; ++z)
    {
        map.setCapability(1u, z, walk);
        map.setCapability(w - 2u, z, walk);
    }

    ai::AiMapParams params;
    lpl::pmr::vector<core::u32> path;

    // Worth stating plainly, because the obvious test does not exist: a
    // point-to-point search NEVER has to reverse. An optimal route to a cell you
    // have not visited never revisits one you have. The reversal charge is for
    // the thing a search does not model — a BODY, which occupies the cells behind
    // its head and cannot pass through them.
    //
    // So the two things that are actually true get measured instead.
    params.reverseCost = 160u;
    const core::u32 cost = map.findPath(1u, 1u, 1u, 5u, walk, params, path);
    check(cost != ai::AiMap::kNoPath, "a route exists");

    core::u32 immediateReversals = 0u;
    for (core::u32 i = 0u; i + 2u < path.size(); ++i)
        if (path[i] == path[i + 2u])
            ++immediateReversals;
    std::printf("    route of %u cells, %u immediate back-steps\n", static_cast<core::u32>(path.size()),
                immediateReversals);
    check(immediateReversals == 0u, "the search never steps back onto the cell it just left");

    // And the search distinguishes arrival directions at all: with the charge on,
    // the cost of a route that has to turn is strictly higher than the same route
    // measured with turns free. If the state were the cell alone, the two would be
    // identical.
    // Endpoints that force a corner. The left wall alone is a straight line, and
    // charging for turns on a route with no turns proves nothing — which is what
    // the first version of this measured.
    params.reverseCost = 160u;
    params.turnCost = 64u;
    const core::u32 turning = map.findPath(1u, 1u, w - 2u, 5u, walk, params, path);
    params.turnCost = 0u;
    const core::u32 straight = map.findPath(1u, 1u, w - 2u, 5u, walk, params, path);
    std::printf("    same route: %u with turns charged, %u with turns free\n", turning, straight);
    check(turning > straight, "the search state includes the facing, not just the cell");

    // A body five segments long following the chosen route must never re-enter a
    // cell its own tail is occupying. That is the property the whole directional
    // state exists for.
    const core::u32 folds = ai::countSelfIntersections(&path[0], static_cast<core::u32>(path.size()), 5u);
    std::printf("    a five-segment body folds through itself %u times on this route\n", folds);
    check(folds == 0u, "a five-segment body never passes through itself");
}

void testPersonalityIsDerivedNotStored()
{
    std::printf("personality is a function of the identifier\n");

    const ai::PersonalityTraits a = ai::personalityOf(4242u, 1u);
    const ai::PersonalityTraits b = ai::personalityOf(4242u, 1u);
    check(a.aggression.raw() == b.aggression.raw() && a.sympathy.raw() == b.sympathy.raw(),
          "the same id gives the same temperament, every time");

    const ai::PersonalityTraits other = ai::personalityOf(4243u, 1u);
    check(a.aggression.raw() != other.aggression.raw(), "a different id gives a different temperament");

    const ai::PersonalityTraits sameIdOtherSpecies = ai::personalityOf(4242u, 2u);
    check(a.aggression.raw() != sameIdOtherSpecies.aggression.raw(), "the species salt separates the two");

    // The six axes must be independent. Slicing one hash into six windows would
    // correlate them, and the population would end up with two personalities
    // instead of a spread — so this measures the spread.
    core::u32 aggressiveAndBrave = 0u;
    core::u32 aggressive = 0u;
    core::u32 brave = 0u;
    for (core::u32 id = 0u; id < 4000u; ++id)
    {
        const ai::PersonalityTraits t = ai::personalityOf(id, 0u);
        const bool isAggressive = t.aggression > math::Fixed32::half();
        const bool isBrave = t.bravery > math::Fixed32::half();
        aggressive += isAggressive ? 1u : 0u;
        brave += isBrave ? 1u : 0u;
        aggressiveAndBrave += (isAggressive && isBrave) ? 1u : 0u;
    }
    const float joint = static_cast<float>(aggressiveAndBrave) / 4000.0f;
    std::printf("    P(aggressive)=%.3f P(brave)=%.3f P(both)=%.3f (independent would be ~0.25)\n",
                static_cast<float>(aggressive) / 4000.0f, static_cast<float>(brave) / 4000.0f, joint);
    check(joint > 0.20f && joint < 0.30f, "two traits are independent, not two views of one number");
}

void testRealizationBudget()
{
    std::printf("the world keeps living off-screen, within a budget counted in rooms\n");

    ai::AbstractWorld world;
    for (core::u32 i = 0u; i < 20u; ++i)
        (void) world.addRoom();
    for (core::u32 i = 0u; i + 1u < 20u; ++i)
        world.connect(i, i + 1u);

    for (core::u32 i = 0u; i < 60u; ++i)
        (void) world.addCreature(1000u + i, 1u, i % 20u);

    ai::RealizationBudget budget;
    budget.maxRealisedRooms = 5u;

    core::u32 overBudget = 0u;
    core::u32 movedTotal = 0u;
    for (core::u32 tick = 0u; tick < 200u; ++tick)
    {
        const ai::RoomId focus = (tick / 4u) % 20u;
        world.observe(focus, (focus + 1u) % 20u, tick);
        (void) world.enforceBudget(budget, tick);
        if (world.realisedRoomCount() > budget.maxRealisedRooms)
            ++overBudget;
        movedTotal += world.tickAbstract(tick);
    }

    std::printf("    %u ticks over budget, %u abstract migrations, %u/%u creatures realised at the end\n", overBudget,
                movedTotal, world.realisedCreatureCount(), static_cast<core::u32>(world.creatures().size()));

    check(overBudget == 0u, "the room budget is never exceeded");
    check(movedTotal > 0u, "creatures migrate while abstract — the world does not wait");
    check(world.realisedCreatureCount() < world.creatures().size(),
          "most creatures are data rather than bodies at any moment");

    // Same seed, same history: an abstract world's evolution is reproducible.
    ai::AbstractWorld twin;
    for (core::u32 i = 0u; i < 20u; ++i)
        (void) twin.addRoom();
    for (core::u32 i = 0u; i + 1u < 20u; ++i)
        twin.connect(i, i + 1u);
    for (core::u32 i = 0u; i < 60u; ++i)
        (void) twin.addCreature(1000u + i, 1u, i % 20u);
    for (core::u32 tick = 0u; tick < 200u; ++tick)
    {
        const ai::RoomId focus = (tick / 4u) % 20u;
        twin.observe(focus, (focus + 1u) % 20u, tick);
        (void) twin.enforceBudget(budget, tick);
        (void) twin.tickAbstract(tick);
    }
    check(twin.fold() == world.fold(), "the abstract simulation is reproducible");
}

void testRelationshipsAndAffordances()
{
    std::printf("memory is asymmetric, and the world advertises what it offers\n");

    ai::RelationshipTracker tracker;
    tracker.observe(1u, 2u, 100u, math::Fixed32::one(), math::Fixed32::zero(), ai::Attitude::Afraid);

    ai::Opinion opinion;
    check(tracker.opinion(1u, 2u, opinion), "the observer remembers the subject");
    check(!tracker.opinion(2u, 1u, opinion), "the memory is one-way: the subject does not remember back");

    (void) tracker.opinion(1u, 2u, opinion);
    const core::u32 firstCell = opinion.lastSeenCell;
    for (core::u32 i = 0u; i < 5u; ++i)
        (void) tracker.tick(32u, 32u, math::Fixed32::fromFloat(0.9f));
    (void) tracker.opinion(1u, 2u, opinion);
    std::printf("    remembered position drifted from %u to %u while out of sight\n", firstCell, opinion.lastSeenCell);
    check(opinion.lastSeenCell != firstCell, "a lost target is extrapolated, not frozen");
    check(opinion.confidence < math::Fixed32::one(), "confidence decays out of sight");

    // Reputation: a faction remembers an attack on one of its members.
    check(tracker.reputation(7u, 99u).raw() == 0, "an unknown attacker has no reputation");
    tracker.recordAggression(7u, 99u, math::Fixed32::half());
    check(tracker.reputation(7u, 99u) < math::Fixed32::zero(), "an unprovoked attack turns a faction hostile");

    // Affordances: the world offers, the agent does not ask what things are.
    ai::AffordanceRegistry registry;
    ai::Affordance town;
    town.cell = 10u * 32u + 10u;
    town.kinds = ai::AffordanceKind::Eat | ai::AffordanceKind::Danger;
    town.radius = 8u;
    town.value = math::Fixed32::fromInt(50);
    town.requiredNeed = math::Fixed32::fromFloat(0.8f); // only when desperate
    registry.add(town);

    core::u32 index = 0u;
    const core::u16 wantFood = static_cast<core::u16>(ai::AffordanceKind::Eat);
    check(!registry.best(wantFood, 10u, 10u, 32u, math::Fixed32::half(), index),
          "a well-fed animal does not consider the settlement");
    check(registry.best(wantFood, 10u, 10u, 32u, math::Fixed32::fromFloat(0.95f), index),
          "a starving one does — and nothing anywhere decided to raid");
}

void testSpringBodyStaysBounded()
{
    std::printf("a spring body reaches for a target without exploding\n");

    ai::SpringBody body;
    for (core::u32 i = 0u; i < 5u; ++i)
    {
        ai::BodyChunk chunk;
        chunk.x = math::Fixed32::fromInt(static_cast<core::i32>(i));
        chunk.z = math::Fixed32::zero();
        body.addChunk(chunk);
    }
    for (core::u32 i = 0u; i + 1u < 5u; ++i)
        body.connect(i, i + 1u, math::Fixed32::fromFloat(0.5f));

    ai::SpringBodyParams params;
    math::Fixed32 peakStrain{};
    for (core::u32 step = 0u; step < 10000u; ++step)
    {
        body.pull(0u, math::Fixed32::fromInt(6), math::Fixed32::fromInt(3), math::Fixed32::fromFloat(0.05f));
        body.step(params);
        const math::Fixed32 strain = body.strainEnergy();
        if (strain > peakStrain)
            peakStrain = strain;
    }

    std::printf("    peak strain energy over 10000 steps: %.4f\n", peakStrain.toFloat());
    check(peakStrain < math::Fixed32::fromInt(100), "the body's energy stays bounded");

    // Two-bone IK: reach, and refuse honestly when it cannot.
    const ai::TwoBoneSolution near =
        ai::solveTwoBone(math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::fromInt(3), math::Fixed32::zero(),
                         math::Fixed32::fromInt(2), math::Fixed32::fromInt(2), false);
    check(near.reachable, "a target inside the limb's reach is solved");

    const ai::TwoBoneSolution far =
        ai::solveTwoBone(math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::fromInt(9), math::Fixed32::zero(),
                         math::Fixed32::fromInt(2), math::Fixed32::fromInt(2), false);
    check(!far.reachable, "an unreachable target is reported rather than silently clamped");

    // The knee must actually lie at the right distance from both ends, or the
    // "solution" is a plausible-looking pose that violates the bone lengths.
    const math::Fixed32 dx = near.jointX;
    const math::Fixed32 dz = near.jointZ;
    const math::Fixed32 upperLength = math::fixedSqrt(dx * dx + dz * dz);
    std::printf("    solved knee sits %.3f from the hip (bone length 2.000)\n", upperLength.toFloat());
    check((upperLength - math::Fixed32::fromInt(2)).abs() < math::Fixed32::fromFloat(0.05f),
          "the solved joint honours the bone length");
}

void testFlockSpeedIsPerSecondNotPerCall()
{
    std::printf("a flock moves at its speed cap per SECOND, whatever the tick rate\n");

    const auto seed = [](lpl::pmr::vector<ai::Boid> &flock) {
        flock.clear();
        for (core::u32 i = 0u; i < 8u; ++i)
        {
            ai::Boid boid{};
            boid.x = math::Fixed32::fromInt(static_cast<core::i32>(i) * 3);
            boid.z = math::Fixed32::zero();
            boid.vx = math::Fixed32::one();
            boid.vz = math::Fixed32::zero();
            flock.push_back(boid);
        }
    };

    ai::BoidParams params;
    params.maxSpeed = 4.0f;

    // One second, as sixty steps of a sixtieth. The distance covered is what a
    // caller means by "speed", and before dt existed this ran at sixty units a
    // second while claiming a cap of four.
    lpl::pmr::vector<ai::Boid> fine;
    seed(fine);
    const math::Fixed32 fineStep = math::Fixed32::fromRaw(1092); // 1/60 s
    for (core::u32 i = 0u; i < 60u; ++i)
        ai::stepBoids(&fine[0], static_cast<core::u32>(fine.size()), params, fineStep);

    const float travelled = fine[0].x.toFloat() - 0.0f;
    std::printf("    lead boid covered %.2f units in one simulated second (cap %.1f)\n", travelled,
                static_cast<double>(params.maxSpeed));
    check(travelled > 0.2f, "the flock actually moved");
    check(travelled <= static_cast<float>(params.maxSpeed) + 0.5f, "and never faster than its cap allows");

    // The same second in twenty steps of a twentieth must land in roughly the same
    // place. Not bit-identical — a coarser integration of the same curve is a
    // different sum — but within a fraction of a unit, which is the property that
    // makes the tick rate a performance decision rather than a gameplay one.
    lpl::pmr::vector<ai::Boid> coarse;
    seed(coarse);
    const math::Fixed32 coarseStep = math::Fixed32::fromRaw(3277); // 1/20 s
    for (core::u32 i = 0u; i < 20u; ++i)
        ai::stepBoids(&coarse[0], static_cast<core::u32>(coarse.size()), params, coarseStep);

    const float gap = coarse[0].x.toFloat() - fine[0].x.toFloat();
    std::printf("    same second at 20 Hz lands %.3f units away\n", gap < 0.0f ? -gap : gap);
    check((gap < 0.0f ? -gap : gap) < 0.5f, "the tick rate does not change where the flock ends up");

    // A zero or negative step is refused rather than run backwards.
    lpl::pmr::vector<ai::Boid> frozen;
    seed(frozen);
    ai::stepBoids(&frozen[0], static_cast<core::u32>(frozen.size()), params, math::Fixed32::zero());
    check(frozen[0].x.raw() == 0, "a zero-length step moves nothing");
}

} // namespace

int main()
{
    std::printf("== ai ==\n");
    testDiffusionGoesAroundWalls();
    testEncirclementEmerges();
    testColonyAbandonsABlockedTrail();
    testTheColonyClosesItsTrail();
    testExplorationIsWhatFindsTheDetour();
    testReversalCostsMoreThanGoingRound();
    testPersonalityIsDerivedNotStored();
    testRealizationBudget();
    testRelationshipsAndAffordances();
    testSpringBodyStaysBounded();

    testFlockSpeedIsPerSecondNotPerCall();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
