/**
 * @file LivingRecipe.cpp
 * @brief The living parity run.
 *
 * One function, and its whole job is to be boring in a very specific way: every
 * loop visits in a fixed order, every random draw comes from a stream seeded from
 * the recipe, and nothing consults a clock, a container's address or an
 * iteration order that depends on hashing. Those are the four ways a simulation
 * usually stops being reproducible, and each of them has cost this project a day
 * at some point.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecology/LivingRecipe.hpp>

#include <lpl/procgen/Grid.hpp>
#include <lpl/procgen/Random.hpp>
#include <lpl/std/vector.hpp>

namespace lpl::ecology {

namespace {

constexpr core::u32 kFnvOffset = 0x811C9DC5u;
constexpr core::u32 kFnvPrime = 0x01000193u;

/// Folds one 32-bit word, byte by byte, exactly as every other gate does.
void foldWord(core::u32 &hash, core::u32 word) noexcept
{
    for (core::u32 byte = 0u; byte < 4u; ++byte)
    {
        hash ^= (word >> (byte * 8u)) & 0xFFu;
        hash *= kFnvPrime;
    }
}

/// Folds a Fixed32 by its RAW Q16.16 word — never by a decimal rendering of it.
void foldFixed(core::u32 &hash, math::Fixed32 value) noexcept { foldWord(hash, static_cast<core::u32>(value.raw())); }

} // namespace

LivingResult runLiving(const LivingRecipe &recipe)
{
    LivingResult result{};

    if (recipe.width == 0u || recipe.depth == 0u || recipe.channels == 0u || recipe.rooms == 0u)
        return result;

    // ── The trophic web ──────────────────────────────────────────────────────
    //
    // Four levels, each eating the one below, so the cascade the module exists to
    // produce is actually in the run: remove the top and the mesopredator is
    // released. A two-species web would fold just as deterministically and prove
    // a tenth as much.
    TrophicWeb web;
    core::u32 primary = 0u;
    {
        const core::u32 declared =
            recipe.speciesCount < kMaxLivingSpecies ? recipe.speciesCount : kMaxLivingSpecies;
        for (core::u32 i = 0u; i < declared; ++i)
            (void) web.add(recipe.species[i].params, recipe.species[i].initial, recipe.species[i].preyIndex);

        // The heredity pass reads the pressure on the first CONSUMER, which is
        // what its collapse threshold is relative to. With no consumer declared
        // it falls back to species 0, so a producer-only web still runs.
        for (core::u32 i = 0u; i < declared; ++i)
            if (recipe.species[i].preyIndex != Species::kNoPrey)
            {
                primary = i;
                break;
            }
    }
    if (web.species.empty())
        return result;

    // ── The breeding population ──────────────────────────────────────────────
    lpl::pmr::vector<Genome> population;
    population.reserve(recipe.genomes);
    core::u32 heredityStream = recipe.seed ^ 0x5EED0001u;
    {
        Genome founder{};
        for (core::u32 i = 0u; i < recipe.genomes; ++i)
            population.push_back(mutate(founder, 8u, 0.20f, heredityStream));
    }

    // ── The field and the agents on it ───────────────────────────────────────
    ai::StigmergyField field{recipe.width, recipe.depth, recipe.channels};

    lpl::pmr::vector<core::u32> antX;
    lpl::pmr::vector<core::u32> antZ;
    antX.reserve(recipe.ants);
    antZ.reserve(recipe.ants);
    {
        procgen::Random placement{recipe.seed ^ 0xA47C0102u};
        for (core::u32 i = 0u; i < recipe.ants; ++i)
        {
            antX.push_back(placement.below(recipe.width));
            antZ.push_back(placement.below(recipe.depth));
        }
    }

    // One nest cell, so the trails have somewhere to converge on rather than
    // wandering into a uniform field forever.
    const core::u32 nest = (recipe.depth / 2u) * recipe.width + (recipe.width / 2u);
    ai::seedPheromoneField(field, 0u, &nest, 1u, math::Fixed32::fromInt(40));

    // ── The flock ────────────────────────────────────────────────────────────
    lpl::pmr::vector<ai::Boid> flock;
    flock.reserve(recipe.boids);
    {
        procgen::Random spawn{recipe.seed ^ 0xB01D0303u};
        for (core::u32 i = 0u; i < recipe.boids; ++i)
        {
            ai::Boid boid{};
            boid.x = math::Fixed32::fromInt(spawn.range(-8, 8));
            boid.z = math::Fixed32::fromInt(spawn.range(-8, 8));
            boid.vx = spawn.unit() - math::Fixed32::half();
            boid.vz = spawn.unit() - math::Fixed32::half();
            flock.push_back(boid);
        }
    }

    // ── The abstract world ───────────────────────────────────────────────────
    //
    // A ring with chords: every room has at least two exits, so a creature never
    // sits in a dead end for the whole run and the migration count stays a real
    // measure rather than a constant.
    ai::AbstractWorld abstractWorld;
    for (core::u32 i = 0u; i < recipe.rooms; ++i)
        (void) abstractWorld.addRoom();
    for (core::u32 i = 0u; i < recipe.rooms; ++i)
        abstractWorld.connect(i, (i + 1u) % recipe.rooms);
    for (core::u32 i = 0u; i + 3u < recipe.rooms; i += 3u)
        abstractWorld.connect(i, i + 3u);
    {
        procgen::Random spread{recipe.seed ^ 0xC0DE0404u};
        for (core::u32 i = 0u; i < recipe.creatures; ++i)
            (void) abstractWorld.addCreature(i + 1u, i % 3u, spread.below(recipe.rooms));
    }

    // ── The social layer ─────────────────────────────────────────────────────
    lpl::pmr::vector<PackMember> members;
    members.reserve(recipe.packMembers);
    for (core::u32 i = 0u; i < recipe.packMembers; ++i)
    {
        PackMember member{};
        member.id = i + 1u;
        member.lineage = i % 4u;
        member.fitness = math::Fixed32::fromInt(static_cast<core::i32>(i % 7u));
        members.push_back(member);
    }
    core::u32 socialStream = recipe.seed ^ 0x50C10505u;

    // ── The run ──────────────────────────────────────────────────────────────
    core::u32 antStream = recipe.seed ^ 0xA57E0606u;
    lpl::pmr::vector<core::u32> trail;
    trail.reserve(recipe.ticks);

    for (core::u32 tick = 0u; tick < recipe.ticks; ++tick)
    {
        // 1. Populations. Lotka-Volterra with a carrying capacity and a refuge
        //    floor, so a collapse bottoms out instead of going negative.
        web.step(1u);

        // 2. Heredity. The whole population breeds once per tick, each pair drawn
        //    from the same stream, so a generation is a deterministic function of
        //    the previous one.
        {
            const math::Fixed32 local = web.species[primary].population;
            const math::Fixed32 capacity = web.species[primary].params.capacity;
            lpl::pmr::vector<Genome> next;
            next.reserve(population.size());
            for (core::u32 i = 0u; i < population.size(); ++i)
            {
                procgen::Random pick{heredityStream ^ (0x9E3779B9u * (i + 1u))};
                const core::u32 a = pick.below(static_cast<core::u32>(population.size()));
                const core::u32 b = pick.below(static_cast<core::u32>(population.size()));
                next.push_back(breed(population[a], population[b], local, capacity, recipe.heredity, heredityStream));
            }
            population = next;
        }

        // 3. The field: agents walk it, deposit behind them, then it evaporates.
        //    In that order — evaporating first would erase the trail the agents
        //    just laid, which is the classic way to make an ACO field look like
        //    noise.
        for (core::u32 i = 0u; i < antX.size(); ++i)
        {
            bool explored = false;
            const core::u32 direction =
                ai::chooseAntMove(field, recipe.foraging, antX[i], antZ[i], antStream, explored);
            if (direction != ai::StigmergyField::kNoDirection)
            {
                const core::i32 nx = static_cast<core::i32>(antX[i]) + procgen::kNeighbor8X[direction];
                const core::i32 nz = static_cast<core::i32>(antZ[i]) + procgen::kNeighbor8Z[direction];
                if (nx >= 0 && nz >= 0 && static_cast<core::u32>(nx) < recipe.width &&
                    static_cast<core::u32>(nz) < recipe.depth)
                {
                    antX[i] = static_cast<core::u32>(nx);
                    antZ[i] = static_cast<core::u32>(nz);
                }
            }
            trail.clear();
            trail.push_back(antZ[i] * recipe.width + antX[i]);
            field.depositTrail(0u, &trail[0], 1u, recipe.foraging.depositQuality);
            field.deposit(recipe.channels > 1u ? 1u : 0u, antX[i], antZ[i], math::Fixed32::one());
        }
        field.step(recipe.stigmergy);

        // 4. The flock.
        if (!flock.empty())
            ai::stepBoids(&flock[0], static_cast<core::u32>(flock.size()), recipe.flock, recipe.stepSeconds);

        // 5. The abstract world. The focus walks the ring, so the realisation
        //    budget is genuinely exercised: a stationary focus would realise the
        //    same four rooms for forty-eight ticks and never abstract one.
        const ai::RoomId focus = tick % recipe.rooms;
        abstractWorld.observe(focus, (focus + 1u) % recipe.rooms, tick);
        (void) abstractWorld.enforceBudget(recipe.budget, tick);
        result.migrations += abstractWorld.tickAbstract(tick);

        // 6. The social layer.
        const PackEvents events =
            stepPacks(&members[0], static_cast<core::u32>(members.size()), recipe.packs, socialStream);
        result.alphaChanges += events.alphaChanges;
    }

    // ── The folds ────────────────────────────────────────────────────────────
    result.populationSignature = web.fold();
    result.stigmergySignature = field.fold();

    core::u32 genomeHash = kFnvOffset;
    for (core::u32 i = 0u; i < population.size(); ++i)
    {
        foldFixed(genomeHash, population[i].maxSpeed);
        foldFixed(genomeHash, population[i].vision);
        foldFixed(genomeHash, population[i].strength);
        foldFixed(genomeHash, population[i].absorption);
        foldFixed(genomeHash, population[i].size);
    }
    result.genomeSignature = genomeHash;

    // The social signature covers both halves of the social state: the abstract
    // world's own fold, then every pack member. Folding only the first would let
    // the pack layer diverge silently, which is precisely the failure mode the
    // separate signatures exist to prevent.
    core::u32 socialHash = kFnvOffset;
    foldWord(socialHash, abstractWorld.fold());
    // The flock, folded rather than merely stepped. It was run by this recipe from
    // the start and folded by nothing, which is the emptiest kind of coverage: the
    // gate paid for the computation and would not have noticed the result diverging.
    for (core::u32 i = 0u; i < flock.size(); ++i)
    {
        foldFixed(socialHash, flock[i].x);
        foldFixed(socialHash, flock[i].z);
        foldFixed(socialHash, flock[i].vx);
        foldFixed(socialHash, flock[i].vz);
    }
    for (core::u32 i = 0u; i < members.size(); ++i)
    {
        foldWord(socialHash, members[i].id);
        foldWord(socialHash, members[i].pack);
        foldWord(socialHash, members[i].lineage);
        foldFixed(socialHash, members[i].fitness);
        foldWord(socialHash, members[i].alpha ? 1u : 0u);
        foldWord(socialHash, members[i].alive ? 1u : 0u);
    }
    result.socialSignature = socialHash;

    // ── The counters ─────────────────────────────────────────────────────────
    for (core::u32 i = 0u; i < web.species.size(); ++i)
        if (web.species[i].population <= web.species[i].params.refuge)
            ++result.extinctions;

    const PopulationStats stats =
        strengthStats(population.empty() ? nullptr : &population[0], static_cast<core::u32>(population.size()));
    for (core::u32 i = 0u; i < population.size(); ++i)
        if (isAnomaly(population[i], stats, recipe.heredity))
            ++result.anomalies;

    result.realisedRooms = abstractWorld.realisedRoomCount();

    const math::Fixed32 floorValue = math::Fixed32::fromFloat(recipe.stigmergy.floor);
    for (core::u32 c = 0u; c < recipe.channels; ++c)
        for (core::u32 z = 0u; z < recipe.depth; ++z)
            for (core::u32 x = 0u; x < recipe.width; ++x)
                if (field.value(c, x, z) > floorValue)
                    ++result.trailCells;

    // "Well formed" is not "nothing went wrong" — a collapse is a legitimate
    // outcome. It is the weaker claim that the run actually ran: the field holds
    // a trail, the population survived, and the budget did its job.
    result.ok = (result.trailCells != 0u && !population.empty() && result.realisedRooms != 0u &&
                 result.realisedRooms <= recipe.budget.maxRealisedRooms) ?
                    1u :
                    0u;
    return result;
}

} // namespace lpl::ecology
