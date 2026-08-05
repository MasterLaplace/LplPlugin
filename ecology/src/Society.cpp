/**
 * @file Society.cpp
 * @brief Implementation of packs, isolation, overflow and invasion.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecology/Society.hpp>

#include <lpl/math/FixedMath.hpp>

namespace lpl::ecology {

namespace {

core::u32 nextRandom(core::u32 &stream)
{
    stream ^= stream << 13;
    stream ^= stream >> 17;
    stream ^= stream << 5;
    if (stream == 0u)
        stream = 0x9E3779B9u;
    return stream;
}

/**
 * @brief The member that should lead a pack.
 *
 * Highest fitness, ties broken by the lower identifier. The tie-break is
 * load-bearing rather than tidy: a pack of siblings has near-identical fitness,
 * and without it the leader would be whichever member the loop happened to reach
 * first — so two machines running the same simulation would crown different
 * animals and everything downstream would diverge.
 */
core::u32 bestOfPack(const PackMember *members, core::u32 count, core::u32 pack)
{
    core::u32 best = 0xFFFFFFFFu;
    for (core::u32 i = 0u; i < count; ++i)
    {
        if (!members[i].alive || members[i].pack != pack)
            continue;
        if (best == 0xFFFFFFFFu)
        {
            best = i;
            continue;
        }
        if (members[i].fitness > members[best].fitness ||
            (members[i].fitness == members[best].fitness && members[i].id < members[best].id))
            best = i;
    }
    return best;
}

core::u32 packSize(const PackMember *members, core::u32 count, core::u32 pack)
{
    core::u32 size = 0u;
    for (core::u32 i = 0u; i < count; ++i)
        if (members[i].alive && members[i].pack == pack)
            ++size;
    return size;
}

core::u32 highestPackId(const PackMember *members, core::u32 count)
{
    core::u32 highest = 0u;
    for (core::u32 i = 0u; i < count; ++i)
        if (members[i].pack != kSolitary && members[i].pack + 1u > highest)
            highest = members[i].pack + 1u;
    return highest;
}

} // namespace

PackEvents stepPacks(PackMember *members, core::u32 count, const PackParams &params, core::u32 &stream)
{
    PackEvents events;
    if (members == nullptr || count == 0u)
        return events;

    core::u32 nextPack = highestPackId(members, count);

    // ── Formation: lone kin band together ───────────────────────────────────
    for (core::u32 i = 0u; i < count; ++i)
    {
        if (!members[i].alive || members[i].pack != kSolitary)
            continue;

        // Look for an existing pack of the same lineage first — adoption is
        // cheaper than founding, ecologically and computationally.
        core::u32 joined = kSolitary;
        if (params.adoptStrays)
        {
            for (core::u32 j = 0u; j < count && joined == kSolitary; ++j)
            {
                if (!members[j].alive || members[j].pack == kSolitary)
                    continue;
                if (members[j].lineage != members[i].lineage)
                    continue;
                if (packSize(members, count, members[j].pack) >= params.maxSize)
                    continue;
                joined = members[j].pack;
            }
        }

        if (joined != kSolitary)
        {
            members[i].pack = joined;
            ++events.adopted;
            continue;
        }

        // Otherwise found one with another stray of the same lineage.
        for (core::u32 j = i + 1u; j < count; ++j)
        {
            if (!members[j].alive || members[j].pack != kSolitary)
                continue;
            if (members[j].lineage != members[i].lineage)
                continue;
            members[i].pack = nextPack;
            members[j].pack = nextPack;
            ++nextPack;
            ++events.formed;
            break;
        }
    }

    // ── Budding: a pack that outgrows its territory splits ──────────────────
    const core::u32 packCount = nextPack;
    for (core::u32 pack = 0u; pack < packCount; ++pack)
    {
        core::u32 size = packSize(members, count, pack);
        if (size <= params.maxSize)
            continue;

        // The splinter is led by the SECOND best — the ambitious beta who will
        // never inherit. Taking the weakest members instead would make budding a
        // cull; taking the strongest would decapitate the parent pack.
        const core::u32 alpha = bestOfPack(members, count, pack);
        core::u32 beta = 0xFFFFFFFFu;
        for (core::u32 i = 0u; i < count; ++i)
        {
            if (!members[i].alive || members[i].pack != pack || i == alpha)
                continue;
            if (beta == 0xFFFFFFFFu || members[i].fitness > members[beta].fitness ||
                (members[i].fitness == members[beta].fitness && members[i].id < members[beta].id))
                beta = i;
        }
        if (beta == 0xFFFFFFFFu)
            continue;

        const core::u32 splinter = nextPack++;
        core::u32 moved = 0u;
        const core::u32 target = size / 2u;
        members[beta].pack = splinter;
        ++moved;
        for (core::u32 i = 0u; i < count && moved < target; ++i)
        {
            if (!members[i].alive || members[i].pack != pack || i == alpha)
                continue;
            members[i].pack = splinter;
            ++moved;
        }
        ++events.budded;
    }

    // ── Alpha election ──────────────────────────────────────────────────────
    for (core::u32 pack = 0u; pack < nextPack; ++pack)
    {
        const core::u32 leader = bestOfPack(members, count, pack);
        for (core::u32 i = 0u; i < count; ++i)
        {
            if (!members[i].alive || members[i].pack != pack)
                continue;
            const bool shouldLead = i == leader;
            if (members[i].alpha != shouldLead)
            {
                members[i].alpha = shouldLead;
                if (shouldLead)
                    ++events.alphaChanges;
            }
        }
    }

    // ── Dissolution: a pack too small to be one ─────────────────────────────
    for (core::u32 pack = 0u; pack < nextPack; ++pack)
    {
        const core::u32 size = packSize(members, count, pack);
        if (size == 0u || size >= params.minSize)
            continue;
        for (core::u32 i = 0u; i < count; ++i)
            if (members[i].alive && members[i].pack == pack)
            {
                members[i].pack = kSolitary;
                members[i].alpha = false;
                ++events.scattered;
            }
        ++events.dissolved;
    }

    (void) stream;
    return events;
}

PackEvents killMember(PackMember *members, core::u32 count, core::u32 id, const PackParams &params, core::u32 &stream)
{
    PackEvents events;
    if (members == nullptr || count == 0u)
        return events;

    core::u32 victim = 0xFFFFFFFFu;
    for (core::u32 i = 0u; i < count; ++i)
        if (members[i].id == id && members[i].alive)
            victim = i;
    if (victim == 0xFFFFFFFFu)
        return events;

    const bool wasAlpha = members[victim].alpha;
    const core::u32 pack = members[victim].pack;
    members[victim].alive = false;
    members[victim].alpha = false;
    members[victim].pack = kSolitary;

    if (!wasAlpha || pack == kSolitary)
        return events;

    // The consequence a trophy hunter does not price in. A pack that loses its
    // leader may hold together under the next in line — or shatter into solitary
    // aggressive animals with no territory, which is how a profitable hunt turns
    // into a problem on the trade roads a week later.
    if ((nextRandom(stream) & 0xFu) < params.dissolutionChance16)
    {
        for (core::u32 i = 0u; i < count; ++i)
            if (members[i].alive && members[i].pack == pack)
            {
                members[i].pack = kSolitary;
                members[i].alpha = false;
                ++events.scattered;
            }
        ++events.dissolved;
    }
    return events;
}

Genome applyIslandRule(const Genome &genome, math::Fixed32 ancestralSize, bool isolated, const IslandParams &params)
{
    if (!isolated)
        return genome;

    // The direction comes from what the species WAS, not what it currently is.
    // Reading the current size turns the threshold into an unstable equilibrium
    // that both a mouse and an elephant converge onto and never leave — measured
    // at 0.996 and 1.003 after two hundred generations, which is the rule
    // inverted. Isolation drives lineages apart; it cannot do that if the rule
    // forgets which lineage it is looking at.
    const math::Fixed32 small = math::Fixed32::fromFloat(params.smallThreshold);
    const math::Fixed32 target = ancestralSize < small ? math::Fixed32::fromFloat(params.giantTarget) :
                                                         math::Fixed32::fromFloat(params.dwarfTarget);
    const math::Fixed32 pressure = math::Fixed32::fromFloat(params.pressure);

    Genome pushed = genome;
    pushed.size = genome.size + (target - genome.size) * pressure;

    // Body size is not a cosmetic gene: a bigger animal hits harder and moves
    // less nimbly. Letting size drift without its consequences would make the
    // island rule a visual effect.
    const math::Fixed32 ratio = genome.size.raw() != 0 ? pushed.size / genome.size : math::Fixed32::one();
    pushed.strength = genome.strength * ratio;
    pushed.maxSpeed = ratio.raw() != 0 ? genome.maxSpeed / ratio : genome.maxSpeed;
    return pushed;
}

core::u32 markIsolatedRegions(const procgen::Grid<core::u32> &regions, core::u32 regionCount,
                              const IslandParams &params, lpl::pmr::vector<core::u8> &outIsolated)
{
    outIsolated.clear();
    outIsolated.resize(regionCount, core::u8{0});
    if (regions.empty() || regionCount == 0u)
        return 0u;

    lpl::pmr::vector<core::u32> sizes(regionCount, 0u);
    for (core::u32 i = 0u; i < regions.cellCount(); ++i)
        if (regions[i] < regionCount)
            ++sizes[regions[i]];

    // A share of the map, not a cell count. A forty-cell pocket is an island on a
    // small map and a puddle on a large one, and only the ratio can tell.
    const math::Fixed32 threshold = math::Fixed32::fromFloat(params.isolationShare) *
                                    math::Fixed32::fromInt(static_cast<core::i32>(regions.cellCount()));

    core::u32 isolated = 0u;
    for (core::u32 r = 0u; r < regionCount; ++r)
    {
        if (sizes[r] == 0u)
            continue;
        if (math::Fixed32::fromInt(static_cast<core::i32>(sizes[r])) < threshold)
        {
            outIsolated[r] = 1u;
            ++isolated;
        }
    }
    return isolated;
}

OverflowState evaluateOverflow(math::Fixed32 population, math::Fixed32 capacity, const math::Fixed32 *energies,
                               core::u32 count, const OverflowParams &params)
{
    OverflowState state;
    if (capacity.raw() <= 0)
        return state;

    state.pressure = population / capacity;
    state.overcrowded = state.pressure > math::Fixed32::fromFloat(params.overcrowdedAbove);

    const math::Fixed32 starvingBelow = math::Fixed32::fromFloat(params.starvingBelow);
    for (core::u32 i = 0u; i < count; ++i)
        if (energies != nullptr && energies[i] < starvingBelow)
            ++state.starving;

    // The raid condition, and it is deliberately not "the region decided to
    // attack". It is: too many animals, and enough of them hungry enough to stop
    // caring about the thing that used to keep them away.
    state.raiding = state.overcrowded && state.starving > 0u;
    return state;
}

void seedInvasion(const Genome &ancestor, core::u32 count, core::f32 spread, core::u32 &stream,
                  lpl::pmr::vector<Genome> &out)
{
    out.clear();
    out.reserve(count);

    const math::Fixed32 swing = math::Fixed32::fromFloat(spread);
    for (core::u32 i = 0u; i < count; ++i)
    {
        // Every founder is a near-copy of one ancestor. That is the founder
        // effect: numerically dangerous, genetically almost uniform — and
        // therefore vulnerable to anything that works on one of them.
        const auto jitter = [&](math::Fixed32 gene) {
            const math::Fixed32 unit = math::Fixed32::fromRaw(static_cast<core::i32>(nextRandom(stream) & 0xFFFFu));
            const math::Fixed32 delta = (unit * math::Fixed32::fromInt(2) - math::Fixed32::one()) * swing;
            return gene * (math::Fixed32::one() + delta);
        };

        Genome founder;
        founder.maxSpeed = jitter(ancestor.maxSpeed);
        founder.vision = jitter(ancestor.vision);
        founder.strength = jitter(ancestor.strength);
        founder.absorption = jitter(ancestor.absorption);
        founder.size = jitter(ancestor.size);
        out.push_back(founder);
    }
}

math::Fixed32 geneticDiversity(const Genome *genomes, core::u32 count)
{
    if (genomes == nullptr || count < 2u)
        return math::Fixed32::zero();

    // Coefficient of variation — the standard deviation over the mean — averaged
    // across the genes. Dividing by the mean is what makes it comparable between
    // a species whose strength is 5 and one whose strength is 500.
    const auto variation = [&](math::Fixed32 Genome::*gene) {
        math::Fixed32 total{};
        for (core::u32 i = 0u; i < count; ++i)
            total = total + genomes[i].*gene;
        const math::Fixed32 mean = total / math::Fixed32::fromInt(static_cast<core::i32>(count));
        if (mean.raw() == 0)
            return math::Fixed32::zero();

        math::Fixed32 variance{};
        for (core::u32 i = 0u; i < count; ++i)
        {
            const math::Fixed32 d = genomes[i].*gene - mean;
            variance = variance + d * d;
        }
        variance = variance / math::Fixed32::fromInt(static_cast<core::i32>(count));
        return math::fixedSqrt(variance) / mean;
    };

    math::Fixed32 total = variation(&Genome::maxSpeed) + variation(&Genome::vision) + variation(&Genome::strength) +
                          variation(&Genome::absorption) + variation(&Genome::size);
    return total / math::Fixed32::fromInt(5);
}

} // namespace lpl::ecology
