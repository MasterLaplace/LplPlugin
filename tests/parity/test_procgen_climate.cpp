/**
 * @file test_procgen_climate.cpp
 * @brief Probes for the six-axis climate field and the profile classification.
 *
 * Three claims are worth testing here, and none of them is "it runs":
 *
 *  1. **Every axis is normalised.** An axis left in absolute units is a threshold
 *     measured against a distribution that moves with the map — the defect that
 *     made six biomes of twelve unreachable, and that flooded a small map with
 *     rivers. So the spread of every axis is measured at four map sizes.
 *  2. **The palette is reachable.** Counting biomes is the only way to notice
 *     that a profile has been walled off by its neighbours.
 *  3. **Climate is continuous.** The whole reason for a nearest-profile lookup
 *     over a continuous field is that a glacier cannot abut a dune sea. That is
 *     checkable: no two adjacent cells may hold biomes whose profiles are far
 *     apart in climate space.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/Climate.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/Hydrology.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

#include <lpl/ecs/Registry.hpp>

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

/// A varied world: continents, ranges, and enough range for a lapse rate to bite.
procgen::Heightfield makeTerrain(core::u32 size, core::u32 seed)
{
    procgen::NoiseParams base;
    base.seed = seed;
    base.frequency = 3.0f / static_cast<core::f32>(size);
    base.amplitude = 20.0f;
    base.octaves = 5u;

    procgen::Heightfield field = procgen::generateNoiseHeightfield(size, size, base);

    procgen::NoiseParams ridges;
    ridges.seed = seed ^ 0x51EDu;
    ridges.frequency = 7.0f / static_cast<core::f32>(size);
    ridges.amplitude = 9.0f;
    ridges.octaves = 4u;
    ridges.kind = procgen::NoiseKind::Ridged;
    procgen::addNoiseLayer(field, ridges);

    procgen::normalizeHeights(field, math::Fixed32::fromFloat(-10.0f), math::Fixed32::fromFloat(16.0f));
    return field;
}

struct Built {
    procgen::Heightfield terrain;
    procgen::ClimateField climate;
    procgen::BiomeMap biomes;
};

Built build(core::u32 size, core::u32 seed)
{
    Built out;
    out.terrain = makeTerrain(size, seed);

    procgen::MoistureParams wet;
    wet.seaLevel = -4.0f;
    const procgen::DrainageNetwork network = procgen::computeDrainage(out.terrain);
    const procgen::Heightfield moisture = procgen::computeMoisture(out.terrain, network, wet);

    procgen::ClimateParams axes;
    axes.seaLevel = -4.0f;
    out.climate = procgen::computeClimate(out.terrain, moisture, network, axes);

    const procgen::Grid<core::u8> lakes = procgen::lakeMask(network, out.terrain);
    out.biomes = procgen::classifyBiomes(out.terrain, out.climate, procgen::BiomeParams{}, &lakes);
    return out;
}

/// Lowest, median and highest value of an axis, as floats for printing.
void axisSpread(const procgen::Heightfield &axis, float &low, float &median, float &high)
{
    math::Fixed32 lo{};
    math::Fixed32 hi{};
    (void) procgen::heightRange(axis, lo, hi);
    low = lo.toFloat();
    high = hi.toFloat();

    // Median through a coarse histogram: exact enough to catch "everything is
    // crushed into the bottom decile", which is the failure being looked for.
    core::u32 buckets[64] = {};
    for (core::u32 i = 0u; i < axis.cellCount(); ++i)
    {
        core::i32 b = (axis[i].raw() * 63) >> 16;
        if (b < 0)
            b = 0;
        if (b > 63)
            b = 63;
        ++buckets[b];
    }
    const core::u32 half = axis.cellCount() / 2u;
    core::u32 running = 0u;
    median = 0.5f;
    for (core::u32 b = 0u; b < 64u; ++b)
    {
        running += buckets[b];
        if (running >= half)
        {
            median = static_cast<float>(b) / 63.0f;
            break;
        }
    }
}

const char *axisName(procgen::ClimateAxis a)
{
    switch (a)
    {
    case procgen::ClimateAxis::Temperature: return "temperature";
    case procgen::ClimateAxis::Moisture: return "moisture";
    case procgen::ClimateAxis::Continentalness: return "continentalness";
    case procgen::ClimateAxis::Erosion: return "erosion";
    case procgen::ClimateAxis::Depth: return "depth";
    case procgen::ClimateAxis::Weirdness: return "weirdness";
    case procgen::ClimateAxis::Count: break;
    }
    return "?";
}

// ─────────────────────────────────────────────────────────────────────────────

void testAxesAreNormalised()
{
    std::printf("climate axes are normalised, at every map size\n");

    const core::u32 sizes[] = {24u, 48u, 96u, 128u};
    for (core::u32 s = 0u; s < 4u; ++s)
    {
        const Built world = build(sizes[s], 1337u);
        check(!world.climate.empty(), "the climate field is built");

        for (core::u32 a = 0u; a < procgen::kClimateAxisCount; ++a)
        {
            float low = 0.0f;
            float median = 0.0f;
            float high = 0.0f;
            axisSpread(world.climate.axes[a], low, median, high);

            const bool inRange = low >= -0.001f && high <= 1.001f;
            if (!inRange)
                std::printf("    %ux%u %-16s [%.3f .. %.3f]\n", sizes[s], sizes[s],
                            axisName(static_cast<procgen::ClimateAxis>(a)), low, high);
            check(inRange, "axis stays inside [0, 1]");
        }

        // The rank-normalised axis is the one that would betray a min-max scaling
        // slipped back in: its median has to sit near the middle whatever the
        // input's shape, and a heavy tail would drag it far below.
        float low = 0.0f;
        float median = 0.0f;
        float high = 0.0f;
        axisSpread(world.climate[procgen::ClimateAxis::Erosion], low, median, high);
        std::printf("    %3ux%-3u erosion median = %.3f\n", sizes[s], sizes[s], median);
        check(median > 0.30f && median < 0.70f, "the rank-normalised axis has a centred median");
    }
}

void testPaletteIsReachable()
{
    std::printf("the biome palette is reachable\n");

    core::u32 everSeen[static_cast<core::u32>(procgen::BiomeId::Count)] = {};
    for (core::u32 seed = 0u; seed < 8u; ++seed)
    {
        const Built world = build(128u, 1337u + seed * 7919u);
        core::u32 counts[static_cast<core::u32>(procgen::BiomeId::Count)] = {};
        procgen::countBiomes(world.biomes, counts);
        for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
            everSeen[i] += counts[i];
    }

    core::u32 reached = 0u;
    std::printf("    ");
    for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
    {
        if (everSeen[i] != 0u)
            ++reached;
        else
            std::printf("[MISSING %s] ", procgen::biomeName(static_cast<procgen::BiomeId>(i)));
    }
    std::printf("%u/%u biomes reached over 8 seeds\n", reached, static_cast<core::u32>(procgen::BiomeId::Count));

    // Not "most of them" — all of them. A profile no world can produce is a lie
    // in the table, and the point of counting is to be told about it.
    check(reached == static_cast<core::u32>(procgen::BiomeId::Count), "every biome is reachable");
}

void testClimateIsContinuous()
{
    std::printf("climate is continuous: no impossible neighbours\n");

    core::u32 profileCount = 0u;
    const procgen::BiomeProfile *profiles = procgen::biomeProfiles(profileCount);

    // Squared distance between two profiles' centres, on the axes either cares
    // about. Two biomes further apart than this may not touch.
    const auto centreDistance = [&](procgen::BiomeId a, procgen::BiomeId b) -> float {
        const procgen::BiomeProfile *pa = nullptr;
        const procgen::BiomeProfile *pb = nullptr;
        for (core::u32 i = 0u; i < profileCount; ++i)
        {
            if (profiles[i].id == a)
                pa = &profiles[i];
            if (profiles[i].id == b)
                pb = &profiles[i];
        }
        if (pa == nullptr || pb == nullptr)
            return 0.0f; // One of them is hydrological (ocean, beach, lake): exempt.

        // Temperature and moisture only, and that is the claim being tested
        // rather than a convenient subset: the guarantee is that a continuous
        // climate has to cross every intermediate isovalue of THOSE two, so a
        // glacier cannot border a dune sea. The other four axes are either
        // terrain-derived (continentalness, erosion — they may change sharply at
        // a ridge) or exist precisely to break the pattern (weirdness), so
        // demanding continuity of them would be demanding the opposite of what
        // they are for.
        float sum = 0.0f;
        for (const procgen::ClimateAxis axis : {procgen::ClimateAxis::Temperature, procgen::ClimateAxis::Moisture})
        {
            const float d = pa->center[axis].toFloat() - pb->center[axis].toFloat();
            sum += d * d;
        }
        return sum;
    };

    const Built world = build(128u, 4242u);

    // Only cells the CLIMATE decided are in scope. Ocean, beach, lake and the
    // bare-summit override are altitude and hydrology, and those are hard
    // thresholds by construction — a shoreline IS a discontinuity, and testing
    // the climate model against it would be measuring the wrong thing.
    procgen::Grid<core::u8> byClimate{world.biomes.width(), world.biomes.depth(), core::u8{0}};
    for (core::u32 z = 0u; z < world.biomes.depth(); ++z)
        for (core::u32 x = 0u; x < world.biomes.width(); ++x)
        {
            math::Fixed32 ignored{};
            byClimate.at(x, z) =
                procgen::nearestBiomeProfile(world.climate.at(x, z), ignored) == world.biomes.at(x, z) ? 1u : 0u;
        }

    float worst = 0.0f;
    procgen::BiomeId worstA = procgen::BiomeId::Count;
    procgen::BiomeId worstB = procgen::BiomeId::Count;

    for (core::u32 z = 0u; z + 1u < world.biomes.depth(); ++z)
    {
        for (core::u32 x = 0u; x + 1u < world.biomes.width(); ++x)
        {
            if (byClimate.at(x, z) == 0u)
                continue;
            const procgen::BiomeId here = world.biomes.at(x, z);
            const bool eastIsClimate = byClimate.at(x + 1u, z) != 0u;
            const bool southIsClimate = byClimate.at(x, z + 1u) != 0u;
            const procgen::BiomeId east = world.biomes.at(x + 1u, z);
            const procgen::BiomeId south = world.biomes.at(x, z + 1u);
            const bool climateNeighbour[2] = {eastIsClimate, southIsClimate};
            const procgen::BiomeId neighbours[2] = {east, south};
            for (core::u32 n = 0u; n < 2u; ++n)
            {
                if (!climateNeighbour[n])
                    continue;
                const procgen::BiomeId other = neighbours[n];
                if (other == here)
                    continue;
                const float d = centreDistance(here, other);
                if (d > worst)
                {
                    worst = d;
                    worstA = here;
                    worstB = other;
                }
            }
        }
    }

    if (worstA != procgen::BiomeId::Count)
        std::printf("    furthest adjacent pair: %s | %s at squared distance %.3f\n", procgen::biomeName(worstA),
                    procgen::biomeName(worstB), worst);

    // Snow (T=0.05) against desert (T=0.85) would be 0.64 on temperature alone.
    // Anything under half that is a transition, not a discontinuity.
    check(worst < 0.32f, "no two adjacent land biomes are opposite ends of the space");
}

void testEndemismAndTreeLine()
{
    std::printf("endemism restricts a species to some regions\n");

    const auto count = [](core::f32 share, core::u32 &outProps) {
        procgen::WorldBuilder builder{2026u};
        builder.terrain(96u, 96u).normalize(-10.0f, 16.0f).regions(16u);

        procgen::ScatterRule rule;
        rule.biome = procgen::BiomeId::Grassland;
        rule.density = 0.20f;
        rule.endemicShare = share;
        builder.scatter(rule);

        ecs::Registry registry;
        const procgen::BuiltWorldStats stats = builder.materialize(registry);
        outProps = stats.propEntities;
    };

    core::u32 everywhere = 0u;
    core::u32 restricted = 0u;
    count(1.0f, everywhere);
    count(0.35f, restricted);
    std::printf("    cosmopolitan %u props, endemic(0.35) %u props\n", everywhere, restricted);

    check(everywhere > 0u, "a cosmopolitan rule places props");
    check(restricted > 0u, "an endemic rule still places props somewhere");
    check(restricted < everywhere, "endemism removes ground the species may use");

    // Same world, same rule, twice: the region draw is keyed to the world seed,
    // so it must not depend on when the rule ran.
    core::u32 again = 0u;
    count(0.35f, again);
    check(again == restricted, "the endemic draw is a property of the world, not of the run");
}

} // namespace

int main()
{
    std::printf("== procgen climate ==\n");
    testAxesAreNormalised();
    testPaletteIsReachable();
    testClimateIsContinuous();
    testEndemismAndTreeLine();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
