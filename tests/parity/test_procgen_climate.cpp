#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/Climate.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/Hydrology.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

#include <lpl/ecs/Registry.hpp>

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
 * @brief A varied world: continents, ranges, and enough range for a lapse rate to bite.
 */
lpl::procgen::Heightfield makeTerrain(lpl::core::u32 size, lpl::core::u32 seed)
{
    lpl::procgen::NoiseParams base;
    base.seed = seed;
    base.frequency = 3.0f / static_cast<lpl::core::f32>(size);
    base.amplitude = 20.0f;
    base.octaves = 5u;

    lpl::procgen::Heightfield field = lpl::procgen::generateNoiseHeightfield(size, size, base);

    lpl::procgen::NoiseParams ridges;
    ridges.seed = seed ^ 0x51EDu;
    ridges.frequency = 7.0f / static_cast<lpl::core::f32>(size);
    ridges.amplitude = 9.0f;
    ridges.octaves = 4u;
    ridges.kind = lpl::procgen::NoiseKind::Ridged;
    lpl::procgen::addNoiseLayer(field, ridges);

    lpl::procgen::normalizeHeights(field, lpl::math::Fixed32::fromFloat(-10.0f), lpl::math::Fixed32::fromFloat(16.0f));
    return field;
}

struct Built {
    lpl::procgen::Heightfield terrain;
    lpl::procgen::ClimateField climate;
    lpl::procgen::BiomeMap biomes;
};

Built build(lpl::core::u32 size, lpl::core::u32 seed)
{
    Built out;
    out.terrain = makeTerrain(size, seed);

    lpl::procgen::MoistureParams wet;
    wet.seaLevel = -4.0f;
    const lpl::procgen::DrainageNetwork network = lpl::procgen::computeDrainage(out.terrain);
    const lpl::procgen::Heightfield moisture = lpl::procgen::computeMoisture(out.terrain, network, wet);

    lpl::procgen::ClimateParams axes;
    axes.seaLevel = -4.0f;
    out.climate = lpl::procgen::computeClimate(out.terrain, moisture, network, axes);

    const lpl::procgen::Grid<lpl::core::u8> lakes = lpl::procgen::lakeMask(network, out.terrain);
    out.biomes = lpl::procgen::classifyBiomes(out.terrain, out.climate, lpl::procgen::BiomeParams{}, &lakes);
    return out;
}

/**
 * @brief Lowest, median and highest value of an axis, as floats for printing.
 *
 * The median comes from a coarse histogram: exact enough to catch "everything is crushed into the bottom decile",
 * which is the failure being looked for.
 */
void axisSpread(const lpl::procgen::Heightfield &axis, float &low, float &median, float &high)
{
    lpl::math::Fixed32 lo{};
    lpl::math::Fixed32 hi{};
    (void) lpl::procgen::heightRange(axis, lo, hi);
    low = lo.toFloat();
    high = hi.toFloat();

    lpl::core::u32 buckets[64] = {};
    for (lpl::core::u32 i = 0u; i < axis.cellCount(); ++i)
    {
        lpl::core::i32 b = (axis[i].raw() * 63) >> 16;
        if (b < 0)
            b = 0;
        if (b > 63)
            b = 63;
        ++buckets[b];
    }
    const lpl::core::u32 half = axis.cellCount() / 2u;
    lpl::core::u32 running = 0u;
    median = 0.5f;
    for (lpl::core::u32 b = 0u; b < 64u; ++b)
    {
        running += buckets[b];
        if (running >= half)
        {
            median = static_cast<float>(b) / 63.0f;
            break;
        }
    }
}

const char *axisName(lpl::procgen::ClimateAxis a)
{
    switch (a)
    {
    case lpl::procgen::ClimateAxis::Temperature: return "temperature";
    case lpl::procgen::ClimateAxis::Moisture: return "moisture";
    case lpl::procgen::ClimateAxis::Continentalness: return "continentalness";
    case lpl::procgen::ClimateAxis::Erosion: return "erosion";
    case lpl::procgen::ClimateAxis::Depth: return "depth";
    case lpl::procgen::ClimateAxis::Weirdness: return "weirdness";
    case lpl::procgen::ClimateAxis::Count: break;
    }
    return "?";
}

/**
 * @brief Climate axes are normalised, at every map size.
 *
 * The rank-normalised axis is the one that would betray a min-max scaling slipped back in: its median has to sit near
 * the middle whatever the input's shape, and a heavy tail would drag it far below.
 */
void testAxesAreNormalised()
{
    std::printf("climate axes are normalised, at every map size\n");

    const lpl::core::u32 sizes[] = {24u, 48u, 96u, 128u};
    for (lpl::core::u32 s = 0u; s < 4u; ++s)
    {
        const Built world = build(sizes[s], 1337u);
        check(!world.climate.empty(), "the climate field is built");

        for (lpl::core::u32 a = 0u; a < lpl::procgen::kClimateAxisCount; ++a)
        {
            float low = 0.0f;
            float median = 0.0f;
            float high = 0.0f;
            axisSpread(world.climate.axes[a], low, median, high);

            const bool inRange = low >= -0.001f && high <= 1.001f;
            if (!inRange)
                std::printf("    %ux%u %-16s [%.3f .. %.3f]\n", sizes[s], sizes[s],
                            axisName(static_cast<lpl::procgen::ClimateAxis>(a)), low, high);
            check(inRange, "axis stays inside [0, 1]");
        }

        float low = 0.0f;
        float median = 0.0f;
        float high = 0.0f;
        axisSpread(world.climate[lpl::procgen::ClimateAxis::Erosion], low, median, high);
        std::printf("    %3ux%-3u erosion median = %.3f\n", sizes[s], sizes[s], median);
        check(median > 0.30f && median < 0.70f, "the rank-normalised axis has a centred median");
    }
}

/**
 * @brief The biome palette is reachable.
 *
 * Not "most of them" — all of them. A profile no world can produce is a lie in the table, and the point of counting
 * is to be told about it.
 */
void testPaletteIsReachable()
{
    std::printf("the biome palette is reachable\n");

    lpl::core::u32 everSeen[static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count)] = {};
    for (lpl::core::u32 seed = 0u; seed < 8u; ++seed)
    {
        const Built world = build(128u, 1337u + seed * 7919u);
        lpl::core::u32 counts[static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count)] = {};
        lpl::procgen::countBiomes(world.biomes, counts);
        for (lpl::core::u32 i = 0u; i < static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count); ++i)
            everSeen[i] += counts[i];
    }

    lpl::core::u32 reached = 0u;
    std::printf("    ");
    for (lpl::core::u32 i = 0u; i < static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count); ++i)
    {
        if (everSeen[i] != 0u)
            ++reached;
        else
            std::printf("[MISSING %s] ", lpl::procgen::biomeName(static_cast<lpl::procgen::BiomeId>(i)));
    }
    std::printf("%u/%u biomes reached over 8 seeds\n", reached,
                static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count));

    check(reached == static_cast<lpl::core::u32>(lpl::procgen::BiomeId::Count), "every biome is reachable");
}

/**
 * @brief Climate is continuous: no impossible neighbours.
 *
 * Neighbours are compared by the squared distance between their profiles' centres. Two biomes further apart than the
 * threshold may not touch. A biome with no profile is hydrological (ocean, beach, lake): exempt.
 *
 * Temperature and moisture only, and that is the claim being tested rather than a convenient subset: the guarantee is
 * that a continuous climate has to cross every intermediate isovalue of THOSE two, so a glacier cannot border a dune
 * sea. The other four axes are either terrain-derived (continentalness, erosion — they may change sharply at a ridge)
 * or exist precisely to break the pattern (weirdness), so demanding continuity of them would be demanding the
 * opposite of what they are for.
 *
 * Only cells the CLIMATE decided are in scope. Ocean, beach, lake and the bare-summit override are altitude and
 * hydrology, and those are hard thresholds by construction — a shoreline IS a discontinuity, and testing the climate
 * model against it would be measuring the wrong thing.
 *
 * Snow (T=0.05) against desert (T=0.85) would be 0.64 on temperature alone. Anything under half that is a
 * transition, not a discontinuity.
 */
void testClimateIsContinuous()
{
    std::printf("climate is continuous: no impossible neighbours\n");

    lpl::core::u32 profileCount = 0u;
    const lpl::procgen::BiomeProfile *profiles = lpl::procgen::biomeProfiles(profileCount);

    const auto centreDistance = [&](lpl::procgen::BiomeId a, lpl::procgen::BiomeId b) -> float {
        const lpl::procgen::BiomeProfile *pa = nullptr;
        const lpl::procgen::BiomeProfile *pb = nullptr;
        for (lpl::core::u32 i = 0u; i < profileCount; ++i)
        {
            if (profiles[i].id == a)
                pa = &profiles[i];
            if (profiles[i].id == b)
                pb = &profiles[i];
        }
        if (pa == nullptr || pb == nullptr)
            return 0.0f;
        float sum = 0.0f;
        for (const lpl::procgen::ClimateAxis axis :
             {lpl::procgen::ClimateAxis::Temperature, lpl::procgen::ClimateAxis::Moisture})
        {
            const float d = pa->center[axis].toFloat() - pb->center[axis].toFloat();
            sum += d * d;
        }
        return sum;
    };

    const Built world = build(128u, 4242u);

    lpl::procgen::Grid<lpl::core::u8> byClimate{world.biomes.width(), world.biomes.depth(), lpl::core::u8{0}};
    for (lpl::core::u32 z = 0u; z < world.biomes.depth(); ++z)
        for (lpl::core::u32 x = 0u; x < world.biomes.width(); ++x)
        {
            lpl::math::Fixed32 ignored{};
            byClimate.at(x, z) =
                lpl::procgen::nearestBiomeProfile(world.climate.at(x, z), ignored) == world.biomes.at(x, z) ? 1u : 0u;
        }

    float worst = 0.0f;
    lpl::procgen::BiomeId worstA = lpl::procgen::BiomeId::Count;
    lpl::procgen::BiomeId worstB = lpl::procgen::BiomeId::Count;

    for (lpl::core::u32 z = 0u; z + 1u < world.biomes.depth(); ++z)
    {
        for (lpl::core::u32 x = 0u; x + 1u < world.biomes.width(); ++x)
        {
            if (byClimate.at(x, z) == 0u)
                continue;
            const lpl::procgen::BiomeId here = world.biomes.at(x, z);
            const bool eastIsClimate = byClimate.at(x + 1u, z) != 0u;
            const bool southIsClimate = byClimate.at(x, z + 1u) != 0u;
            const lpl::procgen::BiomeId east = world.biomes.at(x + 1u, z);
            const lpl::procgen::BiomeId south = world.biomes.at(x, z + 1u);
            const bool climateNeighbour[2] = {eastIsClimate, southIsClimate};
            const lpl::procgen::BiomeId neighbours[2] = {east, south};
            for (lpl::core::u32 n = 0u; n < 2u; ++n)
            {
                if (!climateNeighbour[n])
                    continue;
                const lpl::procgen::BiomeId other = neighbours[n];
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

    if (worstA != lpl::procgen::BiomeId::Count)
        std::printf("    furthest adjacent pair: %s | %s at squared distance %.3f\n", lpl::procgen::biomeName(worstA),
                    lpl::procgen::biomeName(worstB), worst);

    check(worst < 0.32f, "no two adjacent land biomes are opposite ends of the space");
}

/**
 * @brief Endemism restricts a species to some regions.
 *
 * Same world, same rule, twice: the region draw is keyed to the world seed, so it must not depend on when the rule
 * ran.
 */
void testEndemismAndTreeLine()
{
    std::printf("endemism restricts a species to some regions\n");

    const auto count = [](lpl::core::f32 share, lpl::core::u32 &outProps) {
        lpl::procgen::WorldBuilder builder{2026u};
        builder.terrain(96u, 96u).normalize(-10.0f, 16.0f).regions(16u);

        lpl::procgen::ScatterRule rule;
        rule.biome = lpl::procgen::BiomeId::Grassland;
        rule.density = 0.20f;
        rule.endemicShare = share;
        builder.scatter(rule);

        lpl::ecs::Registry registry;
        const lpl::procgen::BuiltWorldStats stats = builder.materialize(registry);
        outProps = stats.propEntities;
    };

    lpl::core::u32 everywhere = 0u;
    lpl::core::u32 restricted = 0u;
    count(1.0f, everywhere);
    count(0.35f, restricted);
    std::printf("    cosmopolitan %u props, endemic(0.35) %u props\n", everywhere, restricted);

    check(everywhere > 0u, "a cosmopolitan rule places props");
    check(restricted > 0u, "an endemic rule still places props somewhere");
    check(restricted < everywhere, "endemism removes ground the species may use");

    lpl::core::u32 again = 0u;
    count(0.35f, again);
    check(again == restricted, "the endemic draw is a property of the world, not of the run");
}

} // namespace

/**
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
 */
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
