#include <lpl/procgen/Voronoi.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(voronoi);

namespace {

/**
 * @brief Two squares rounded to Q16.16 can each err by half a raw word: twice that, in squared raw words.
 */
constexpr lpl::core::u64 kSquareRounding = 1ull << 17;

/**
 * @brief The distance from cell (@p x, @p z) to @p site under @p metric, exact, squared for the Euclidean
 *        metric, in raw words.
 */
[[nodiscard]] lpl::core::u64 exactDistance(lpl::core::u32 x, lpl::core::u32 z, const lpl::procgen::VoronoiSite &site,
                                           lpl::procgen::DistanceMetric metric)
{
    const lpl::core::i64 signedX = static_cast<lpl::core::i64>(x) * 65536 - site.x.raw();
    const lpl::core::i64 signedZ = static_cast<lpl::core::i64>(z) * 65536 - site.z.raw();
    const lpl::core::u64 dx = static_cast<lpl::core::u64>(signedX < 0 ? -signedX : signedX);
    const lpl::core::u64 dz = static_cast<lpl::core::u64>(signedZ < 0 ? -signedZ : signedZ);

    if (metric == lpl::procgen::DistanceMetric::Manhattan)
        return dx + dz;
    if (metric == lpl::procgen::DistanceMetric::Chebyshev)
        return dx > dz ? dx : dz;
    return dx * dx + dz * dz;
}

/**
 * @brief The cells of an unwarped diagram whose region is not their nearest site.
 *
 * @details Every site is compared, so the count does not depend on how far the diagram searched. Under the
 * Euclidean metric a region within the rounding of two squares of the nearest one still counts as the
 * nearest.
 */
[[nodiscard]] lpl::core::u32 cellsAwayFromTheirNearestSite(const lpl::procgen::VoronoiDiagram &diagram,
                                                           lpl::procgen::DistanceMetric metric)
{
    const lpl::core::u64 tolerance = metric == lpl::procgen::DistanceMetric::Euclidean ? kSquareRounding : 0u;
    lpl::core::u32 misplaced = 0u;

    for (lpl::core::u32 z = 0u; z < diagram.regions.depth(); ++z)
    {
        for (lpl::core::u32 x = 0u; x < diagram.regions.width(); ++x)
        {
            const lpl::core::u16 region = diagram.regions.at(x, z);
            if (region >= diagram.sites.size())
            {
                ++misplaced;
                continue;
            }
            const lpl::core::u64 own = exactDistance(x, z, diagram.sites[region], metric);
            lpl::core::u64 nearest = own;
            for (const lpl::procgen::VoronoiSite &site : diagram.sites)
            {
                const lpl::core::u64 distance = exactDistance(x, z, site, metric);
                nearest = distance < nearest ? distance : nearest;
            }
            misplaced += own > nearest + tolerance ? 1u : 0u;
        }
    }
    return misplaced;
}

/**
 * @brief Regions of 200 cells, whose points lie farther from a site than the 181 units at which two
 *        Fixed32 squares overflow.
 */
[[nodiscard]] lpl::procgen::VoronoiParams largeCells()
{
    lpl::procgen::VoronoiParams params;
    params.width = 400u;
    params.depth = 48u;
    params.cellSize = 200u;
    params.jitter = 0.8f;
    params.seed = 1337u;
    return params;
}

} // namespace

/**
 * @brief A cell farther than 181 units from the sites it compares still finds the nearest one.
 *
 * @details The Euclidean distance used to be squared in Fixed32, whose sum of two squares overflows past
 * about 181 units: with regions of 200 cells, 15157 of 19200 cells went to another region.
 */
LPL_TEST(a_large_region_holds_its_nearest_cells)
{
    const lpl::procgen::VoronoiDiagram diagram = lpl::procgen::computeVoronoi(largeCells());

    test.check(diagram.regionCount == 2u, "two regions of 200 cells cover the grid");
    test.check(cellsAwayFromTheirNearestSite(diagram, lpl::procgen::DistanceMetric::Euclidean) == 0u,
               "every cell belongs to its nearest site");
}

/**
 * @brief Under a strong jitter, a site two coarse cells away is searched when it can be the nearest.
 *
 * @details One ring of coarse cells used to be searched whatever the jitter. Under Manhattan a site two
 * cells away can be nearer than every site of the first ring once the jitter passes 1/3: at a jitter of 1,
 * seeds 2, 3 and 1337 each left one or two cells in another region.
 */
LPL_TEST(a_strong_jitter_searches_far_enough)
{
    lpl::core::u32 misplaced = 0u;
    constexpr lpl::core::u32 kSeeds[] = {2u, 3u, 1337u};
    constexpr lpl::procgen::DistanceMetric kMetrics[] = {lpl::procgen::DistanceMetric::Euclidean,
                                                         lpl::procgen::DistanceMetric::Manhattan,
                                                         lpl::procgen::DistanceMetric::Chebyshev};

    for (const lpl::core::u32 seed : kSeeds)
    {
        for (const lpl::procgen::DistanceMetric metric : kMetrics)
        {
            lpl::procgen::VoronoiParams params;
            params.cellSize = 8u;
            params.jitter = 1.0f;
            params.seed = seed;
            params.metric = metric;
            misplaced += cellsAwayFromTheirNearestSite(lpl::procgen::computeVoronoi(params), metric);
        }
    }
    test.check(misplaced == 0u, "at a jitter of 1, every cell belongs to its nearest site under the three metrics");
}

/**
 * @brief The Euclidean distance field saturates where a square no longer fits in Fixed32.
 */
LPL_TEST(the_distance_field_saturates_beyond_the_word)
{
    const lpl::procgen::VoronoiDiagram diagram = lpl::procgen::computeVoronoi(largeCells());
    const lpl::procgen::Grid<lpl::math::Fixed32> field = lpl::procgen::regionDistanceField(diagram);
    lpl::core::u32 negative = 0u;
    lpl::core::u32 saturated = 0u;

    for (lpl::core::u32 i = 0u; i < field.cellCount(); ++i)
    {
        negative += field[i] < lpl::math::Fixed32::zero() ? 1u : 0u;
        saturated += field[i] == lpl::math::Fixed32::max() ? 1u : 0u;
    }
    test.check(negative == 0u, "no squared distance wraps below zero");
    test.check(saturated > 0u, "the cells beyond 181 units hold the highest value");
}
