#include <lpl/procgen/Streaming.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(streaming);

namespace {

/**
 * @brief Three residents outside the release radius of a source at the origin, nearest first.
 */
constexpr lpl::procgen::ChunkCoord kLeaving[] = {
    {3,  0},
    {10, 0},
    {6,  0}
};

/**
 * @brief A radius of two chunks for both generation and release, and room to release one chunk a tick.
 */
[[nodiscard]] lpl::procgen::StreamingParams oneReleaseATick()
{
    lpl::procgen::StreamingParams params;
    params.generateRadius = 2u;
    params.releaseRatio16 = 16u;
    params.maxGeneratePerTick = 0u;
    params.maxReleasePerTick = 1u;
    return params;
}

} // namespace

/**
 * @brief Under its release budget, the plan releases the resident furthest from every source.
 *
 * @details The releases were kept in resident order and the budget dropped the last ones, so the chunk
 * released was whichever came first in the list: here the nearest of the three.
 */
LPL_TEST(the_budget_releases_the_furthest_first)
{
    const lpl::procgen::GenerationSource origin{};
    const lpl::procgen::StreamingPlan plan = lpl::procgen::planStreaming(&origin, 1u, kLeaving, 3u, oneReleaseATick());

    test.check(plan.toRelease.size() == 1u, "one chunk is released");
    test.check(plan.toRelease.size() == 1u && plan.toRelease[0] == kLeaving[1], "and it is the furthest");
}

/**
 * @brief With no source, the plan releases every resident within its budget.
 */
LPL_TEST(no_source_still_keeps_the_budget)
{
    const lpl::procgen::StreamingPlan plan = lpl::procgen::planStreaming(nullptr, 0u, kLeaving, 3u, oneReleaseATick());

    test.check(plan.toRelease.size() == 1u, "one chunk a tick is released when nothing is left to stream around");
}

/**
 * @brief The pool ignores a slot it never created, and counts as recycled only a slot served before.
 */
LPL_TEST(the_pool_counts_what_it_recycles)
{
    lpl::procgen::ChunkPool pool;

    pool.reserve(2u);
    const lpl::core::u32 first = pool.acquire();
    const lpl::core::u32 second = pool.acquire();
    test.check(first != lpl::procgen::ChunkPool::kNoSlot && second != lpl::procgen::ChunkPool::kNoSlot,
               "two fresh slots are served");
    test.check(pool.recycled() == 0u, "and neither is a recycled one");

    pool.release(5u);
    test.check(pool.capacity() == 2u && pool.live() == 2u, "a slot past the pool is ignored");

    pool.release(first);
    test.check(pool.acquire() == first && pool.recycled() == 1u, "a slot released and served again is recycled");
}
